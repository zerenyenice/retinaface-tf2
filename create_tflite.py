import tensorflow as tf

from modules.models import RetinaFaceModel
from modules.utils import load_yaml
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image
import numpy as np
from pathlib import PosixPath

#############
### saving model weights or tf_model
# gdd.download_file_from_google_drive(file_id='16HBH2bpSY3TQ_STryWFe72CIcUzp6GRy',
#                                     dest_path='./checkpoint.zip',
#                                     unzip=True)
#
# cfg = load_yaml('configs/retinaface_mbv2.yaml')
#
# model = RetinaFaceModel(cfg, training=False, iou_th=0.4, score_th=0.5)
#
# checkpoint_dir = 'retinaface_mbv2'
# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#
# tf.saved_model.save(model,'tfmodel_')
#
# model.save_weights('model_weights_2_6.h5')

# download exported saved_model for creating tflite model,
# only this way, we are able to create tflite model without getting error

gdd.download_file_from_google_drive(file_id='1k0Xs9lCPWfQulk0J1CAlcqW93XoSLoZR',
                                   dest_path='./tfmodel.zip',
                                   unzip=True)

tflite_input_image_dims=(288,480)

# Need to create an representative_dataset if you are going to create a fully quantized model,
# for float16 model you don't need to
data_path = 'data/representative'
img_path_list = list(PosixPath(data_path).glob(pattern='*.jpg'))

image_data = []
for img_i in img_path_list:
    tmp = np.array(Image.open(img_i).resize(tflite_input_image_dims)).astype('float32')
    image_data.append(tmp)

image_data = np.stack(image_data,0)

def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((image_data)).repeat().batch(1).take(len(image_data)*3):
    yield [tf.dtypes.cast(data, tf.float32)]



# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model('tfmodel_')
#converter.experimental_enable_resource_variables = True
#converter.experimental_new_converter = True
#converter.experimental_new_quantizer = True

# float16 setup
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]
#tflite_quant_model = converter.convert()

# uint8 setup
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32  # or tf.uint8
converter.inference_output_type = tf.float32  # or tf.uint8
tflite_quant_model = converter.convert()


# Save the model.
with open('litemodel_uin8.tflite', 'wb') as f:
  f.write(tflite_quant_model)

# Test lite model
interpreter = tf.lite.Interpreter(model_path="litemodel_uin8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], image_data[[-6]])
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)