
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path

print("Loading pretrained AI model...")

# Load the pretrained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

print("Model loaded successfully!")

# Get the image path from the user
image_path = input(
    "\nEnter the image file path: "
).strip().strip('"')

# Check whether the image exists
if not Path(image_path).is_file():
    print("Error: Image file not found. Please check the path.")

else:
    try:
        # Open and prepare the image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        # Convert the image into a NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Preprocess the image for MobileNetV2
        image_array = (
            tf.keras.applications.mobilenet_v2
            .preprocess_input(image_array)
        )

        # Predict the image class
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get the top 3 predictions
        top_predictions = (
            tf.keras.applications.mobilenet_v2
            .decode_predictions(predictions, top=3)[0]
        )

        # Display the results
        print("\n========== CLASSIFICATION RESULTS ==========")
        print("Image:", Path(image_path).name)

        for rank, (class_id, label, score) in enumerate(
            top_predictions, start=1
        ):
            readable_label = label.replace("_", " ").title()

            print(
                f"{rank}. {readable_label}: "
                f"{score * 100:.2f}%"
            )

        print("============================================")

    except Exception as error:
        print("\nCould not classify the image.")
        print("Error:", error)
