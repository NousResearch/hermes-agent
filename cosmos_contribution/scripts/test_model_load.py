import torch
import os

model_path = r"D:\Cosmos\cosmos\checkpoints\cosmos\cosmos_best.pt"
print(f"Checking {model_path}...")
print(f"File size: {os.path.getsize(model_path)} bytes")

try:
    # Try loading just the header/meta if possible, or a partial load
    # To avoid memory issues, we just try to see if torch.load recognizes the format
    data = torch.load(model_path, map_location='cpu', weights_only=True)
    print("SUCCESS: Torch successfully identified and loaded the file header.")
    print(f"Keys found: {list(data.keys())[:5]}...")
except Exception as e:
    print(f"FAILURE: {e}")
