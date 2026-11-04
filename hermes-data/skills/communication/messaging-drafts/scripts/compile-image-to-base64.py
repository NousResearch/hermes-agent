#!/usr/bin/env python3
"""
Compress an image and produce a base64 data URI for inline email embedding.
Used when the user says "encode the image as base64 in the HTML".

Usage:
    python3 compile-image-to-base64.py <input_image> [--width=800] [--quality=85]

Output:
    Prints the data:image/jpeg;base64,... string to stdout.
    Saves to <input>_b64.txt as a side effect.
"""
import sys, base64, os
from PIL import Image

args = sys.argv[1:]
if not args:
    print(__doc__, file=sys.stderr)
    sys.exit(1)

src = args[0]
width = 800
quality = 85
for a in args[1:]:
    if a.startswith("--width="): width = int(a.split("=")[1])
    if a.startswith("--quality="): quality = int(a.split("=")[1])

img = Image.open(src).convert("RGB")
w, h = img.size
if w > width:
    img = img.resize((width, int(h * width / w)), Image.LANCZOS)

dst = src + ".compressed.jpg"
img.save(dst, "JPEG", quality=quality, optimize=True)
print(f"Compressed: {os.path.getsize(dst)//1024} KB (was {os.path.getsize(src)//1024} KB)", file=sys.stderr)

with open(dst, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

data_uri = f"data:image/jpeg;base64,{b64}"
out_path = src + "_b64.txt"
with open(out_path, "w") as f:
    f.write(data_uri)
print(f"Base64 saved: {out_path} ({len(b64)//1024} KB text)", file=sys.stderr)
print(data_uri)