---
name: pixel-art-arcade
description: Convert images into bold arcade-era pixel art with a 16-color palette, Floyd-Steinberg dithering, and 8px block scaling.
version: 1.1.0
author: dodo-reach
license: MIT
metadata:
  hermes:
    tags: [creative, pixel-art, arcade, retro, image]
    category: creative
---

# Pixel Art Arcade

Convert any image into authentic 80s/90s arcade cabinet pixel art. This skill uses a 16-color palette with Floyd-Steinberg dithering and 8px block scaling for a bold, high-impact retro look.

When needed, you may adjust block size, palette size, and enhancement strength slightly to fit the source image or the user's request, but keep the result unmistakably arcade-style: bold, chunky, and high-impact.

## When to Use

- The user wants pixel art with maximum visual impact
- A retro arcade aesthetic fits posters, covers, social posts, sprites, or backgrounds
- The subject can tolerate aggressive simplification and chunky 8px blocks

## Procedure

1. Boost contrast to `1.8x`, color to `1.5x`, and sharpness to `1.2x`.
2. Lightly posterize the image to simplify tonal regions before quantization.
3. Downscale the image to `w // 8` by `h // 8` with `Image.NEAREST`.
4. Quantize the reduced image to 16 colors with Floyd-Steinberg dithering.
5. Upscale back to the original size with `Image.NEAREST`.
6. Save the output as PNG.

## Code

```python
from PIL import Image, ImageEnhance, ImageOps

def pixel_art_arcade(input_path, output_path):
    """
    Convert an image to arcade cabinet style.

    Args:
        input_path: path to source image
        output_path: path to save the resulting PNG
    """
    img = Image.open(input_path).convert("RGB")

    # Initial boost for heavily limited palette
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Color(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(1.2)

    # Light posterization helps separate tonal regions before quantization
    img = ImageOps.posterize(img, 5)

    w, h = img.size
    small = img.resize((max(1, w // 8), max(1, h // 8)), Image.NEAREST)

    # Quantize after downscaling so dithering is applied at block level
    quantized = small.quantize(colors=16, dither=Image.FLOYDSTEINBERG)
    result = quantized.resize((w, h), Image.NEAREST)

    result.save(output_path, "PNG")
    return result
```

## Example Usage

```python
pixel_art_arcade("/path/to/image.jpg", "/path/to/output.png")
```

## Technical Specs

| Parameter | Value |
|-----------|-------|
| Palette | 16 colors |
| Block size | 8px |
| Dithering | Floyd-Steinberg (after downscale) |
| Pre-processing | Light posterization before quantization |
| Resize method | Nearest Neighbor (downscale and upscale) |
| Output format | PNG |

## Result Style

Best for posters, album covers, bold hero images, and other cases where you want the feeling of an arcade cabinet screen glowing in a dark room.

## Why This Order Works

Floyd-Steinberg dithering distributes quantization error to adjacent pixels. Applying it after downscaling keeps that error diffusion aligned with the reduced pixel grid, so each dithered pixel maps cleanly to a final enlarged block. Quantizing before downscaling can waste the dithering pattern on full-resolution detail that disappears during resize.

A light posterization step before downscaling can improve separation between tonal regions, which helps photographic inputs read more like stylized pixel art instead of simple pixelated photos.

## Pitfalls

- `8px` blocks are aggressive and can destroy fine detail
- Highly detailed photographs may simplify too much
- For softer, more detailed retro output, prefer `pixel-art-snes`

## Verification

The output is correct if:

- A PNG file is created at the output path
- The image shows clear 8px pixel blocks
- Dithering is visible in gradients
- The palette is limited to about 16 colors
- The overall look feels consistent with arcade-era pixel art

## Dependencies

- Python 3
- Pillow

```bash
pip install Pillow
```
