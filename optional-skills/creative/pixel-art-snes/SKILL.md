---
name: pixel-art-snes
description: Convert images into SNES-style pixel art with a 32-color palette, Floyd-Steinberg dithering, and 4px block scaling.
version: 1.1.0
author: dodo-reach
license: MIT
metadata:
  hermes:
    tags: [creative, pixel-art, snes, retro, image]
    category: creative
---

# Pixel Art SNES

Convert any image into authentic SNES-style pixel art. This skill uses a 32-color palette with Floyd-Steinberg dithering and 4px block scaling for a cleaner 16-bit console look with more detail retention than arcade-style output.

When needed, you may adjust block size, palette size, and enhancement strength slightly to fit the source image or the user's request, but keep the result unmistakably SNES-style: cleaner, more detailed, and still clearly retro.

## When to Use

- The user wants a classic 16-bit console aesthetic
- The output needs more retained detail than the arcade variant
- The target use case is sprites, characters, or detailed retro illustrations

## Procedure

1. Boost contrast to `1.6x`, color to `1.4x`, and sharpness to `1.2x`.
2. Lightly posterize the image to reduce photographic noise while preserving more detail.
3. Downscale the image to `w // 4` by `h // 4` with `Image.NEAREST`.
4. Quantize the reduced image to 32 colors with Floyd-Steinberg dithering.
5. Upscale back to the original size with `Image.NEAREST`.
6. Save the output as PNG.

## Code

```python
from PIL import Image, ImageEnhance, ImageOps

def pixel_art_snes(input_path, output_path):
    """
    Convert an image to SNES style.

    Args:
        input_path: path to source image
        output_path: path to save the resulting PNG
    """
    img = Image.open(input_path).convert("RGB")

    # Initial boost
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Color(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.2)

    # Lighter posterization preserves more detail while reducing photographic noise
    img = ImageOps.posterize(img, 6)

    w, h = img.size
    small = img.resize((max(1, w // 4), max(1, h // 4)), Image.NEAREST)

    # Quantize after downscaling so dithering is applied at block level
    quantized = small.quantize(colors=32, dither=Image.FLOYDSTEINBERG)
    result = quantized.resize((w, h), Image.NEAREST)

    result.save(output_path, "PNG")
    return result
```

## Example Usage

```python
pixel_art_snes("/path/to/image.jpg", "/path/to/output.png")
```

## Technical Specs

| Parameter | Value |
|-----------|-------|
| Palette | 32 colors |
| Block size | 4px |
| Dithering | Floyd-Steinberg (after downscale) |
| Pre-processing | Light posterization before quantization |
| Resize method | Nearest Neighbor (downscale and upscale) |
| Output format | PNG |

## Result Style

Best for characters, sprites, and detailed illustrations where you want a polished 16-bit console feel and stronger feature retention than the arcade variant.

## Why This Order Works

Floyd-Steinberg dithering distributes quantization error to adjacent pixels. Applying it after downscaling keeps that error diffusion aligned with the reduced pixel grid, so each dithered pixel maps cleanly to a final enlarged block. Quantizing before downscaling can waste the dithering pattern on full-resolution detail that disappears during resize.

A light posterization step before downscaling can improve separation between tonal regions, which helps photographic inputs read more like stylized pixel art instead of simple pixelated photos while still retaining more detail than the arcade variant.

## Pitfalls

- `4px` blocks are still aggressive on small or busy images
- Realistic subjects can become noisy because of the higher color count
- For simpler subjects that need maximum punch, prefer `pixel-art-arcade`

## Verification

The output is correct if:

- A PNG file is created at the output path
- The image shows clear 4px pixel blocks
- Dithering is visible in gradients
- The palette is limited to about 32 colors
- The overall look feels consistent with SNES-era pixel art

## Dependencies

- Python 3
- Pillow

```bash
pip install Pillow
```
