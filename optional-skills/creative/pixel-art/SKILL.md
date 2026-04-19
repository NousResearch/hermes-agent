---
name: pixel-art
description: Convert images into retro pixel art using named presets (arcade, snes) with Floyd-Steinberg dithering. Arcade is bold and chunky; SNES is cleaner with more detail retention.
version: 1.2.0
author: dodo-reach
license: MIT
metadata:
  hermes:
    tags: [creative, pixel-art, arcade, snes, retro, image]
    category: creative
---

# Pixel Art

Convert any image into retro-style pixel art. One function with named presets that select different aesthetics:

- `arcade` — 16-color palette, 8px blocks. Bold, chunky, high-impact. 80s/90s arcade cabinet feel.
- `snes` — 32-color palette, 4px blocks. Cleaner 16-bit console look with more detail retention.

The core pipeline is identical across presets — what changes is palette size, block size, and the strength of contrast/color/posterize pre-processing. All presets use Floyd-Steinberg dithering applied AFTER downscale so error diffusion aligns with the final pixel grid.

## When to Use

- User wants retro pixel art from a source image
- Posters, album covers, social posts, sprites, characters, backgrounds
- Subject can tolerate aggressive simplification (arcade) or benefits from retained detail (snes)

## Preset Picker

| Preset | Palette | Block | Best for |
|--------|---------|-------|----------|
| `arcade` | 16 colors | 8px | Posters, hero images, bold covers, simple subjects |
| `snes` | 32 colors | 4px | Characters, sprites, detailed illustrations, photos |

Default is `arcade` for maximum stylistic punch. Switch to `snes` when the subject has detail worth preserving.

## Procedure

1. Pick a preset (`arcade` or `snes`) based on the aesthetic you want.
2. Boost contrast, color, and sharpness using the preset's enhancement values.
3. Lightly posterize the image to simplify tonal regions before quantization.
4. Downscale to `w // block` by `h // block` with `Image.NEAREST`.
5. Quantize the reduced image to the preset's palette size with Floyd-Steinberg dithering.
6. Upscale back to the original size with `Image.NEAREST`.
7. Save the output as PNG.

## Code

```python
from PIL import Image, ImageEnhance, ImageOps

PRESETS = {
    "arcade": {
        "contrast": 1.8,
        "color": 1.5,
        "sharpness": 1.2,
        "posterize_bits": 5,
        "block": 8,
        "palette": 16,
    },
    "snes": {
        "contrast": 1.6,
        "color": 1.4,
        "sharpness": 1.2,
        "posterize_bits": 6,
        "block": 4,
        "palette": 32,
    },
}


def pixel_art(input_path, output_path, preset="arcade", **overrides):
    """
    Convert an image to retro pixel art.

    Args:
        input_path: path to source image
        output_path: path to save the resulting PNG
        preset: "arcade" or "snes"
        **overrides: optionally override any preset field
                     (contrast, color, sharpness, posterize_bits, block, palette)

    Returns:
        The resulting PIL.Image.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose from: {sorted(PRESETS)}"
        )

    cfg = {**PRESETS[preset], **overrides}

    img = Image.open(input_path).convert("RGB")

    # Stylistic boost — stronger for smaller palettes
    img = ImageEnhance.Contrast(img).enhance(cfg["contrast"])
    img = ImageEnhance.Color(img).enhance(cfg["color"])
    img = ImageEnhance.Sharpness(img).enhance(cfg["sharpness"])

    # Light posterization separates tonal regions before quantization
    img = ImageOps.posterize(img, cfg["posterize_bits"])

    w, h = img.size
    block = cfg["block"]
    small = img.resize(
        (max(1, w // block), max(1, h // block)),
        Image.NEAREST,
    )

    # Quantize AFTER downscaling so dithering aligns with the final pixel grid
    quantized = small.quantize(
        colors=cfg["palette"], dither=Image.FLOYDSTEINBERG
    )
    result = quantized.resize((w, h), Image.NEAREST)

    result.save(output_path, "PNG")
    return result
```

## Example Usage

```python
# Bold arcade look (default)
pixel_art("/path/to/image.jpg", "/path/to/arcade.png")

# Cleaner SNES look with more detail
pixel_art("/path/to/image.jpg", "/path/to/snes.png", preset="snes")

# Override individual parameters — e.g. tighter palette with SNES block size
pixel_art(
    "/path/to/image.jpg",
    "/path/to/custom.png",
    preset="snes",
    palette=16,
)
```

## Why This Order Works

Floyd-Steinberg dithering distributes quantization error to adjacent pixels. Applying it AFTER downscaling keeps that error diffusion aligned with the reduced pixel grid, so each dithered pixel maps cleanly to a final enlarged block. Quantizing before downscaling wastes the dithering pattern on full-resolution detail that disappears during resize.

A light posterization step before downscaling improves separation between tonal regions, which helps photographic inputs read as stylized pixel art instead of simple pixelated photos.

Stronger pre-processing (higher contrast/color) pairs with smaller palettes because fewer colors have to carry the whole image. SNES runs softer enhancements because 32 colors can represent gradients and mid-tones directly.

## Pitfalls

- `arcade` 8px blocks are aggressive and can destroy fine detail — use `snes` for subjects that need retention
- Busy photographs can become noisy under `snes` because the larger palette preserves small variations — use `arcade` to flatten them
- Very small source images (<~100px wide) may collapse under 8px blocks. `max(1, w // block)` guards against zero dimensions, but output will be visually degenerate.
- Fractional overrides for `block` or `palette` will break quantization — keep them as positive integers.

## Verification

Output is correct if:

- A PNG file is created at the output path
- The image shows clear square pixel blocks at the preset's block size
- Dithering is visible in gradients
- The palette is limited to approximately the preset's color count
- The overall look matches the targeted era (arcade or SNES)

## Dependencies

- Python 3
- Pillow

```bash
pip install Pillow
```
