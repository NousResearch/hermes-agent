"""Pixel art converter — Floyd-Steinberg dithering with preset or named palette.

Named hardware palettes (NES, GameBoy, PICO-8, C64, etc.) ported from
pixel-art-studio (MIT) — see ATTRIBUTION.md.

Usage (import):
    from pixel_art import pixel_art
    pixel_art("in.png", "out.png", preset="arcade")
    pixel_art("in.png", "out.png", preset="nes")
    pixel_art("in.png", "out.png", palette="PICO_8", block=6)

Usage (CLI):
    python pixel_art.py in.png out.png --preset nes
"""

from PIL import Image, ImageEnhance, ImageOps

try:
    from .palettes import PALETTES, build_palette_image
except ImportError:
    from palettes import PALETTES, build_palette_image


PRESETS = {
    # ── Original presets (adaptive palette) ─────────────────────────────
    "arcade": {
        "contrast": 1.8, "color": 1.5, "sharpness": 1.2,
        "posterize_bits": 5, "block": 8, "palette": 16,
    },
    "snes": {
        "contrast": 1.6, "color": 1.4, "sharpness": 1.2,
        "posterize_bits": 6, "block": 4, "palette": 32,
    },
    # ── Hardware-accurate presets (named palette) ───────────────────────
    "nes": {
        "contrast": 1.5, "color": 1.4, "sharpness": 1.2,
        "posterize_bits": 6, "block": 8, "palette": "NES",
    },
    "gameboy": {
        "contrast": 1.5, "color": 1.0, "sharpness": 1.2,
        "posterize_bits": 6, "block": 8, "palette": "GAMEBOY_ORIGINAL",
    },
    "gameboy_pocket": {
        "contrast": 1.5, "color": 1.0, "sharpness": 1.2,
        "posterize_bits": 6, "block": 8, "palette": "GAMEBOY_POCKET",
    },
    "pico8": {
        "contrast": 1.6, "color": 1.3, "sharpness": 1.2,
        "posterize_bits": 6, "block": 6, "palette": "PICO_8",
    },
    "c64": {
        "contrast": 1.6, "color": 1.3, "sharpness": 1.2,
        "posterize_bits": 6, "block": 8, "palette": "C64",
    },
    "apple2": {
        "contrast": 1.8, "color": 1.4, "sharpness": 1.2,
        "posterize_bits": 5, "block": 10, "palette": "APPLE_II_HI",
    },
    "teletext": {
        "contrast": 1.8, "color": 1.5, "sharpness": 1.2,
        "posterize_bits": 5, "block": 10, "palette": "TELETEXT",
    },
    "mspaint": {
        "contrast": 1.6, "color": 1.4, "sharpness": 1.2,
        "posterize_bits": 6, "block": 8, "palette": "MICROSOFT_WINDOWS_PAINT",
    },
    "mono_green": {
        "contrast": 1.8, "color": 0.0, "sharpness": 1.2,
        "posterize_bits": 5, "block": 6, "palette": "MONO_GREEN",
    },
    "mono_amber": {
        "contrast": 1.8, "color": 0.0, "sharpness": 1.2,
        "posterize_bits": 5, "block": 6, "palette": "MONO_AMBER",
    },
    # ── Artistic palette presets ────────────────────────────────────────
    "neon": {
        "contrast": 1.8, "color": 1.6, "sharpness": 1.2,
        "posterize_bits": 5, "block": 6, "palette": "NEON_CYBER",
    },
    "pastel": {
        "contrast": 1.2, "color": 1.3, "sharpness": 1.1,
        "posterize_bits": 6, "block": 6, "palette": "PASTEL_DREAM",
    },
}


def pixel_art(input_path, output_path, preset="arcade", **overrides):
    """Convert an image to retro pixel art.

    Args:
        input_path: path to source image
        output_path: path to save the resulting PNG
        preset: one of PRESETS (arcade, snes, nes, gameboy, pico8, c64, ...)
        **overrides: optionally override any preset field. In particular:
            palette: int (adaptive N colors) OR str (named palette from PALETTES)
            block:   int pixel block size
            contrast / color / sharpness / posterize_bits: numeric enhancers

    Returns:
        The resulting PIL.Image.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose from: {sorted(PRESETS)}"
        )
    cfg = {**PRESETS[preset], **overrides}

    img = Image.open(input_path).convert("RGB")

    img = ImageEnhance.Contrast(img).enhance(cfg["contrast"])
    img = ImageEnhance.Color(img).enhance(cfg["color"])
    img = ImageEnhance.Sharpness(img).enhance(cfg["sharpness"])
    img = ImageOps.posterize(img, cfg["posterize_bits"])

    w, h = img.size
    block = cfg["block"]
    small = img.resize(
        (max(1, w // block), max(1, h // block)),
        Image.NEAREST,
    )

    # Quantize AFTER downscale so Floyd-Steinberg aligns with final pixel grid.
    pal = cfg["palette"]
    if isinstance(pal, str):
        # Named hardware/artistic palette
        pal_img = build_palette_image(pal)
        quantized = small.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
    else:
        # Adaptive N-color palette (original behavior)
        quantized = small.quantize(colors=int(pal), dither=Image.FLOYDSTEINBERG)

    result = quantized.resize((w, h), Image.NEAREST)
    result.save(output_path, "PNG")
    return result


def main():
    import argparse
    p = argparse.ArgumentParser(description="Convert image to pixel art.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--preset", default="arcade", choices=sorted(PRESETS))
    p.add_argument("--palette", default=None,
                   help=f"Override palette: int or name from {sorted(PALETTES)}")
    p.add_argument("--block", type=int, default=None)
    args = p.parse_args()

    overrides = {}
    if args.palette is not None:
        try:
            overrides["palette"] = int(args.palette)
        except ValueError:
            overrides["palette"] = args.palette
    if args.block is not None:
        overrides["block"] = args.block

    pixel_art(args.input, args.output, preset=args.preset, **overrides)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
