#!/usr/bin/env python3
"""Caption a local image with meme-style text.

This script is intentionally narrow:
- it takes an already-downloaded image
- it adds captions
- it can trim white padding before rendering the caption

Usage:
    python meme_caption.py input.png output.png "TOP TEXT" "BOTTOM TEXT"
    python meme_caption.py --trim-padding input.png output.png "TOP TEXT" "BOTTOM TEXT"
    python meme_caption.py --bars input.png output.png "TOP TEXT" "BOTTOM TEXT"
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image

import generate_meme

DEFAULT_PADDING_THRESHOLD = 248
DEFAULT_PADDING_MARGIN = 8


def _trim_padding(image: Image.Image, threshold: int = DEFAULT_PADDING_THRESHOLD, margin: int = DEFAULT_PADDING_MARGIN) -> Image.Image:
    """Trim near-white padding from the image while keeping a small margin."""
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size

    left = width
    top = height
    right = -1
    bottom = -1

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            if r >= threshold and g >= threshold and b >= threshold:
                continue
            left = min(left, x)
            top = min(top, y)
            right = max(right, x)
            bottom = max(bottom, y)

    if right < left or bottom < top:
        return rgba

    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width - 1, right + margin)
    bottom = min(height - 1, bottom + margin)
    return rgba.crop((left, top, right + 1, bottom + 1))


def caption_image(
    input_path: str | Path,
    output_path: str | Path,
    top_text: str,
    bottom_text: str = "",
    *,
    trim_padding: bool = False,
    bars: bool = False,
) -> str:
    """Caption a local image and save the result."""
    image_path = Path(input_path)
    output = Path(output_path)
    texts = [text for text in [top_text, bottom_text] if text]

    if trim_padding:
        with Image.open(image_path) as original:
            trimmed = _trim_padding(original)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = Path(tmp.name)
            try:
                trimmed.save(temp_path)
                return generate_meme.generate_from_image(str(temp_path), texts, str(output), use_bars=bars)
            finally:
                temp_path.unlink(missing_ok=True)

    return generate_meme.generate_from_image(str(image_path), texts, str(output), use_bars=bars)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caption a local image with meme-style text.")
    parser.add_argument("--trim-padding", action="store_true", help="Trim white padding before captioning")
    parser.add_argument("--bars", action="store_true", help="Render text in black bars above and below the image")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to the output image")
    parser.add_argument("top_text", help="Top caption")
    parser.add_argument("bottom_text", nargs="?", default="", help="Bottom caption")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = caption_image(
        args.input,
        args.output,
        args.top_text,
        args.bottom_text,
        trim_padding=args.trim_padding,
        bars=args.bars,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
