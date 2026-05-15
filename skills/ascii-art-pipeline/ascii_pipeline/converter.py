"""
Pure-Python ASCII and Braille image conversion.
Replaces ascii-image-converter binary with Pillow-based implementation.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterable, Sequence
from PIL import Image


IntensityMap = Sequence[str]  # ordered from dark (low) to light (high)


def _charset_to_ramps(charset: str) -> list[float]:
    """Build intensity ramp positions in [0,1] for each character."""
    n = len(charset)
    # Linear quantile positions: center of each bin
    return [(i + 0.5) / n for i in range(n)]


def _map_intensity(img: Image.Image, charset: str | IntensityMap) -> list[str]:
    """
    Core pixel→glyph mapper.
    Accepts either a charset string or precomputed ramp floats.
    Returns list of lines.
    """
    if isinstance(charset, str):
        ramp = _charset_to_ramps(charset)
        chars = list(charset)
    else:
        ramp = list(charset)
        chars = list(ramp.keys()) if hasattr(ramp, 'keys') else list(charset)

    # img must be grayscale
    if img.mode != "L":
        img = img.convert("L")

    width, height = img.size
    pixels = np.array(img, dtype=np.float32) / 255.0  # normalized 0–1

    # Vectorize the lookup
    n = len(chars)
    indices = np.minimum(np.floor(pixels * n), n - 1).astype(np.uint8)
    # Build char array same shape
    char_arr = np.array(chars, dtype="<U1")
    lines = ["".join(char_arr[indices[y, :]]) for y in range(height)]
    return lines


def convert_to_ascii(
    image: Image.Image | str | Path,
    width: int,
    height: int,
    charset: str,
    *,
    resample=Image.Resampling.LANCZOS,
) -> list[str]:
    """
    Render a grayscale image to ASCII art of exact dimensions.

    Args:
        image: PIL Image or path to image file
        width: Target character width
        height: Target character height
        charset: String of characters ordered dark→light
        resample: Pillow resampling filter for downscaling

    Returns:
        List of height strings, each of length width
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image

    img = img.convert("L").resize((width, height), resample)
    return _map_intensity(img, charset)


def _braille_dither(img: Image.Image, width: int, height: int, dither: bool = True) -> np.ndarray:
    """
    Floyd–Steinberg dither at Braille dot resolution (2×4 per character cell).
    Returns a 2D boolean array of shape (height, width) where True means "raise dot".
    """
    # Character cell is 2 wide × 4 tall dots
    dot_w, dot_h = width * 2, height * 4

    # Resize to dot grid with high-quality filter
    img_resized = img.resize((dot_w, dot_h), Image.Resampling.LANCZOS)

    if dither:
        # Floyd–Steinberg dither → binary (0=black background, 255=white ink)
        img_dithered = img_resized.convert("1", dither=Image.FLOYDSTEINBERG)
    else:
        # Simple threshold at 128
        img_dithered = img_resized.point(lambda p: 0 if p < 128 else 255, "1")

    # Convert to NumPy boolean: True = white pixel (dot raised)
    arr = np.array(img_dithered.getdata(), dtype=bool).reshape((dot_h, dot_w))
    return arr


# Braille Unicode patterns: bit layout inside the 0x2800 cell
# (row, col) → bit (U+2800 base)
_BRAILLE_BITMAP = {
    (0, 0): 0x01,
    (0, 1): 0x08,
    (1, 0): 0x02,
    (1, 1): 0x10,
    (2, 0): 0x04,
    (2, 1): 0x20,
    (3, 0): 0x40,
    (3, 1): 0x80,
}


def convert_to_braille(
    image: Image.Image | str | Path,
    width: int,
    height: int,
    *,
    dither: bool = True,
) -> list[str]:
    """
    Render a grayscale image to Braille ASCII art of exact dimensions.
    Implements Floyd–Steinberg dithering at the Braille dot grid.
    White (foreground) pixels become raised dots; black is blank.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image

    if img.mode != "L":
        img = img.convert("L")

    dots = _braille_dither(img, width, height, dither)

    lines = []
    for y in range(height):
        row_chars = []
        for x in range(width):
            bits = 0
            for dy in range(4):
                for dx in range(2):
                    if dots[y * 4 + dy, x * 2 + dx]:
                        bits |= _BRAILLE_BITMAP[(dy, dx)]
            row_chars.append(chr(0x2800 + bits))
        lines.append("".join(row_chars))
    return lines


def normalize_lines(lines: Iterable[str], width: int, height: int) -> list[str]:
    """Ensure the ASCII art block matches exact grid dimensions."""
    rendered = [str(line).rstrip("\n") for line in lines]
    if len(rendered) < height:
        rendered.extend([" " * width] * (height - len(rendered)))
    rendered = rendered[:height]
    return [(line + " " * width)[:width] for line in rendered]
