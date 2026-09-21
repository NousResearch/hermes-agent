"""Deterministic, brand-tuned image finishing pass.

Turns a clean AI-generated base image into a "designed", on-brand artifact.
The effects are standard image-processing techniques (grade, grain, dither,
vignette, chromatic aberration, halftone, palette grade) reimplemented here
and driven per brand by brand_style.BRAND_VISUALS[brand]["treatment"].

This is the quality lever: the raw generation is never the final deliverable.

CLI:
  python3 postprocess.py in.png out.png --brand plenishd [--intensity 0.8]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from brand_style import for_brand


# ── individual effects (each takes/returns a PIL RGB image) ──────────────

def _contrast(img: Image.Image, s: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(1.0 + s)


def _saturate(img: Image.Image, s: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(1.0 + s)


def _desaturate(img: Image.Image, s: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(max(0.0, 1.0 - s))


def _warm_grade(img: Image.Image, s: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    lum = (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114) / 255.0
    shadow = (1.0 - lum)[..., None]
    arr[..., 0] += shadow[..., 0] * s * 14
    arr[..., 2] -= shadow[..., 0] * s * 8
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _cool_grade(img: Image.Image, s: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    lum = (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114) / 255.0
    shadow = (1.0 - lum)
    arr[..., 2] += shadow * s * 12
    arr[..., 0] -= shadow * s * 6
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _film_grain(img: Image.Image, s: float) -> Image.Image:
    h, w = img.height, img.width
    noise = np.random.normal(0, 255 * 0.05 * s, (h, w, 1)).astype(np.float32)
    arr = np.asarray(img, dtype=np.float32) + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _vignette(img: Image.Image, s: float) -> Image.Image:
    w, h = img.size
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    dist = np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2)
    mask = np.clip(1.0 - dist * s, 1.0 - s, 1.0)[..., None]
    arr = np.asarray(img, dtype=np.float32) * mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _chromatic(img: Image.Image, s: float) -> Image.Image:
    px = max(1, int(round(2.0 * s)))
    arr = np.asarray(img)
    out = arr.copy()
    out[..., 0] = np.roll(arr[..., 0], -px, axis=1)
    out[..., 2] = np.roll(arr[..., 2], px, axis=1)
    return Image.fromarray(out)


def _scanlines(img: Image.Image, s: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr[::2, :, :] *= (1.0 - 0.25 * s)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _paper_texture(img: Image.Image, s: float) -> Image.Image:
    w, h = img.size
    base = np.random.normal(0, 1, (max(1, h // 4), max(1, w // 4))).astype(np.float32)
    rng = max(float(base.max() - base.min()), 1e-6)
    tex = Image.fromarray(((base - base.min()) / rng * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    n = (np.asarray(tex, dtype=np.float32) / 255.0 - 0.5)[..., None]
    arr = np.asarray(img, dtype=np.float32) * (1.0 + n * 0.3 * s)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _halftone(img: Image.Image, s: float) -> Image.Image:
    w, h = img.size
    dot = 3
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = xx % dot, yy % dot
    d = np.sqrt((cx - dot / 2) ** 2 + (cy - dot / 2) ** 2)
    lum = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    mask = np.clip(1.0 - d / (dot * 0.7), 0, 1) * (1.0 - lum) * s
    arr = np.asarray(img, dtype=np.float32) * (1.0 - mask[..., None] * 0.18)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _palette_grade(img: Image.Image, palette_hex: list[str], s: float) -> Image.Image:
    """Gentle pull toward the brand ink palette. Kept low for photographic bases."""
    pal = Image.new("P", (1, 1))
    colours: list[int] = []
    for hx in palette_hex:
        hx = hx.lstrip("#")
        colours += [int(hx[i:i + 2], 16) for i in (0, 2, 4)]
    colours += [0] * (768 - len(colours))
    pal.putpalette(colours)
    q = img.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")
    return Image.blend(img, q, float(np.clip(s, 0, 1)))


# effect-name -> callable(img, strength). Order below is the pipeline order.
_PIPELINE = [
    ("contrast", _contrast),
    ("saturate", _saturate),
    ("desaturate", _desaturate),
    ("warm_grade", _warm_grade),
    ("cool_grade", _cool_grade),
    ("chromatic", _chromatic),
    ("halftone", _halftone),
    ("scanlines", _scanlines),
    ("paper_texture", _paper_texture),
    ("film_grain", _film_grain),
    ("vignette", _vignette),
    ("palette", None),  # special-cased (needs palette list)
]


def apply(img: Image.Image, brand: str, intensity: float = 1.0) -> Image.Image:
    """Apply the brand's treatment. ``intensity`` globally scales every effect."""
    visuals = for_brand(brand)
    treatment = visuals.get("treatment", {})
    img = img.convert("RGB")
    for name, fn in _PIPELINE:
        if name == "palette":
            strength = treatment.get("palette", 0.0) * intensity
            if strength > 0:
                img = _palette_grade(img, visuals["palette"], strength)
            continue
        strength = treatment.get(name, 0.0) * intensity
        if strength > 0:
            img = fn(img, strength)
    return img


def process_file(input_path: str, output_path: str, brand: str, intensity: float = 1.0) -> str:
    img = Image.open(input_path)
    out = apply(img, brand, intensity)
    out.save(output_path, "PNG")
    return output_path


def finish_imagery(img: Image.Image, light: bool = False, intensity: float = 1.0) -> Image.Image:
    """Mandatory analog finish for reference-anchored imagery (palette-agnostic).

    The standard requires every generated image to carry analog craft so it never
    reads as "too clean". Dark palettes get scanline + grain + vignette; light
    palettes (e.g. warm-editorial) get paper texture + grain instead, so the look
    stays appropriate. See brand_imagery_standard.md section 6.
    """
    img = img.convert("RGB")
    img = _contrast(img, 1.10 * intensity)
    img = _halftone(img, 0.22 * intensity)
    if light:
        img = _paper_texture(img, 0.5 * intensity)
        img = _film_grain(img, 0.5 * intensity)
    else:
        img = _scanlines(img, 0.25 * intensity)
        img = _film_grain(img, 0.7 * intensity)
        img = _vignette(img, 0.4 * intensity)
    return img


def finish_file(input_path: str, output_path: str, light: bool = False,
                intensity: float = 1.0) -> str:
    out = finish_imagery(Image.open(input_path), light=light, intensity=intensity)
    out.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--brand", required=True)
    ap.add_argument("--intensity", type=float, default=1.0)
    args = ap.parse_args()
    out = args.output or args.input.replace(".png", "-treated.png")
    process_file(args.input, out, args.brand, args.intensity)
    print(f"treated -> {out}  ({os.path.getsize(out)//1024}KB)")
