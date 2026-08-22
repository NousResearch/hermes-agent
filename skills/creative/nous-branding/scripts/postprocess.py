#!/usr/bin/env python3
"""
postprocess.py — Nous/Hermes analog branding overlay.
Adds: warm grade, CRT scanlines, film grain, Risomorphism dither,
      vignette, chromatic aberration, screen-print bleed, paper texture.

v9 adds imprint mode for the newer reference target: constrained inks,
xerox decay, registration offset, scuffs, and technical-art print texture.

Usage:
  python3 postprocess.py input.png [output.png] [--intensity 0.5] [--mode standard|risograph|nous|imprint]
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

PALETTE_HEX = [
    "#F7EDE3", "#E8E0D4", "#C15811", "#F59E0B",
    "#0E2723", "#1A3A32", "#00AEEF", "#8B5CF6",
    "#6B7280", "#2D5016", "#0A0A1A", "#D946EF",
    "#BFE8F2", "#071616", "#F04A23", "#D8D061",
]


def warm_grade(img, s=0.15):
    arr = np.array(img).astype(np.float32)
    lum = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114) / 255
    shadow = 1.0 - lum
    arr[:, :, 0] += shadow * s * 12
    arr[:, :, 2] -= shadow * s * 6
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def crt_scanlines(img, opacity=0.06, every=2):
    w, h = img.size
    overlay = Image.new("L", (w, h), 255)
    px = overlay.load()
    for y in range(h):
        if y % every == 0:
            for x in range(w):
                px[x, y] = int(255 * (1 - opacity))
    return Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), overlay)


def film_grain(img, intensity=0.04):
    w, h = img.size
    noise = np.random.normal(0, 255 * intensity, (h, w, 3)).astype(np.float32)
    return Image.fromarray(np.clip(np.array(img).astype(np.float32) + noise, 0, 255).astype(np.uint8))


def bayer_dither(img, strength=0.3, levels=12):
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32)
    bayer_n = (bayer / 16.0 - 0.5) * strength
    w, h = img.size
    tile = np.tile(bayer_n, ((h + 3) // 4, (w + 3) // 4))[:h, :w]
    tile3 = np.stack([tile] * 3, axis=-1) * 255
    arr = np.array(img).astype(np.float32)
    q = np.round(arr / (256 / levels)) * (256 / levels)
    d = q + tile3 * (256 / levels) / levels
    return Image.fromarray(np.clip(d, 0, 255).astype(np.uint8))


def vignette(img, strength=0.35):
    w, h = img.size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx**2 + yy**2)
    mask = np.clip(1.0 - (dist / np.sqrt(2)) * strength, 0.3, 1.0)
    arr = np.array(img).astype(np.float32)
    m3 = np.stack([mask] * 3, axis=-1)
    v = arr * m3
    edge = (1 - mask) * 8
    v[:, :, 0] += edge
    v[:, :, 2] -= edge * 0.5
    return Image.fromarray(np.clip(v, 0, 255).astype(np.uint8))


def chromatic_aberration(img, shift=1.0):
    arr = np.array(img)
    px = max(1, int(round(shift)))
    r = np.roll(arr[:, :, 0], -px, axis=1)
    b = np.roll(arr[:, :, 2], px, axis=1)
    out = arr.copy()
    out[:, :, 0] = r
    out[:, :, 2] = b
    return Image.fromarray(out)


def screen_print_texture(img, intensity=0.3):
    """Simulate risograph/screen-print: visible halftone dots + slight ink spread."""
    w, h = img.size
    dot_size = 3
    yy, xx = np.mgrid[0:h, 0:w]
    offset = (np.arange(h) // dot_size % 2) * (dot_size // 2)
    cx = (xx + offset[:, None]) % dot_size
    cy = yy % dot_size
    dist = np.sqrt((cx - dot_size / 2) ** 2 + (cy - dot_size / 2) ** 2)
    lum = np.array(img.convert("L")).astype(float) / 255.0
    dot_mask = np.clip(1.0 - dist / (dot_size * 0.7), 0, 1)
    dot_mask = dot_mask * (1.0 - lum) * intensity
    dot_3 = np.stack([dot_mask] * 3, axis=-1)
    arr = np.array(img).astype(np.float32)
    textured = arr * (1.0 - dot_3 * 0.15)
    return Image.fromarray(np.clip(textured, 0, 255).astype(np.uint8))


def paper_texture(img, intensity=0.15):
    """Add subtle paper/canvas fiber texture."""
    w, h = img.size
    noise = np.random.normal(0, 1, (max(1, h // 4), max(1, w // 4))).astype(np.float32)
    denom = max(float(noise.max() - noise.min()), 1e-6)
    n_img = Image.fromarray(((noise - noise.min()) / denom * 255).astype(np.uint8))
    n_img = n_img.resize((w, h), Image.Resampling.BILINEAR)
    n_arr = np.array(n_img).astype(float) / 255.0
    n_3 = np.stack([n_arr] * 3, axis=-1)
    arr = np.array(img).astype(float)
    textured = arr * (1.0 + (n_3 - 0.5) * intensity)
    return Image.fromarray(np.clip(textured, 0, 255).astype(np.uint8))


def ink_bleed(img, intensity=0.2):
    """Slight blur + darken at edges to simulate ink spread."""
    blurred = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr_orig = np.array(img).astype(float)
    arr_blur = np.array(blurred).astype(float)
    edges_img = img.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=0.6))
    edges = np.array(edges_img).astype(float)
    if edges.max() > 0:
        edges = edges / edges.max()
    edge_3 = np.stack([edges] * 3, axis=-1)
    blended = arr_orig * (1 - edge_3 * intensity) + arr_blur * (edge_3 * intensity)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def palette_compress(img, strength=0.45):
    """Pull clean RGB renders toward a limited 2-4 ink print palette."""
    palette = Image.new("P", (1, 1))
    colors = []
    for hx in PALETTE_HEX:
        hx = hx.lstrip("#")
        colors.extend([int(hx[i:i + 2], 16) for i in (0, 2, 4)])
    colors.extend([0] * (768 - len(colors)))
    palette.putpalette(colors)
    quantized = img.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")
    return Image.blend(img, quantized, float(np.clip(strength, 0, 1)))


def xerox_threshold(img, strength=0.25):
    """Blend in degraded xerox contrast without destroying all midtones."""
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.0 + 2.4 * strength)
    arr = np.array(gray).astype(np.float32)
    noise = np.random.normal(0, 24 * strength, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    poster = np.where(arr > (128 - 18 * strength), 235, 20).astype(np.uint8)
    tint = ImageOps.colorize(Image.fromarray(poster), black="#071616", white="#DCEAF0")
    return Image.blend(img, tint, float(np.clip(strength * 0.55, 0, 0.45)))


def registration_offset(img, shift=1.0, opacity=0.35):
    """Simulate misregistered cyan/orange ink plates."""
    px = max(1, int(round(shift)))
    op = float(np.clip(opacity, 0, 1))
    arr = np.array(img).astype(np.float32)
    cyan = np.roll(arr, -px, axis=1)
    orange = np.roll(arr, px, axis=0)
    out = arr.copy()
    out[:, :, 1] = out[:, :, 1] * (1 - op * 0.16) + cyan[:, :, 1] * op * 0.16
    out[:, :, 2] = out[:, :, 2] * (1 - op * 0.24) + cyan[:, :, 2] * op * 0.24
    out[:, :, 0] = out[:, :, 0] * (1 - op * 0.18) + orange[:, :, 0] * op * 0.18
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def plate_wobble(img, strength=0.35):
    """Introduce subtle row-wise print wobble so crisp generated lines stop feeling digital."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    rng = np.random.default_rng()
    coarse = rng.normal(0, max(0.15, strength), max(4, h // 48))
    offsets = np.interp(np.arange(h), np.linspace(0, h - 1, len(coarse)), coarse)
    out = np.empty_like(arr)
    for y in range(h):
        out[y] = np.roll(arr[y], int(round(offsets[y])), axis=0)
    return Image.fromarray(out)


def print_scuffs(img, intensity=0.25):
    """Add sparse scuffs, scratches, and imperfect ink pickup."""
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    scuff = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng()
    for _ in range(int(24 * intensity) + 3):
        y = int(rng.integers(0, h))
        x0 = int(rng.integers(0, max(1, w - 1)))
        length = int(rng.integers(max(8, w // 24), max(12, w // 5)))
        thickness = int(rng.integers(1, 3))
        x1 = min(w, x0 + length)
        scuff[max(0, y - thickness):min(h, y + thickness + 1), x0:x1] = float(rng.uniform(0.25, 0.8))
    scuff_img = Image.fromarray((scuff * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.6))
    mask = (np.array(scuff_img).astype(np.float32) / 255.0)[:, :, None]
    paper = np.array(Image.new("RGB", img.size, "#F7EDE3")).astype(np.float32)
    out = arr * (1 - mask * intensity * 0.35) + paper * (mask * intensity * 0.35)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def process(input_path, output_path, intensity=0.5, mode="nous"):
    print(f"Processing: {input_path} (mode={mode}, intensity={intensity})")
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    print(f"  Input: {w}×{h}")
    s = intensity

    steps = [
        ("Warm grade", lambda: warm_grade(img, 0.12 * s)),
        ("CRT scanlines", lambda: crt_scanlines(img, 0.04 * s, 2)),
        ("Film grain", lambda: film_grain(img, 0.035 * s)),
        ("Bayer dither", lambda: bayer_dither(img, 0.2 * s, 12)),
        ("Vignette", lambda: vignette(img, 0.3 * s)),
        ("Chromatic aberr.", lambda: chromatic_aberration(img, 0.8 * s)),
    ]

    if mode in ("risograph", "nous", "imprint"):
        steps.extend([
            ("Screen print", lambda: screen_print_texture(img, 0.25 * s)),
            ("Paper texture", lambda: paper_texture(img, 0.12 * s)),
            ("Ink bleed", lambda: ink_bleed(img, 0.15 * s)),
        ])

    if mode == "imprint":
        steps.extend([
            ("Palette compress", lambda: palette_compress(img, 0.55 * s)),
            ("Xerox threshold", lambda: xerox_threshold(img, 0.35 * s)),
            ("Registration", lambda: registration_offset(img, 1.4 * s, 0.45 * s)),
            ("Plate wobble", lambda: plate_wobble(img, 0.7 * s)),
            ("Print scuffs", lambda: print_scuffs(img, 0.35 * s)),
        ])

    for i, (name, fn) in enumerate(steps):
        print(f"  [{i + 1}/{len(steps)}] {name}...")
        img = fn()

    img.save(output_path, "PNG")
    sz = os.path.getsize(output_path)
    print(f"  Saved: {output_path} ({sz // 1024}KB)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--intensity", "-i", type=float, default=0.5)
    parser.add_argument("--mode", "-m", choices=["standard", "risograph", "nous", "imprint"], default="nous")
    args = parser.parse_args()
    out = args.output or args.input.replace(".png", "-processed.png")
    process(args.input, out, args.intensity, args.mode)
    print("Done.")
