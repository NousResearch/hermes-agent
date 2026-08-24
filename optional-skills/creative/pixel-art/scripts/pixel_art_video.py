"""Pixel art video — overlay procedural animations onto a source image.

Takes any image (typically pre-processed with pixel_art()) and overlays
animated pixel effects (stars, rain, fireflies, etc.), then encodes to MP4
(and optionally GIF) via ffmpeg.

Scene animations ported from pixel-art-studio (MIT) — see ATTRIBUTION.md.
The generative/Pollinations code is intentionally dropped — Hermes uses
`image_generate` + `pixel_art()` for base frames instead.

Usage (import):
    from pixel_art_video import pixel_art_video
    pixel_art_video("frame.png", "out.mp4", scene="night", duration=6)

Usage (CLI):
    python pixel_art_video.py frame.png out.mp4 --scene night --duration 6 --gif
"""

import math
import os
import random
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw


# ── Pixel drawing helpers ──────────────────────────────────────────────

def _px(draw, x, y, color, size=2):
    x, y = int(x), int(y)
    W, H = draw.im.size
    if 0 <= x < W and 0 <= y < H:
        draw.rectangle([x, y, x + size - 1, y + size - 1], fill=color)


def _pixel_cross(draw, x, y, color, arm=2):
    x, y = int(x), int(y)
    for i in range(-arm, arm + 1):
        _px(draw, x + i, y, color, 1)
        _px(draw, x, y + i, color, 1)


# ── Animation init/draw pairs ──────────────────────────────────────────

def init_stars(rng, W, H):
    return [(rng.randint(0, W), rng.randint(0, H // 2)) for _ in range(15)]

def draw_stars(draw, stars, t, W, H):
    for i, (sx, sy) in enumerate(stars):
        if math.sin(t * 2.0 + i * 0.7) > 0.65:
            _pixel_cross(draw, sx, sy, (255, 255, 220), arm=2)


def init_fireflies(rng, W, H):
    return [{"x": rng.randint(20, W - 20), "y": rng.randint(H // 4, H - 20),
             "phase": rng.uniform(0, 6.28), "speed": rng.uniform(0.3, 0.8)}
            for _ in range(10)]

def draw_fireflies(draw, ff, t, W, H):
    for f in ff:
        if math.sin(t * 1.5 + f["phase"]) < 0.15:
            continue
        _px(draw,
            f["x"] + math.sin(t * f["speed"] + f["phase"]) * 3,
            f["y"] + math.cos(t * f["speed"] * 0.7) * 2,
            (200, 255, 100), 2)


def init_leaves(rng, W, H):
    return [{"x": rng.randint(0, W), "y": rng.randint(-H, 0),
             "speed": rng.uniform(0.5, 1.5), "wobble": rng.uniform(0.02, 0.05),
             "phase": rng.uniform(0, 6.28),
             "color": rng.choice([(180, 120, 50), (160, 100, 40), (200, 140, 60)])}
            for _ in range(12)]

def draw_leaves(draw, leaves, t, W, H):
    for leaf in leaves:
        _px(draw,
            leaf["x"] + math.sin(t * leaf["wobble"] + leaf["phase"]) * 15,
            (leaf["y"] + t * leaf["speed"] * 20) % (H + 40) - 20,
            leaf["color"], 2)


def init_dust_motes(rng, W, H):
    return [{"x": rng.randint(30, W - 30), "y": rng.randint(30, H - 30),
             "phase": rng.uniform(0, 6.28), "speed": rng.uniform(0.2, 0.5),
             "amp": rng.uniform(2, 6)} for _ in range(20)]

def draw_dust_motes(draw, motes, t, W, H):
    for m in motes:
        if math.sin(t * 2.0 + m["phase"]) > 0.3:
            _px(draw,
                m["x"] + math.sin(t * 0.3 + m["phase"]) * m["amp"],
                m["y"] - (m["speed"] * t * 15) % H,
                (255, 210, 100), 1)


def init_sparkles(rng, W, H):
    return [(rng.randint(W // 4, 3 * W // 4), rng.randint(H // 4, 3 * H // 4),
             rng.uniform(0, 6.28),
             rng.choice([(180, 200, 255), (255, 220, 150), (200, 180, 255)]))
            for _ in range(10)]

def draw_sparkles(draw, sparkles, t, W, H):
    for sx, sy, phase, color in sparkles:
        if math.sin(t * 1.8 + phase) > 0.6:
            _pixel_cross(draw, sx, sy, color, arm=2)


def init_rain(rng, W, H):
    return [{"x": rng.randint(0, W), "y": rng.randint(0, H),
             "speed": rng.uniform(4, 8)} for _ in range(30)]

def draw_rain(draw, rain, t, W, H):
    for r in rain:
        y = (r["y"] + t * r["speed"] * 20) % H
        _px(draw, r["x"], y, (120, 150, 200), 1)
        _px(draw, r["x"], y + 4, (100, 130, 180), 1)


def init_lightning(rng, W, H):
    return {"timer": 0, "flash": False, "rng": rng}

def draw_lightning(draw, state, t, W, H):
    state["timer"] += 1
    if state["timer"] > 45 and state["rng"].random() < 0.04:
        state["flash"] = True
        state["timer"] = 0
    if state["flash"]:
        for x in range(0, W, 4):
            for y in range(0, H // 3, 3):
                if state["rng"].random() < 0.12:
                    _px(draw, x, y, (255, 255, 240), 2)
        state["flash"] = False


def init_bubbles(rng, W, H):
    return [{"x": rng.randint(20, W - 20), "y": rng.randint(H, H * 2),
             "speed": rng.uniform(0.3, 0.8), "size": rng.choice([1, 2, 2])}
            for _ in range(15)]

def draw_bubbles(draw, bubbles, t, W, H):
    for b in bubbles:
        x = b["x"] + math.sin(t * 0.5 + b["x"]) * 3
        y = b["y"] - (t * b["speed"] * 20) % (H + 40)
        if 0 < y < H:
            _px(draw, x, y, (150, 200, 255), b["size"])


def init_embers(rng, W, H):
    return [{"x": rng.randint(0, W), "y": rng.randint(0, H),
             "speed": rng.uniform(0.3, 0.9), "phase": rng.uniform(0, 6.28),
             "color": rng.choice([(255, 150, 30), (255, 100, 20), (255, 200, 50)])}
            for _ in range(18)]

def draw_embers(draw, embers, t, W, H):
    for e in embers:
        x = e["x"] + math.sin(t * 0.4 + e["phase"]) * 5
        y = e["y"] - (t * e["speed"] * 15) % H
        if math.sin(t * 2.5 + e["phase"]) > 0.2:
            _px(draw, x, y, e["color"], 2)


def init_snowflakes(rng, W, H):
    return [{"x": rng.randint(0, W), "y": rng.randint(-H, 0),
             "speed": rng.uniform(0.3, 0.6), "wobble": rng.uniform(0.04, 0.09),
             "size": rng.choice([2, 2, 3])}
            for _ in range(40)]

def draw_snowflakes(draw, flakes, t, W, H):
    for f in flakes:
        x = f["x"] + math.sin(t * f["wobble"] + f["x"]) * 20
        y = (f["y"] + t * f["speed"] * 8) % (H + 20) - 10
        if f["size"] >= 3:
            _pixel_cross(draw, x, y, (230, 235, 255), arm=1)
        else:
            _px(draw, x, y, (230, 235, 255), 2)


def init_neon_pulse(rng, W, H):
    return [(rng.randint(0, W), rng.randint(0, H), rng.uniform(0, 6.28),
             rng.choice([(255, 0, 200), (0, 255, 255), (255, 50, 150)]))
            for _ in range(8)]

def draw_neon_pulse(draw, points, t, W, H):
    for x, y, phase, color in points:
        if math.sin(t * 2.5 + phase) > 0.5:
            _pixel_cross(draw, x, y, color, arm=3)


def init_heat_shimmer(rng, W, H):
    return [{"x": rng.randint(0, W), "y": rng.randint(H // 2, H),
             "phase": rng.uniform(0, 6.28)} for _ in range(12)]

def draw_heat_shimmer(draw, points, t, W, H):
    for p in points:
        x = p["x"] + math.sin(t * 0.8 + p["phase"]) * 2
        y = p["y"] + math.sin(t * 1.2 + p["phase"]) * 1
        if abs(math.sin(t * 1.5 + p["phase"])) > 0.6:
            _px(draw, x, y, (255, 200, 100), 1)


# ── Scene → animation mapping ──────────────────────────────────────────

SCENES = {
    "night":      ["stars", "fireflies", "leaves"],
    "dusk":       ["fireflies", "sparkles"],
    "tavern":     ["dust_motes", "sparkles"],
    "indoor":     ["dust_motes"],
    "urban":      ["rain", "neon_pulse"],
    "nature":     ["leaves", "fireflies"],
    "magic":      ["sparkles", "fireflies"],
    "storm":      ["rain", "lightning"],
    "underwater": ["bubbles", "sparkles"],
    "fire":       ["embers", "sparkles"],
    "snow":       ["snowflakes", "sparkles"],
    "desert":     ["heat_shimmer", "dust_motes"],
}

# Map scene layer name to (init_fn, draw_fn).
_LAYERS = {
    "stars":        (init_stars, draw_stars),
    "fireflies":    (init_fireflies, draw_fireflies),
    "leaves":       (init_leaves, draw_leaves),
    "dust_motes":   (init_dust_motes, draw_dust_motes),
    "sparkles":     (init_sparkles, draw_sparkles),
    "rain":         (init_rain, draw_rain),
    "lightning":    (init_lightning, draw_lightning),
    "bubbles":      (init_bubbles, draw_bubbles),
    "embers":       (init_embers, draw_embers),
    "snowflakes":   (init_snowflakes, draw_snowflakes),
    "neon_pulse":   (init_neon_pulse, draw_neon_pulse),
    "heat_shimmer": (init_heat_shimmer, draw_heat_shimmer),
}


def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install via your package manager or "
            "download from https://ffmpeg.org/"
        )


def pixel_art_video(
    base_image,
    output_path,
    scene="night",
    duration=6,
    fps=15,
    seed=None,
    export_gif=False,
):
    """Overlay pixel animations onto a base image and encode to MP4.

    Args:
        base_image: path to source image (ideally already pixel-art styled)
        output_path: path to MP4 output (GIF sibling written if export_gif=True)
        scene: key from SCENES (night, urban, storm, snow, fire, ...)
        duration: seconds of animation
        fps: frames per second (default 15 for retro feel)
        seed: optional int for reproducible animation placement
        export_gif: also write a GIF alongside the MP4

    Returns:
        (mp4_path, gif_path_or_None)
    """
    if scene not in SCENES:
        raise ValueError(
            f"Unknown scene {scene!r}. Choose from: {sorted(SCENES)}"
        )
    _ensure_ffmpeg()

    base = Image.open(base_image).convert("RGB")
    W, H = base.size

    rng = random.Random(seed if seed is not None else 42)
    layers = []
    for name in SCENES[scene]:
        init_fn, draw_fn = _LAYERS[name]
        layers.append((draw_fn, init_fn(rng, W, H)))

    n_frames = fps * duration
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pixelart_frames_") as frames_dir:
        for frame_idx in range(n_frames):
            canvas = base.copy()
            draw = ImageDraw.Draw(canvas)
            t = frame_idx / fps
            for draw_fn, state in layers:
                draw_fn(draw, state, t, W, H)
            canvas.save(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"))

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-framerate", str(fps),
             "-i", os.path.join(frames_dir, "frame_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             output_path],
            check=True,
        )

        gif_path = None
        if export_gif:
            gif_path = output_path.rsplit(".", 1)[0] + ".gif"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(fps),
                 "-i", os.path.join(frames_dir, "frame_%04d.png"),
                 "-vf",
                 "scale=320:-1:flags=neighbor,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                 "-loop", "0",
                 gif_path],
                check=True,
            )

    return output_path, gif_path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Overlay pixel animations onto an image → MP4.")
    p.add_argument("base_image")
    p.add_argument("output")
    p.add_argument("--scene", default="night", choices=sorted(SCENES))
    p.add_argument("--duration", type=int, default=6)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gif", action="store_true")
    args = p.parse_args()
    mp4, gif = pixel_art_video(
        args.base_image, args.output,
        scene=args.scene, duration=args.duration,
        fps=args.fps, seed=args.seed, export_gif=args.gif,
    )
    print(f"Wrote {mp4}")
    if gif:
        print(f"Wrote {gif}")


if __name__ == "__main__":
    main()
