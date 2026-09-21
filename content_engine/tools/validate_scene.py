#!/usr/bin/env python3
"""Validate the non-infographic SCENE path — one piece per archetype, varied
palette, single-anchor transplant on Sahil's refs. 5 examples (paid, nano-banana).

Usage (env must carry FAL_KEY): from content_engine/
    PYTHONPATH=. ../.venv/bin/python tools/validate_scene.py
"""
from __future__ import annotations
import base64, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import budget, fal_client, postprocess as pp
import imagery_library as lib
import imagery_transplant as it
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "output" / "scene"
OUT.mkdir(parents=True, exist_ok=True)
FAL_KEY = os.getenv("FAL_KEY", "")

# (archetype, palette, title, concept) — one per archetype, palettes varied.
JOBS = [
    ("abstract", "cyber_neon", "THE GHOST IN THE TRANSFORMER",
     "a transformer model's attention rendered as constellations of light and data "
     "particles dissolving into a thinking void"),
    ("mythic", "warm_editorial", "FIRST PRINCIPLES",
     "a marble classical statue of a thinker fused with fine circuitry, mythic gravitas, "
     "shafts of light"),
    ("atmospheric", "blueprint_mono", "WEEK ONE ON THE VPS",
     "a dark server room at night, racks glowing, a lone workstation, cinematic haze and depth"),
    ("tarot", "synthwave", "THE LIBRARY OF BABEL",
     "a mystical tarot card: an infinite library of glowing books spiralling into the cosmos, "
     "a symbolic seeker at the centre"),
    ("creature", "acid_duotone", "INTRODUCING GITRADAR",
     "a bold iconic radar-creature mascot sweeping a galaxy of code repositories, high impact"),
]


def _finish(raw: Path, dst: Path, light: bool) -> Path:
    img = pp.finish_imagery(Image.open(raw), light=light)
    img.save(dst, quality=92)
    return dst


def run() -> None:
    if not FAL_KEY:
        print("FAL_KEY not set"); return
    cells, total = [], 0.0
    for arc, pal_name, title, concept in JOBS:
        pal = lib.PALETTES[pal_name]
        meta = lib.SCENE_ARCHETYPES[arc]
        anchor = sorted(lib._scene_dir(arc).glob("*.[jpw]*"))[0]
        prompt = it.build_scene_prompt(title, concept, meta["desc"], pal["hex"], meta["text"])
        print(f"[scene] {arc} / {pal_name}")
        url = fal_client.upload_file(anchor)
        t0 = time.time()
        raw = fal_client.generate_image_edit(
            prompt, [url], aspect="4:5", output_dir=str(OUT),
            filename=f"scene_{arc}_raw.png") if url else None
        dt = time.time() - t0
        if not raw or not os.path.exists(raw):
            print(f"  FAILED ({dt:.0f}s)"); cells.append((arc, pal_name, anchor, None, "fail")); continue
        fin = _finish(Path(raw), OUT / f"scene_{arc}.png", pal["light"])
        budget.record(0.12, label=f"scene:{arc}:{pal_name}"); total += 0.12
        cells.append((arc, pal_name, anchor, fin, f"{dt:.0f}s"))
        print(f"  OK ({dt:.0f}s)")

    def b(p): return base64.b64encode(Path(p).read_bytes()).decode() if p and Path(p).exists() else ""
    def img(p, m="png"): s = b(p); return f'<img src="data:image/{m};base64,{s}"/>' if s else '<div class=miss>x</div>'
    html = ["""<!doctype html><meta charset=utf-8><title>Scene validation</title>
<style>body{background:#0c0c0c;color:#eee;font:14px system-ui;margin:24px}
.row{display:flex;gap:14px;margin:16px 0;align-items:flex-start}
.row img,.miss{height:340px;border-radius:8px;background:#000}
.lab{width:150px;color:#9cf;font-weight:600}.cap{font-size:12px;color:#888}</style>
<h1>Non-infographic scene validation — archetype × palette</h1>
<p class=cap>Each row: your archetype ANCHOR → finished scene (single-anchor transplant + your subject).</p>"""]
    for arc, pal_name, anchor, fin, note in cells:
        am = "jpeg" if str(anchor).lower().endswith((".jpg", ".jpeg")) else ("webp" if str(anchor).lower().endswith(".webp") else "png")
        html.append(f"<div class=row><div class=lab>{arc}<br><span style=color:#666>{pal_name} · {note}</span></div>"
                    f"{img(anchor, am)}{img(fin)}</div>")
    html.append(f"<p class=cap>Spend: £{total:.2f}</p>")
    (OUT / "index.html").write_text("\n".join(html))
    print("=" * 50); print(f"Gallery: {OUT/'index.html'}  £{total:.2f}")


if __name__ == "__main__":
    run()
