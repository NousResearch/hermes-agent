#!/usr/bin/env python3
"""Scene model bake-off — same scene transplant, cheaper + free models vs nano-pro.

Question: scenes are the one PAID path with no £0 fallback. Can a much cheaper
model (qwen-image-edit ~£0.024, flux-kontext £0.04, nano non-pro £0.039) or the
FREE pollinations (text-only, no anchor) hold acceptable quality vs nano-banana-pro
(£0.12)? 3 concepts × each model. Grid: rows=concept, cols=model.

Usage (env carries FAL_KEY): cd content_engine && PYTHONPATH=. ../.venv/bin/python tools/scene_model_bakeoff.py
"""
from __future__ import annotations
import base64, os, sys, time, uuid
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import budget, fal_client, postprocess as pp
import imagery_library as lib
import imagery_transplant as it
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "output" / "scene_models"
OUT.mkdir(parents=True, exist_ok=True)
FAL_KEY = os.getenv("FAL_KEY", "")

# model key -> (endpoint, £/img, image-field). "" endpoint = free pollinations.
MODELS = [
    ("nano_pro", "fal-ai/nano-banana-pro/edit", 0.12, "image_urls"),
    ("nano", "fal-ai/nano-banana/edit", 0.039, "image_urls"),
    ("kontext", "fal-ai/flux-pro/kontext", 0.04, "image_url"),
    ("qwen", "fal-ai/qwen-image-edit", 0.024, "image_url"),
    ("pollinations", "", 0.0, None),  # free, text-only (no anchor)
]
# (archetype, palette) for the 3 concepts (the strong ones; atmospheric dropped).
CONCEPTS = [
    ("abstract", "cyber_neon", "THE GHOST IN THE TRANSFORMER",
     "a transformer model's attention as constellations of light and data particles "
     "dissolving into a thinking void"),
    ("tarot", "synthwave", "THE LIBRARY OF BABEL",
     "a mystical tarot card: an infinite library of glowing books spiralling into the "
     "cosmos, a symbolic seeker at the centre"),
    ("mythic", "warm_editorial", "FIRST PRINCIPLES",
     "a marble classical statue of a thinker fused with fine circuitry, mythic gravitas"),
]


def _headers():
    return {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}


def _images_from(d):
    return d.get("images") or (d.get("output") or {}).get("images") or ([d["image"]] if "image" in d else [])


def _fal_edit(endpoint, payload):
    r = requests.post(f"https://fal.run/{endpoint}", json=payload, headers=_headers(), timeout=220)
    if r.status_code == 200:
        imgs = _images_from(r.json())
        if imgs:
            u = imgs[0]["url"] if isinstance(imgs[0], dict) else imgs[0]
            return requests.get(u, timeout=90).content
    else:
        print(f"    {r.status_code} {r.text[:120]}")
    return None


def run():
    if not FAL_KEY:
        print("no FAL_KEY"); return
    # upload anchors once per archetype
    grid = {}  # (concept_idx, model) -> path
    total = 0.0
    for ci, (arc, pal_name, title, concept) in enumerate(CONCEPTS):
        anchor = sorted(lib._scene_dir(arc).glob("*.[jpw]*"))[0]
        pal = lib.PALETTES[pal_name]; meta = lib.SCENE_ARCHETYPES[arc]
        prompt = it.build_scene_prompt(title, concept, meta["desc"], pal["hex"], meta["text"])
        anchor_url = fal_client.upload_file(anchor)
        for mkey, endpoint, cost, field in MODELS:
            print(f"[{arc}] {mkey}")
            t0 = time.time(); data = None
            try:
                if mkey == "pollinations":
                    from image_generator import _pollinations_generate
                    txt = (f"{meta['desc']}. Subject: {concept}. PALETTE: {pal['hex']}. "
                           f"Cinematic, analog grain, concept-art, no gibberish text.")
                    data = _pollinations_generate(txt)
                else:
                    payload = {"prompt": prompt, "aspect_ratio": "4:5", "num_images": 1}
                    payload[field] = [anchor_url] if field == "image_urls" else anchor_url
                    data = _fal_edit(endpoint, payload)
            except Exception as e:  # noqa: BLE001
                print(f"    ERROR {e}")
            dt = time.time() - t0
            if not data:
                print(f"    FAIL ({dt:.0f}s)"); continue
            raw = OUT / f"{arc}_{mkey}_raw.png"; raw.write_bytes(data)
            fin = OUT / f"{arc}_{mkey}.png"
            try:
                pp.finish_imagery(Image.open(raw), light=pal["light"]).save(fin, quality=90)
            except Exception:
                fin = raw
            if cost:
                budget.record(cost, label=f"scenebake:{arc}:{mkey}"); total += cost
            grid[(ci, mkey)] = fin
            print(f"    OK ({dt:.0f}s)")

    # gallery grid
    def b(p): return base64.b64encode(Path(p).read_bytes()).decode() if p and Path(p).exists() else ""
    def img(p): s = b(p); return f'<img src="data:image/png;base64,{s}"/>' if s else '<div class=miss>fail</div>'
    cols = "".join(f"<th>{m} <span style=color:#888>£{c:.3f}</span></th>" for m, _, c, _ in MODELS)
    rows = ""
    for ci, (arc, pal_name, title, _) in enumerate(CONCEPTS):
        cells = "".join(f"<td>{img(grid.get((ci, m)))}</td>" for m, _, _, _ in MODELS)
        rows += f"<tr><th>{arc}<br><span style=color:#888>{pal_name}</span></th>{cells}</tr>"
    html = f"""<!doctype html><meta charset=utf-8><title>Scene model bake-off</title>
<style>body{{background:#0c0c0c;color:#eee;font:13px system-ui;margin:20px}}
table{{border-collapse:collapse}}td,th{{padding:6px;vertical-align:top}}
img,.miss{{width:230px;border-radius:6px;background:#000}}.miss{{height:288px;display:flex;align-items:center;justify-content:center;color:#a55}}</style>
<h1>Scene model bake-off — quality vs cost</h1>
<p style=color:#888>Same anchor + scene prompt. nano-pro = quality ceiling (£0.12). Looking for "a lot cheaper" that still holds up. pollinations = free, no anchor.</p>
<table><tr><th></th>{cols}</tr>{rows}</table>
<p style=color:#888>Spend: £{total:.2f}</p>"""
    (OUT / "index.html").write_text(html)
    print("=" * 50); print(f"Gallery: {OUT/'index.html'}  £{total:.2f}")


if __name__ == "__main__":
    run()
