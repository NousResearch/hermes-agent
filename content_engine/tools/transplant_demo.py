#!/usr/bin/env python3
"""Dual-anchor transplant demo — prove repeatable C-level quality.

Technique: feed nano-banana-pro/edit TWO anchors —
  1. a baoyu infographic LAYOUT exemplar (dense, designed structure)
  2. one of Sahil's dark cyber refs (our brand DNA: palette/texture/mood)
— then swap in a real topic + exact labels, styled to our DNA, heavier craft
post-process. This transplants a great design instead of inventing a sparse one
(the failure mode of the first validation).

Usage (env must carry FAL_KEY):
    cd content_engine && PYTHONPATH=. ../.venv/bin/python tools/transplant_demo.py
"""
from __future__ import annotations

import base64
import os
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import budget
import postprocess as pp
from PIL import Image

REFS = Path.home() / "content-references" / "Referencecontent"
LAYOUTS = Path.home() / "content-references" / "baoyu-screenshots" / "infographic-layouts"
OUT = Path(__file__).resolve().parent.parent / "output" / "transplant"
OUT.mkdir(parents=True, exist_ok=True)
FAL_KEY = os.getenv("FAL_KEY", "")
EDIT_ID = "fal-ai/nano-banana-pro/edit"
COST = 0.12

# Sahil DNA/style anchor — a dark, neon, grainy ref (index 24 = Dark Factory).
DNA_REF = REFS / "WhatsApp Image 2026-06-10 at 01.59.57 (5).jpeg"

DNA = ("Render in our brand DNA (take palette/texture/mood from the SECOND "
       "reference): deep near-black background, electric blue #3847FF with violet "
       "and magenta neon accents, heavy analog texture (film grain, halftone, CRT "
       "scanlines, ink bleed), distressed condensed uppercase display type, "
       "high-contrast dramatic lighting. No human/anime character. No clutter. "
       "All text crisp, correctly spelled, no gibberish.")

# (id, layout-file, topic title, labels-instruction)
JOBS = [
    ("redo_ladder", "pyramid.webp",
     "5 LEVELS OF AGENT AUTONOMY",
     "Five stacked tiers bottom-to-top, each a labelled band with a one-line "
     "descriptor: L1 Scripted, L2 AI-assisted, L3 Supervised, L4 Autonomous+gate, "
     "L5 Self-improving."),
    ("redo_versus", "scale-balance.webp",
     "SEEDREAM vs NANO-BANANA",
     "A balance/scale weighing two sides. LEFT 'Seedream45': cheap, flat, gibberish "
     "text, £0.03. RIGHT 'Nano-Banana Pro': premium, crisp text, on-brand, £0.12. "
     "Right side clearly wins."),
    ("new_iceberg", "iceberg.webp",
     "WHAT YOU SEE vs WHAT THE AGENT DOES",
     "Iceberg: above water 'One daily post'. Below water the hidden mass: "
     "'Signal scan', 'Dedup', 'Draft', 'Quality gates', 'Illustrate', 'Publish'."),
    ("new_bridge", "bridge.webp",
     "FROM CONTEXT OVERFLOW TO RESCUE HANDLE",
     "Left cliff 'Tool result 180k tokens'. Bridge across labelled steps: 'Store as "
     "SHA blob', 'Hand model a 1.7k handle', 'Pass by reference'. Right cliff "
     "'99x smaller context'."),
]


def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else ("image/webp" if path.suffix.lower()==".webp" else "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _upload(path: Path) -> str:
    try:
        init = requests.post(
            "https://rest.alpha.fal.ai/storage/upload/initiate",
            headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
            json={"content_type": "image/jpeg", "file_name": f"{uuid.uuid4().hex}.jpg"},
            timeout=30)
        if init.status_code == 200:
            d = init.json()
            if requests.put(d["upload_url"], data=path.read_bytes(),
                            headers={"Content-Type": "image/jpeg"}, timeout=60).status_code in (200, 201):
                return d["file_url"]
    except Exception as exc:  # noqa: BLE001
        print(f"  [upload] {exc}; data URI")
    return _data_uri(path)


def _generate(urls: list[str], prompt: str) -> bytes | None:
    payload = {"prompt": prompt, "image_urls": urls,
               "aspect_ratio": "4:5", "num_images": 1, "resolution": "2K"}
    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(f"https://fal.run/{EDIT_ID}", json=payload, headers=headers, timeout=220)
        if r.status_code == 200:
            imgs = r.json().get("images", [])
            if imgs:
                return requests.get(imgs[0]["url"], timeout=90).content
            print(f"  no images: {list(r.json())[:6]}")
        else:
            print(f"  {r.status_code}: {r.text[:160]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  gen error: {exc}")
    return None


def _finish(raw: Path, dst: Path) -> Path:
    img = Image.open(raw).convert("RGB")
    img = pp._contrast(img, 1.15)
    img = pp._halftone(img, 0.28)
    img = pp._scanlines(img, 0.30)
    img = pp._film_grain(img, 0.8)
    img = pp._vignette(img, 0.45)
    img.save(dst, quality=92)
    return dst


def run() -> None:
    if not FAL_KEY:
        print("FAL_KEY not set"); return
    dna_url = _upload(DNA_REF)
    cells, total = [], 0.0
    for jid, layout, title, labels in JOBS:
        lpath = LAYOUTS / layout
        print(f"[transplant] {jid}: layout={layout}")
        layout_url = _upload(lpath)
        prompt = (f"Create an infographic using the FIRST reference image purely as the "
                  f"LAYOUT and information-density template (same structure, same designed "
                  f"density). Title: \"{title}\". Content: {labels}\n\n{DNA}")
        t0 = time.time()
        data = _generate([layout_url, dna_url], prompt)
        dt = time.time() - t0
        if not data:
            print(f"  FAILED ({dt:.0f}s)"); cells.append((jid, layout, lpath, None, None, "failed")); continue
        raw = OUT / f"{jid}_raw.png"; raw.write_bytes(data)
        fin = _finish(raw, OUT / f"{jid}_final.png")
        budget.record(COST, label=f"transplant:{jid}"); total += COST
        cells.append((jid, layout, lpath, raw, fin, f"{dt:.0f}s"))
        print(f"  OK ({dt:.0f}s)")

    def b64(p): return base64.b64encode(Path(p).read_bytes()).decode() if p and Path(p).exists() else ""
    def img(p, mime="png"):
        s = b64(p); return f'<img src="data:image/{mime};base64,{s}"/>' if s else '<div class=miss>(none)</div>'
    html = ["""<!doctype html><meta charset=utf-8><title>Transplant demo</title>
<style>body{background:#0c0c0c;color:#eee;font:14px system-ui;margin:24px}
.row{display:flex;gap:14px;margin:18px 0;align-items:flex-start}
.row img,.miss{height:340px;border-radius:8px;background:#000}
.miss{width:280px;display:flex;align-items:center;justify-content:center;color:#a55}
.lab{width:130px;color:#9cf;font-weight:600}.cap{font-size:12px;color:#888}</style>
<h1>Dual-anchor transplant — baoyu layout + your DNA ref &rarr; finished</h1>
<p class=cap>Each row: baoyu LAYOUT anchor &rarr; raw generation &rarr; finished (heavier craft pass).</p>"""]
    for jid, layout, lpath, raw, fin, note in cells:
        html.append(f"<div class=row><div class=lab>{jid}<br><span style=color:#666>{layout} · {note}</span></div>"
                    f"{img(lpath,'webp')}{img(raw)}{img(fin)}</div>")
    html.append(f"<p class=cap>Spend: £{total:.2f}</p>")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    print("\n" + "=" * 56); print(f"Gallery: {OUT/'index.html'}"); print(f"Spend: £{total:.2f}")


if __name__ == "__main__":
    run()
