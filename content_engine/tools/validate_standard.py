#!/usr/bin/env python3
"""Validate the brand imagery standard end-to-end, reference-anchored.

For each priority lane: take one of Sahil's reference images as the style anchor,
call nano-banana-pro/edit (img2img, preserve-then-add prompt = DNA + lane skeleton
+ real topic), then apply the deterministic post-process grain pass. Assemble a
self-contained HTML gallery: anchor -> raw generation -> finished.

Proves: reference-anchoring locks the look + text is crisp + post-process makes it
on-brand, first-time. Uses Sahil's FAL credit directly (records to ledger, no gate).

Usage (env must carry FAL_KEY):
    cd content_engine && PYTHONPATH=. ../.venv/bin/python tools/validate_standard.py
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
OUT = Path(__file__).resolve().parent.parent / "output" / "validate"
OUT.mkdir(parents=True, exist_ok=True)
FAL_KEY = os.getenv("FAL_KEY", "")
EDIT_ID = "fal-ai/nano-banana-pro/edit"
COST = 0.12

# DNA preserved on every call (the constant brand identity).
DNA = ("KEEP the reference image's visual DNA exactly: deep near-black background, "
       "electric blue #3847FF with violet and magenta neon accents, heavy analog "
       "texture (film grain, halftone dots, CRT scanlines, ink bleed), distressed "
       "condensed uppercase display type, high-contrast dramatic chiaroscuro "
       "lighting, restrained composition with deliberate negative space. "
       "Do NOT use any human or anime character or mascot. No clean corporate/SaaS "
       "look. No clutter. All text must be crisp, correctly spelled, no gibberish.")

# lane -> (anchor index, aspect, change-instruction)
LANES = {
    "A_hero": (2, "1:1",
        "CHANGE the content to: a single atmospheric SYMBOLIC hero scene (no "
        "character) — a glowing secured vault at the heart of a vast data core, "
        "cinematic depth, volumetric light. Minimal text, only the title: "
        "\"SURVIVES COMPACTION\"."),
    "B_ladder": (24, "1:1",
        "CHANGE the content to: a vertical 5-rung LEVEL ladder infographic, "
        "numbered LEVEL 1 to LEVEL 5 bottom to top, each rung one short label. "
        "Title: \"5 LEVELS OF AGENT AUTONOMY\". Rung labels: \"Scripted\", "
        "\"AI-assisted\", \"Supervised\", \"Autonomous + gate\", \"Self-improving\"."),
    "C_skill": (106, "1:1",
        "CHANGE the content to: a dense 4-panel technical skill sheet with quadrants "
        "labelled PROBLEM, FIX, FLOW, ROBUST. Title: \"RESCUE HANDLE\". Short crisp "
        "labels: Problem=\"Tool result too large\", Fix=\"Store as SHA blob\", "
        "Flow=\"Hand model a 1.7k handle\", Robust=\"99x smaller context\"."),
    "D_poster": (84, "1:1",
        "CHANGE the content to: a bold retro halftone announcement poster. Huge "
        "distressed headline: \"NATIVE WINDOWS\" over \"OUT OF BETA\". One-line "
        "subhead: \"Hermes runs natively now\"."),
    "H_data": (8, "1:1",
        "CHANGE the content to: a clean two-bar chart on a dark background, a tall "
        "bar vs a tiny bar, title: \"CONTEXT TOKENS\". Axis labels: \"Before 180k\" "
        "and \"After 1.7k\". Minimal, one accent colour."),
}


def _files():
    return sorted([f for f in os.listdir(REFS)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _fal_upload(path: Path) -> str:
    """Upload to FAL storage; return public file_url. Falls back to data URI."""
    try:
        init = requests.post(
            "https://rest.alpha.fal.ai/storage/upload/initiate",
            headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
            json={"content_type": "image/jpeg", "file_name": f"{uuid.uuid4().hex}.jpg"},
            timeout=30,
        )
        if init.status_code == 200:
            d = init.json()
            put = requests.put(d["upload_url"], data=path.read_bytes(),
                               headers={"Content-Type": "image/jpeg"}, timeout=60)
            if put.status_code in (200, 201):
                return d["file_url"]
        print(f"  [upload] initiate {init.status_code}; using data URI")
    except Exception as exc:  # noqa: BLE001
        print(f"  [upload] {exc}; using data URI")
    return _data_uri(path)


def _generate(anchor_url: str, prompt: str, aspect: str) -> bytes | None:
    payload = {"prompt": prompt, "image_urls": [anchor_url],
               "aspect_ratio": aspect, "num_images": 1, "resolution": "1K"}
    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    # sync first, queue fallback
    try:
        r = requests.post(f"https://fal.run/{EDIT_ID}", json=payload, headers=headers, timeout=180)
        if r.status_code == 200:
            imgs = r.json().get("images", [])
            if imgs:
                return requests.get(imgs[0]["url"], timeout=90).content
            print(f"  no images: {list(r.json())[:6]}")
        else:
            print(f"  sync {r.status_code}: {r.text[:160]}")
            qr = requests.post(f"https://queue.fal.run/{EDIT_ID}", json=payload,
                               headers=headers, timeout=30)
            rid = qr.json().get("request_id") if qr.status_code == 200 else None
            if rid:
                for _ in range(60):
                    time.sleep(3)
                    pr = requests.get(f"https://queue.fal.run/{EDIT_ID}/requests/{rid}",
                                      headers=headers, timeout=15)
                    if pr.status_code == 200 and pr.json().get("status") == "COMPLETED":
                        imgs = pr.json().get("images", [])
                        if imgs:
                            return requests.get(imgs[0]["url"], timeout=90).content
                        break
    except Exception as exc:  # noqa: BLE001
        print(f"  gen error: {exc}")
    return None


def _finish(raw: Path, dst: Path) -> Path:
    """Deterministic cyber-classical analog post-process (the mandatory pass)."""
    img = Image.open(raw).convert("RGB")
    img = pp._contrast(img, 1.12)
    img = pp._halftone(img, 0.18)
    img = pp._scanlines(img, 0.22)
    img = pp._film_grain(img, 0.55)
    img = pp._vignette(img, 0.35)
    img.save(dst, quality=92)
    return dst


def run() -> None:
    if not FAL_KEY:
        print("FAL_KEY not set"); return
    files = _files()
    cells, total = [], 0.0
    for lane, (idx, aspect, change) in LANES.items():
        anchor = REFS / files[idx]
        print(f"[validate] {lane}: anchor [{idx}] {anchor.name}")
        url = _fal_upload(anchor)
        prompt = f"Use the reference image as the STYLE anchor. {DNA}\n\n{change}"
        t0 = time.time()
        data = _generate(url, prompt, aspect)
        dt = time.time() - t0
        if not data:
            print(f"  FAILED ({dt:.0f}s)")
            cells.append((lane, idx, anchor, None, None, "failed"))
            continue
        raw = OUT / f"{lane}_raw.png"; raw.write_bytes(data)
        fin = _finish(raw, OUT / f"{lane}_final.png")
        budget.record(COST, label=f"validate:{lane}")
        total += COST
        cells.append((lane, idx, anchor, raw, fin, f"{dt:.0f}s"))
        print(f"  OK ({dt:.0f}s) -> {fin.name}")

    def b64(p):
        return base64.b64encode(Path(p).read_bytes()).decode() if p and Path(p).exists() else ""
    def cell_img(p, mime="png"):
        s = b64(p)
        return f'<img src="data:image/{mime};base64,{s}"/>' if s else '<div class=miss>(none)</div>'

    html = ["""<!doctype html><meta charset=utf-8><title>Standard validation</title>
<style>body{background:#0c0c0c;color:#eee;font:14px system-ui;margin:24px}
h1{font-weight:600}.row{display:flex;gap:14px;margin:18px 0;align-items:flex-start}
.row img,.miss{width:300px;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000}
.miss{display:flex;align-items:center;justify-content:center;color:#a55}
.lab{width:120px;color:#9cf;font-weight:600}.cap{font-size:12px;color:#888;text-align:center}</style>
<h1>Imagery standard — reference-anchored validation</h1>
<p style=color:#888>Each row: your anchor &rarr; nano-banana-pro/edit (DNA + lane + topic) &rarr; post-processed final.</p>
<div class=row><div class=lab></div><div class=cap style=width:300px>ANCHOR (your ref)</div>
<div class=cap style=width:300px>RAW GENERATION</div><div class=cap style=width:300px>FINISHED (post-processed)</div></div>"""]
    for lane, idx, anchor, raw, fin, note in cells:
        mime = "png" if str(anchor).lower().endswith("png") else "jpeg"
        html.append(
            f"<div class=row><div class=lab>{lane}<br><span style=color:#666>[{idx}] {note}</span></div>"
            f"{cell_img(anchor, mime)}{cell_img(raw)}{cell_img(fin)}</div>")
    html.append(f"<p style=color:#888>Spend: £{total:.2f} ({len(cells)} lanes)</p>")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    print("\n" + "=" * 56)
    print(f"Gallery: {OUT/'index.html'}")
    print(f"Spend:   £{total:.2f}")


if __name__ == "__main__":
    run()
