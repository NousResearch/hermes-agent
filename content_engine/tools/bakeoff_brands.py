#!/usr/bin/env python3
"""Per-brand bake-off: prove nano-banana-pro delivers each brand's DISTINCT look
first-time. No standardisation — five different genres, each on-brand.

Decides the API-vs-manual question: if every brand lands first generation, the
£10 API path is viable and the manual ChatGPT/Google treadmill is unnecessary.

Each cell = one hero, nano-banana-pro, exact prompt, no chain fallback, no OCR
regen. Spend is budget-gated and recorded. Output: a single self-contained
HTML gallery (base64) at output/bakeoff_brands/index.html.

Usage (env must carry FAL_KEY):
    cd content_engine && PYTHONPATH=. ../.venv/bin/python tools/bakeoff_brands.py
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import budget
import fal_client
from fal_client import MODEL_COST_GBP

MODEL = "nano_banana"
OUT = Path(__file__).resolve().parent.parent / "output" / "bakeoff_brands"
OUT.mkdir(parents=True, exist_ok=True)

# Each brand: a genuinely distinct genre + its canonical palette. Prompts are
# art-directed for premium, first-time quality with crisp (not gibberish) text.
BRANDS = [
    ("sahil_twitter", "Sahil / X — dark cyberpunk dev HUD", """
A rich, cinematic, highly detailed cyberpunk digital illustration, concept-art quality.
Scene: a glowing secured context-memory vault at the centre of a vast dark data core,
volumetric neon lighting, atmospheric haze, reflective wet surfaces, deep perspective.
Palette: near-black background (#0A0A0A), electric cyan and magenta neon, a single
crimson accent. Premium, moody, high production value.
Overlay: translucent glassy sci-fi HUD panels with crisp, perfectly legible labels:
header "MEMLOCK PROTOCOL", and rows "Lock context", "Unlock intelligence",
"Re-assert instruction", "Pin once". Text sharp and integrated like a game interface.
Square aspect. No gibberish text, no watermark.
"""),
    ("sahil_linkedin", "Sahil / LinkedIn — restrained HBR editorial", """
A sophisticated, restrained editorial illustration in the style of a Harvard Business
Review feature. Clean, minimal, lots of calm negative space, elegant geometric
abstraction representing an agentic AI system orchestrating many specialised workers.
Palette: warm off-white / light grey paper, charcoal linework, ONE deep indigo accent.
No neon, no cyberpunk. Premium, intellectual, calm, conceptual.
Subtle integrated headline text, perfectly legible: "Agentic systems in practice".
Square aspect. No gibberish text, no watermark, no clutter.
"""),
    ("plenishd", "Plenishd — warm photographic kitchen", """
A warm, appetising, photographic-quality scene of a cosy modern UK kitchen at golden
hour, soft natural window light, shallow depth of field, fresh groceries and a wooden
worktop, a phone resting against a bowl showing a tidy pantry list.
Palette: warm dark wood tones and a deep warm charcoal background (#2C2A28), with a
glowing Plenishd-yellow (#FBBF24) accent on the phone screen. Inviting, premium,
hand-crafted feel.
Tasteful integrated headline, perfectly legible: "Snap. Say. Stock it."
Square aspect. No gibberish text, no watermark.
"""),
    ("coachos", "CoachOS — B&W documentary football", """
A black-and-white documentary photograph of grassroots football on a misty Saturday
morning: a coach with a clipboard beside a muddy pitch, young players warming up in the
background, atmospheric breath in cold air, fine film grain, editorial composition.
Near-monochrome, restrained, high contrast, with exactly ONE selective green accent
(the coach's bib / a cone) in #22C55E. Premium editorial sports documentary feel.
Tasteful integrated headline, perfectly legible: "One session. Twelve players."
Square aspect. No gibberish text, no watermark.
"""),
    ("matchdaymaestro", "MatchdayMaestro — retro halftone poster", """
A bold retro halftone matchday poster, screen-print style, dynamic and energetic.
Subject: a stylised footballer mid-strike with dramatic motion, World Cup 2026 energy,
ben-day dots, gritty print texture, strong diagonal composition.
Palette: off-black background, burnt orange and crimson (#E11D48) duotone with a punchy
yellow (#FBBF24) highlight. Loud, gamified, poster-grade.
Bold integrated headline, perfectly legible: "WORLD CUP 2026".
Square aspect. No gibberish text, no watermark.
"""),
]


def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _img(p: Path | None) -> str:
    if p and p.exists():
        return f'<img src="data:image/png;base64,{_b64(p)}" />'
    return '<div class="miss">(failed)</div>'


def run() -> None:
    cells = []
    total = 0.0
    cost = MODEL_COST_GBP.get(MODEL, 0.12)
    for brand, label, prompt in BRANDS:
        if not budget.can_spend(cost):
            print(f"[bakeoff] budget cap; skip {brand}")
            cells.append((brand, label, None, "budget cap"))
            continue
        print(f"[bakeoff] {brand} ...", flush=True)
        t0 = time.time()
        try:
            path = fal_client.generate_image(
                prompt.strip(), model=MODEL, aspect="square",
                output_dir=str(OUT), filename=f"{brand}.png",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {exc}")
            path = None
        dt = time.time() - t0
        if path and Path(path).exists():
            budget.record(cost, label=f"bakeoff_brand:{brand}")
            total += cost
            cells.append((brand, label, Path(path), f"{dt:.0f}s"))
            print(f"  OK {path} ({dt:.0f}s)")
        else:
            cells.append((brand, label, None, "no image"))
            print(f"  FAILED ({dt:.0f}s)")

    html = ["""<!doctype html><meta charset=utf-8>
<title>Per-brand bake-off — nano-banana-pro</title>
<style>
 body{background:#111;color:#eee;font:14px/1.5 system-ui,sans-serif;margin:24px}
 h1{font-weight:600}
 .grid{display:flex;flex-wrap:wrap;gap:20px}
 .cell{width:360px;background:#1b1b1b;border:1px solid #333;border-radius:10px;padding:12px}
 .cell img,.miss{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#000}
 .miss{display:flex;align-items:center;justify-content:center;color:#a55}
 .lab{font-weight:600;margin:8px 0 2px} .meta{color:#888;font-size:12px}
</style>
<h1>Per-brand bake-off — nano-banana-pro, one hero each, first generation</h1>
<p class=meta>No standardisation: each brand its own genre. The question this
answers: does the API deliver each brand's look first-time?</p>
<div class=grid>"""]
    for brand, label, path, note in cells:
        html.append(f"<div class=cell>{_img(path)}<div class=lab>{label}</div>"
                    f"<div class=meta>{brand} · {MODEL} · £{cost:.3f} · {note}</div></div>")
    html.append("</div>")
    html.append(f"<p class=meta>Total spend: <b>£{total:.2f}</b></p>")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Gallery: {OUT / 'index.html'}")
    print(f"Total:   £{total:.2f}")


if __name__ == "__main__":
    run()
