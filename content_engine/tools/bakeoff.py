#!/usr/bin/env python3
"""Image bake-off: one concept, many (model x concept) cells, side-by-side HTML.

Purpose: stop blind full-pipeline runs. Generate ONE representative concept
(the sahil_twitter dark-HUD "MEMLOCK PROTOCOL" hero, which has a reference
target in ~/Reference Images/) across a small model x prompt matrix, then
build a single self-contained HTML page (base64-embedded) so the winning
direction can be picked once and standardised.

Each cell calls fal_client.generate_image directly (exact model, exact prompt,
NO OCR regen, NO silent chain fallback) so the comparison is clean. Spend is
budget-gated and recorded.

Usage:
    cd content_engine && PYTHONPATH=. ../.venv/bin/python tools/bakeoff.py
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

OUT = Path(__file__).resolve().parent.parent / "output" / "bakeoff"
OUT.mkdir(parents=True, exist_ok=True)
REF_DIR = Path.home() / "Reference Images"

# ── The concept under test (matches tools/preview_images.py memlock draft) ──
TITLE = "MEMLOCK PROTOCOL"
STEPS = ["Lock context", "Unlock intelligence", "Re-assert standing instruction",
         "Hermes plugin", "Pin once"]
SUBJECT = ("a developer's context-memory lock system that survives AI context "
           "compaction: a glowing secured vault/lock at the centre of a data core")

# ── Prompt concepts ────────────────────────────────────────────────────────
# A. Current pipeline output (flat dark-HUD infographic). Reproduced inline so
#    the bake-off is self-contained and does not depend on routing.
P_CURRENT = f"""TYPE: INFOGRAPHIC illustration.
Dark cyberpunk HUD dashboard. Near-black background, thin glowing panel borders,
neon cyan and magenta accents, monospace labels.
Title header: "{TITLE}".
Vertical stack of labelled status rows:
{chr(10).join(f'  - {s}' for s in STEPS)}
TEXT RULE: render only these short labels, no other text, no watermarks.
Clean composition, respect square aspect ratio."""

# B. Rich cinematic SCENE + HUD overlay, NO character. The target aesthetic
#    from the reference, adapted for build-in-public dev content.
P_SCENE_CINEMATIC = f"""A rich, cinematic, highly detailed digital illustration. NOT a flat dashboard,
NOT a simple checklist, NOT icons.
Scene: {SUBJECT}. Dramatic depth, volumetric neon lighting, atmospheric haze,
reflective surfaces, intricate detail, concept-art quality.
Palette: deep near-black background (#0A0A0A) with electric cyan, magenta and
lime neon accents. Premium, moody, high production value.
Overlay: translucent glassy HUD panels framing the scene, with crisp, perfectly
legible short labels reading exactly: "{TITLE}" as the header, and the steps
{', '.join(f'"{s}"' for s in STEPS)}.
The HUD text must be sharp and readable, integrated into the composition like a
sci-fi interface. Square aspect ratio."""

# C. Rich SCENE with a stylised character/mascot + HUD, like the reference set.
P_SCENE_CHARACTER = f"""A rich, cinematic, highly detailed digital illustration in a moody cyberpunk
style, concept-art quality. NOT flat, NOT a checklist.
A lone stylised "memory guardian" agent character standing in a dark, rain-slick
neon-lit data-centre alley, full body, dramatic rim lighting, intricate detail,
atmospheric depth. The character guards a glowing context-lock vault.
Palette: deep near-black background with electric cyan, magenta and lime neon
accents. Premium, atmospheric, high production value.
Overlay: translucent glassy HUD panels around the character with crisp, perfectly
legible labels reading exactly: "{TITLE}" as the header, and the steps
{', '.join(f'"{s}"' for s in STEPS)}.
HUD text sharp and readable, integrated like a game interface. Square aspect."""

# Matrix: (cell_id, label, model, prompt). ~£0.42 total.
MATRIX = [
    ("01_seedream_current",  "seedream45 + current infographic (BASELINE)", "seedream45",  P_CURRENT),
    ("02_nano_current",      "nano-banana-pro + current infographic",        "nano_banana", P_CURRENT),
    ("03_nano_scene",        "nano-banana-pro + cinematic scene + HUD",       "nano_banana", P_SCENE_CINEMATIC),
    ("04_nano_character",    "nano-banana-pro + character scene + HUD",       "nano_banana", P_SCENE_CHARACTER),
    ("05_recraft_scene",     "recraft (cheap) + cinematic scene + HUD",       "recraft",     P_SCENE_CINEMATIC),
]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _img_tag(path: Path | None, missing="(generation failed)") -> str:
    if path and path.exists():
        return f'<img src="data:image/png;base64,{_b64(path)}" />'
    return f'<div class="miss">{missing}</div>'


def run() -> None:
    cells = []
    total = 0.0
    for cell_id, label, model, prompt in MATRIX:
        cost = MODEL_COST_GBP.get(model, 0.032)
        if not budget.can_spend(cost):
            print(f"[bakeoff] budget cap; skip {cell_id} ({model})")
            cells.append((cell_id, label, model, cost, None, "budget cap"))
            continue
        print(f"[bakeoff] {cell_id}: {model} ...", flush=True)
        t0 = time.time()
        try:
            path = fal_client.generate_image(
                prompt, model=model, aspect="square",
                output_dir=str(OUT), filename=f"{cell_id}.png",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {exc}")
            path = None
        dt = time.time() - t0
        if path and Path(path).exists():
            budget.record(cost, label=f"bakeoff:{cell_id}")
            total += cost
            # fal_client saves under output/bakeoff/fal_images/<cell_id>.png
            cells.append((cell_id, label, model, cost, Path(path), f"{dt:.0f}s"))
            print(f"  OK {path} ({dt:.0f}s)")
        else:
            cells.append((cell_id, label, model, cost, None, "no image"))
            print(f"  FAILED ({dt:.0f}s)")

    # Reference + current baseline preview for grounding.
    refs = sorted(REF_DIR.glob("ChatGPT Image*.png"))[:2]
    baseline = OUT.parent / "preview" / "memlock.png"

    html = ["""<!doctype html><meta charset=utf-8>
<title>Image bake-off — MEMLOCK PROTOCOL</title>
<style>
 body{background:#111;color:#eee;font:14px/1.5 system-ui,sans-serif;margin:24px}
 h1{font-weight:600} h2{color:#9cf;margin-top:32px}
 .grid{display:flex;flex-wrap:wrap;gap:20px}
 .cell{width:340px;background:#1b1b1b;border:1px solid #333;border-radius:10px;padding:12px}
 .cell img,.miss{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#000}
 .miss{display:flex;align-items:center;justify-content:center;color:#a55}
 .lab{font-weight:600;margin:8px 0 2px} .meta{color:#888;font-size:12px}
 .ref img{aspect-ratio:auto}
</style>
<h1>Image bake-off — one concept, model × prompt</h1>
<p class=meta>Concept: <b>MEMLOCK PROTOCOL</b> (sahil_twitter). Pick the winning
direction; I standardise it across all brands. Cost recorded below.</p>
"""]

    html.append("<h2>Your reference target + current baseline</h2><div class='grid ref'>")
    for r in refs:
        html.append(f"<div class=cell>{_img_tag(r)}<div class=lab>REFERENCE (target look)</div>"
                    f"<div class=meta>{r.name}</div></div>")
    html.append(f"<div class=cell>{_img_tag(baseline)}<div class=lab>CURRENT PIPELINE (rejected)</div>"
                f"<div class=meta>seedream45, today's output</div></div>")
    html.append("</div>")

    html.append("<h2>Bake-off candidates</h2><div class=grid>")
    for cell_id, label, model, cost, path, note in cells:
        html.append(
            f"<div class=cell>{_img_tag(path)}"
            f"<div class=lab>{label}</div>"
            f"<div class=meta>{model} · £{cost:.3f} · {note}</div></div>"
        )
    html.append("</div>")
    html.append(f"<p class=meta>Total spend this bake-off: <b>£{total:.2f}</b></p>")

    out_html = OUT / "index.html"
    out_html.write_text("\n".join(html), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Bake-off HTML: {out_html}")
    print(f"Total spend:   £{total:.2f}")


if __name__ == "__main__":
    run()
