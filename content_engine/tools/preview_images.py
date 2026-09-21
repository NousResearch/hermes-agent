#!/usr/bin/env python3
"""Preview Gallery: generate fresh images through the new content engine pipeline.

Creates brand-new draft content for 6 targets, runs each through the Seedream
pipeline + OCR gate, and saves results to output/preview/<brand>_<draft_id>.png.

Usage:
    cd content_engine && PYTHONPATH=. python tools/preview_images.py

Output: prints a table of (target | brand | status | path | cost)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure parent is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draft_media import generate_post_image

GALLERY = Path(__file__).resolve().parent.parent / "output" / "preview"
GALLERY.mkdir(parents=True, exist_ok=True)

DRAFTS = [
    # ── GitHub repos ──
    {
        "id": "memlock",
        "brand": "sahil_twitter",
        "platform": "twitter",
        "title": "MEMLOCK PROTOCOL",
        "body_text": "Lock context. Unlock intelligence. Re-assert standing instructions after context compaction. Hermes plugin. Pin once. Survive every compaction.",
        "format": "infographic",
        "pillar": "build_in_public",
        "visual_description": "",
    },
    {
        "id": "toolaria",
        "brand": "sahil_twitter",
        "platform": "twitter",
        "title": "RESCUE HANDLE",
        "body_text": "Tool result too large. Store in SHA256 blob. Hand the model a 1.7k handle. Outline. Search. Pass by reference. 99x smaller context.",
        "format": "infographic",
        "pillar": "build_in_public",
        "visual_description": "",
    },
    {
        "id": "pm_skills",
        "brand": "sahil_linkedin",
        "platform": "linkedin",
        "title": "PM SKILLS MARKETPLACE",
        "body_text": "Discovery. Strategy. Execution. Launch. Growth. Sixty-eight skills. Forty-two workflows. Nine plugins. One CLI. Rigor built into the daily workflow.",
        "format": "framework",
        "pillar": "product_management",
        "visual_description": "",
    },
    # ── Brands ──
    {
        "id": "coachos_session",
        "brand": "coachos",
        "platform": "twitter",
        "title": "ONE SESSION, 12 PLAYERS",
        "body_text": "Drag-drop drills. Set rotation patterns. Auto-calculate rest ratios. Share the plan to parents. Every Saturday, organised.",
        "format": "infographic",
        "pillar": "coaching",
        "visual_description": "",
    },
    {
        "id": "maestro_wc26",
        "brand": "matchdaymaestro",
        "platform": "twitter",
        "title": "WORLD CUP 2026",
        "body_text": "Forty-eight teams. One hundred four matches. Three host nations. Pick your winner before the first kick. Bracket. Trivia. Live scores.",
        "format": "infographic",
        "pillar": "matchday",
        "visual_description": "",
    },
    {
        "id": "plenishd_kitchen",
        "brand": "plenishd",
        "platform": "twitter",
        "title": "SNAP. SAY. STOCK IT.",
        "body_text": "Open the cupboard. Say what is low. Pantry updates. Walk in with shopping bags. Say what you bought. Pantry updates. No scanning. No typing.",
        "format": "infographic",
        "pillar": "food",
        "visual_description": "",
    },
]


def run():
    results = []
    total_cost = 0.0
    for i, draft in enumerate(DRAFTS):
        tag = draft["id"]
        brand = draft["brand"]
        print(f"\n[{i + 1}/{len(DRAFTS)}] {tag} ({brand})...", end=" ", flush=True)
        try:
            path = generate_post_image(draft)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"tag": tag, "brand": brand, "status": "error", "path": None})
            continue
        if path and os.path.exists(path):
            # Copy to gallery with stable name
            import shutil
            dst = GALLERY / f"{tag}.png"
            shutil.copy2(path, dst)
            print(f"OK -> {dst}")
            results.append({"tag": tag, "brand": brand, "status": "ok", "path": str(dst)})
        else:
            print("FAILED (no image returned)")
            results.append({"tag": tag, "brand": brand, "status": "failed", "path": path})

    # Summary
    print("\n" + "=" * 72)
    print(f"Gallery: {GALLERY}/")
    print(f"{'Target':<28} {'Brand':<18} {'Status':<8} Path")
    print("-" * 72)
    for r in results:
        p = os.path.basename(r["path"]) if r["path"] else "-"
        print(f"{r['tag']:<28} {r['brand']:<18} {r['status']:<8} {p}")
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print("=" * 72)
    print(f"{ok_count}/{len(results)} generated successfully in {GALLERY}/")


if __name__ == "__main__":
    run()
