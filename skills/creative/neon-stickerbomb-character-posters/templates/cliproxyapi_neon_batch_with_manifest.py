#!/usr/bin/env python3
"""Reusable neon sticker-bomb batch generator with sidecar manifest.

Copy this template before use. Fill ITEMS with index/title/semantic/slug/prompt.
It generates one image per request through CLIProxyAPI Responses API + image_generation,
saves each PNG as soon as it completes, and updates a manifest after every item.

Intended for Nick's workflow where later commands like `发布 1 3 4` map to the most
recent manifest rather than Discord ordering or Eagle ordering.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "http://127.0.0.1:8317/v1"
KEY = "sk-hermes-cliproxyapi"  # local non-secret proxy key; do not print external secrets
OUTDIR = Path.home() / ".hermes/profiles/jea/cache/images"
STATE = Path.home() / ".hermes/profiles/jea/state"
BATCH_ID = "neon_batch_REPLACE_WITH_TIMESTAMP_OR_TOPIC"
MANIFEST_PATH = STATE / f"{BATCH_ID}.json"
MODEL = "gpt-image-2"
SIZE = "1024x1536"
QUALITY = "medium"

OUTDIR.mkdir(parents=True, exist_ok=True)
STATE.mkdir(parents=True, exist_ok=True)

STYLE_BLOCK = """
STYLE MECHANICS: high-saturation glossy neon cyber-pop sticker-bomb poster, thick bold black manga/comic ink outlines, sticker-cut contour edges, sharp cel-shading, deep black/violet/crimson shadow blocks, wet glossy white highlights on hair, skin, clothing, props, metal, plastic, glass and sticker surfaces, electric cyan and hot magenta rim lighting, acid accent highlights, chromatic aberration, halftone dot shadows, spray-paint grain, rough offset-print registration, screenprint texture, glitch-rim outlines, torn vinyl decals and aggressive sticker-bomb layering.
Important creator-credit integration: include exact readable text “NickZag” as an IN-SCENE GRAPHIC DESIGN ELEMENT, not a watermark or overlay. Integrate it once as a small handwritten tag on a physical object in the scene: prop label, tape strip, clothing patch, sticker, card corner, lens reflection, or poster fragment. Recognizable but not dominant; it shares perspective, lighting, print texture and distortion.
FASHION / ACCESSORY REMIX: preserve the subject’s core silhouette and color language, but remix with glossy covered street-fashion details: technical jacket panels, utility straps, metallic buckles, belts, enamel pins, patches, gloves, rings, holographic labels, printed fabric graphics and prop tags. Stylish, character-faithful, covered; no unnecessary nudity.
Negative constraints: no photorealism, no watercolor, no muted colors, no clean official key art, no boring centered front-facing pose, no repeated standard poster layout, no watermark, no floating signature overlay, no separate logo stamp, no large author mark, no sexualization, no nudity, no cleavage emphasis, no exposed intimate skin, no adult-rated presentation, no deformed hands, no extra fingers, no loss of subject identity.
""".strip()

ITEMS = [
    {
        "index": 1,
        "title": "Character Name / 中文名",
        "semantic": "Character-Name_中文名_neon-stickerbomb-cyber-pop-specific-composition-poster",
        "slug": "character_specific_composition",
        "prompt": f"""Vertical 3:4 glossy neon cyber-pop sticker-bomb character poster of Character Name / 中文名, an adult covered non-explicit character.
COMPOSITION — DISTINCT ARCHETYPE NAME: describe one visibly unique poster skeleton, subject placement, foreground prop, typography motion path, palette, and identity cues.
{STYLE_BLOCK}""",
    },
]


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def check_models() -> None:
    req = urllib.request.Request(BASE + "/models", headers={"Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read(200)
    print("MODELS_OK", flush=True)


def generate_one(item: dict) -> None:
    payload = {
        "model": "gpt-5.5",
        "store": False,
        "instructions": "You must fulfill image generation requests by using the image_generation tool when provided. Generate exactly one image following the user prompt closely.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": item["prompt"]}],
            }
        ],
        "tools": [
            {
                "type": "image_generation",
                "model": MODEL,
                "size": SIZE,
                "quality": QUALITY,
                "output_format": "png",
                "background": "opaque",
                "partial_images": 1,
            }
        ],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "image_generation"}],
        },
    }
    req = urllib.request.Request(
        BASE + "/responses",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())

    for output_item in data.get("output") or []:
        if output_item.get("type") == "image_generation_call" and output_item.get("result"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = OUTDIR / f"cliproxyapi_{MODEL}-medium_{ts}_{item['slug']}.png"
            out.write_bytes(base64.b64decode(output_item["result"]))
            item["status"] = "done"
            item["path"] = str(out)
            item["elapsed_seconds"] = round(time.time() - start, 1)
            print("DONE", item["index"], item["title"], "elapsed", item["elapsed_seconds"], "saved", out, flush=True)
            return

    raise RuntimeError(f"no image_generation_call result for {item['slug']}: status={data.get('status')}")


def main() -> None:
    manifest = {
        "batch_id": BATCH_ID,
        "skill": "neon-stickerbomb-character-posters",
        "folder": "neon",
        "model": MODEL,
        "items": [dict(item, status="pending") for item in ITEMS],
    }
    save_manifest(manifest)
    check_models()

    for item in manifest["items"]:
        try:
            generate_one(item)
        except urllib.error.HTTPError as exc:
            item["status"] = "failed"
            item["error"] = f"HTTP {exc.code}: " + exc.read().decode("utf-8", "ignore")[:1000]
            print("FAIL", item["index"], item["slug"], item["error"], flush=True)
        except Exception as exc:  # keep partial batch manifest recoverable
            item["status"] = "failed"
            item["error"] = repr(exc)
            print("FAIL", item["index"], item["slug"], item["error"], flush=True)
        finally:
            save_manifest(manifest)

    print("MANIFEST", MANIFEST_PATH, flush=True)


if __name__ == "__main__":
    main()
