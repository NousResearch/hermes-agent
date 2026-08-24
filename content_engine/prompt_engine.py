"""Assemble Seedream prompts from the baoyu dimensional method.

The dimensional method (per baoyu_refs/references/style-presets.md):
    prompt = Type (7 options) x Style (23 options) x optional Palette (4 options)
    x per-brand art-direction reference.

Each brand declares its preferred PRESETS for scene-style and data-style
content. ``build_image_prompt`` reads the type from ``content_router`` and
emits a fully-art-directed prompt by embedding the upstream style/palette
files verbatim — never paraphrased, since the exact wording reproduces the
canonical baoyu look.

Text rules (the OCR regen gate enforces these):
  - SCENE and HERO: no text in the image. OCR scans for any text; if any
    significant text is found, the image is rejected and regenerated.
  - DATA types (infographic, comparison, framework, flowchart, timeline):
    short labels (1-3 words) and key-value pairs only. OCR verifies them.
    Title is included as a section header, not rendered as a banner.
"""

from __future__ import annotations
from typing import Tuple

import baoyu_loader as bl
from content_router import content_type_for
from infographic_content import build_structured

# Per-brand preset map. Each brand picks a canonical scene-style and a
# canonical data-style from the upstream baoyu preset names (see
# baoyu_refs/SOURCE.md for the full table). The "ref" string is the
# brand-specific art direction appended to every prompt.
BRAND_STYLE_MAP = {
    "plenishd": {
        "scene":  "lifestyle",          # scene + watercolor — food/kitchen feel
        "data":   "warm-knowledge",     # infographic + vector-illustration + warm
        "model":  "seedream45",
        "ref":    "warm appetising UK kitchen, soft natural light, hand-painted feel",
    },
    "coachos": {
        "scene":  "storytelling",       # scene + warm — Saturday morning pitch
        "data":   "ink-notes-flow",     # flowchart + ink-notes + mono-ink
        "model":  "seedream45",
        "ref":    "B&W documentary grassroots football, restrained, one green accent",
    },
    "matchdaymaestro": {
        "scene":  "cinematic",          # scene + screen-print — matchday poster
        "data":   "editorial-poster",   # comparison + screen-print
        "model":  "seedream45",
        "ref":    "retro halftone matchday poster, off-black background, burnt orange and crimson accents",
    },
    "sahil_twitter": {
        "scene":  "dark-hud",           # infographic + dark-cyberpunk-hud
        "data":   "dark-hud",           # same dark-hud for technical posts
        "model":  "seedream45",
        "ref":    "dev-tool launch, system architecture, performance metrics, command-line announcements, build-in-public",
    },
    "sahil_linkedin": {
        "scene":  "business-compare",   # comparison + elegant
        "data":   "ink-notes-framework",# framework + ink-notes + mono-ink
        "model":  "seedream45",
        "ref":    "HBR-style restrained editorial, monochrome with one indigo accent",
    },
}
_DEFAULT = BRAND_STYLE_MAP["sahil_twitter"]
_DATA_TYPES = {"infographic", "comparison", "framework", "flowchart", "timeline"}
_TEXTLESS_TYPES = {"scene", "hero"}
_ASPECT = {"instagram": "portrait_4_5", "tiktok": "portrait_9_16", "linkedin": "landscape"}


def _aspect(platform: str) -> str:
    return _ASPECT.get((platform or "").lower(), "square")


def _expected_strings(draft: dict, ctype: str) -> list:
    """Text the model must render correctly (for the OCR gate).

    SCENE and HERO are TEXTLESS — the OCR post-check rejects any image with
    significant text. DATA types render short labels only. The title is
    the primary expected text. Body labels are returned but the prompt
    also includes them so the model knows what to render.
    """
    if ctype in _TEXTLESS_TYPES:
        return []
    if (draft.get("visual_text") or "").lower() == "none":
        return []
    out = []
    t = (draft.get("title") or draft.get("topic") or "").strip()
    if 0 < len(t) <= 40:
        out.append(t)
    # Dedupe, cap at 4
    seen, deduped = set(), []
    for s in out:
        s_norm = s.strip().lower()
        if s_norm and s_norm not in seen and len(deduped) < 4:
            seen.add(s_norm)
            deduped.append(s)
    return deduped


def _expected_body_labels(body: str, max_labels: int = 5, max_len: int = 24) -> list:
    """Short body labels to render in the image. These are returned separately
    from _expected_strings so the OCR check can be looser (the model often
    paraphrases or abbreviates)."""
    import re
    parts = re.split(r"[.;\n]|\bbut\b|\bwhereas\b|\bwhile\b", body or "")
    out = []
    for p in parts:
        p = p.strip(" ,;:.-")
        if 1 < len(p.split()) <= 4 and len(p) <= max_len:
            out.append(p)
        if len(out) >= max_labels:
            break
    return out


def _structured_block(draft: dict, ctype: str) -> str:
    """Build the data-type content block. Emits short labels, not full sentences."""
    s = build_structured(draft, ctype)
    title = (s.get("title") or "").strip()
    parts = [f"{ctype.upper()} illustration titled '{title}'."] if title else [f"{ctype.upper()} illustration."]
    if s.get("items"):
        parts.append("Key-value rows:")
        for i in s["items"][:4]:
            parts.append(f"  - {i['name'][:18]}: {i['value'][:24]}")
    if s.get("steps"):
        parts.append("Ordered steps:")
        for n, step in enumerate(s["steps"][:5], 1):
            parts.append(f"  {n}. {step[:30]}")
    if s.get("nodes"):
        parts.append("Nodes:")
        for n, node in enumerate(s["nodes"][:5], 1):
            parts.append(f"  [{n}] {node[:20]}")
    if s.get("points"):
        parts.append("Points:")
        for p in s["points"][:5]:
            parts.append(f"  - {p[:30]}")
    if s.get("takeaway"):
        parts.append(f"Bottom: {s['takeaway'][:50]}")
    return "\n".join(parts)


_MJ_REGISTRY_CACHE = None


def _mj_style_for(draft: dict) -> str:
    """Pick a Midjourney style-card treatment for this image from the 9.5k registry.

    Deterministic per (slug, id): same image regenerates the same style, but
    hero/section images within one article get different treatments. Returns
    an empty string when the registry is unavailable.
    """
    global _MJ_REGISTRY_CACHE
    try:
        if _MJ_REGISTRY_CACHE is None:
            import json as _json
            from pathlib import Path as _P
            _reg_path = _P(__file__).parent / "style_registry" / "registry-full.json"
            _MJ_REGISTRY_CACHE = _json.loads(_reg_path.read_text())["styles"]
        reg = _MJ_REGISTRY_CACHE
        seed_src = f"{draft.get('slug','')}:{draft.get('id','')}"
        key = sum(ord(c) for c in seed_src)
        keys = list(reg.keys())
        card = reg[keys[key % len(keys)]]
        bits = [card.get("style_label", ""), card.get("medium", ""),
                card.get("palette_mood", ""), ", ".join(card.get("best_for", [])[:2])]
        return " · ".join(b for b in bits if b)[:300]
    except Exception:
        return ""


def _scene_block(draft: dict, ctype: str, brand_ref: str) -> str:
    subject = (draft.get("visual_description") or draft.get("title")
               or draft.get("topic") or "").strip()
    return (
        f"{ctype.upper()} scene.\n"
        f"Subject: {subject}.\n"
        f"World: {brand_ref}.\n"
        f"ABSOLUTELY NO TEXT anywhere in the image. No titles, captions, labels, "
        f"watermarks, page numbers, or any other text. Pure visual only."
    )


def _prompt_parts(ctype: str, preset: dict, content_block: str,
                  brand_ref: str, draft: dict | None = None) -> list:
    """Assemble the prompt section list."""
    parts = [f"TYPE: {ctype.upper()} illustration."]
    style = preset.get("style", "")
    if style:
        parts.append(f"STYLE:\n{bl.style_block(style)}")
    palette = preset.get("palette")
    if palette:
        pb = bl.palette_block(palette)
        if pb:
            parts.append(f"PALETTE:\n{pb}")
    parts.append(content_block)
    mj = _mj_style_for(draft)
    if mj:
        parts.append(f"TREATMENT VARIATION: {mj}. "
                     "Apply this as the visual rendering treatment only - keep the "
                     "structured layout and content above intact.")
    if ctype in _TEXTLESS_TYPES:
        parts.append("TEXT RULE: No text in the image. No labels, no captions, no watermarks, no titles. Pure visual scene only.")
    else:
        parts.append("TEXT RULE: Render only the structured labels listed below. No additional captions, watermarks, or marketing text.")
    parts.append(bl.universal_rules())
    return parts


def build_image_prompt(draft: dict) -> Tuple[str, str, str, list]:
    """Return (prompt, aspect, model_key, expected_strings)."""
    brand = (draft.get("brand") or "").lower()
    spec = BRAND_STYLE_MAP.get(brand, _DEFAULT)
    ctype = content_type_for(draft)
    # Position-level overrides win: the illustrator rotates presets/palettes
    # per image so hero and section images never share one treatment.
    preset_name = draft.get("_preset_override") or (
        spec["data"] if ctype in _DATA_TYPES else spec["scene"])
    preset = bl.preset_for(preset_name)
    if draft.get("_palette_override"):
        preset["palette"] = draft["_palette_override"]
    preset.setdefault("type", ctype)
    aspect = _aspect(draft.get("platform"))

    if ctype in _DATA_TYPES:
        content_block = _structured_block(draft, ctype)
    else:
        content_block = _scene_block(draft, ctype, spec["ref"])

    parts = _prompt_parts(ctype, preset, content_block, spec["ref"], draft)
    expected = _expected_strings(draft, ctype)
    if expected:
        # Only the title is enforced via OCR; body labels are aspirational
        # (the model often paraphrases or abbreviates them).
        parts.append("Render the title as a top header in the image, exactly:\n"
                     f'  "{expected[0]}"\n'
                     "No other text. No captions, watermarks, page numbers.")

    prompt = "\n\n".join(parts)
    return prompt, aspect, spec["model"], expected
