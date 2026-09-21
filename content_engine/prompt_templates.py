"""Per-brand, per-post generation prompts.

Builds a textless, photographic scene prompt for the base image: brand
art-direction (from brand_style) + the post's own visual brief + hard
"no text" discipline. The headline is added later as a Pillow overlay, so
the model is never asked to render legible words.
"""
from __future__ import annotations

from brand_style import NEGATIVE_PROMPT, for_brand

# Pillar -> compositional hint, so the scene varies with the post's intent
# instead of every post for a brand looking identical.
_PILLAR_SCENE = {
    "data": "a single striking focal subject with clean negative space for an overlaid stat",
    "data_driven": "a single striking focal subject with clean negative space for an overlaid stat",
    "tutorial": "an over-the-shoulder process shot, hands-on, instructional framing",
    "howto": "an over-the-shoulder process shot, hands-on, instructional framing",
    "promotion": "a hero product-in-context shot, aspirational, centred composition",
    "sly_product": "a hero product-in-context shot, aspirational, centred composition",
}

_DEFAULT_SCENE_HINT = "an evocative editorial scene with clear negative space at the top for an overlaid headline"

# Briefs containing these tokens ask the model to render text/UI/branding,
# which cheap diffusion models butcher. Such legacy briefs are discarded.
_BANNED_BRIEF_TOKENS = (
    "text", "typography", "font", "monospace", "code overlay", "branding",
    "logo", "ui", "interface", "quote-style", "caption", "title", "word",
    "layout", "headline", "label", "screenshot",
)


def _safe_brief(brief: str) -> str:
    low = brief.lower()
    return "" if any(tok in low for tok in _BANNED_BRIEF_TOKENS) else brief


def build_scene_prompt(brand: str, visual_brief: str = "", pillar: str = "", body_text: str = "") -> str:
    """Return a textless base-image prompt for FAL/local generation.

    The scene is driven by art-direction (``visual_brief``), NOT the literal
    post copy: feeding body_text in makes diffusion models try to render those
    words. ``body_text`` is therefore ignored for the scene; the cron agent is
    expected to supply a ``visual_brief``. Without one we fall back to the
    brand's stock scene + a pillar-appropriate composition hint -- always safe
    and textless.
    """
    visuals = for_brand(brand)
    brief = _safe_brief((visual_brief or "").strip())
    hint = _PILLAR_SCENE.get((pillar or "").lower(), _DEFAULT_SCENE_HINT)

    scene = f"{visuals['scene']}. {('Subject: ' + brief[:200] + '. ') if brief else ''}{hint}."
    return (
        f"{scene} Professional photography, high detail, sharp focus, well composed, "
        "cinematic lighting. A clean photographic scene with absolutely no text, "
        "no words, no letters, no UI and no graphics anywhere in the image."
    )


def negative_prompt() -> str:
    return NEGATIVE_PROMPT
