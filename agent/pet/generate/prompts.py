"""Prompt builders for pet generation.

Two prompt shapes: a *base* prompt (prompt-only, produces the canonical look the
user picks between) and per-*state* *row* prompts (grounded on the chosen base,
produce one horizontal strip of N poses). Prompts stay concise and
sprite-production oriented; the identity lock and "one transparent row" framing
matter more than flowery description.

We generate the full petdex/Codex nine-state set (see
:data:`agent.pet.generate.atlas.ROW_SPECS`) so a hatched pet is a valid
``petdex submit`` spritesheet.
"""

from __future__ import annotations

# What each petdex/Codex state should depict (kept short — these go straight into
# the row prompt). Phrased to avoid the common sprite-gen failure modes (detached
# effects, motion lines, shadows). Critical distinction: ``running`` is the
# *working* state (in place), while ``running-right`` / ``running-left`` are the
# actual directional walk/run cycles.
STATE_ACTIONS: dict[str, str] = {
    "idle": "a calm idle loop: subtle breathing, a tiny blink or gentle bob, no big gestures",
    "running-right": (
        "a sideways walk/run locomotion cycle moving to the RIGHT: the character "
        "faces and travels right with clear directional steps, a smooth gait loop"
    ),
    "running-left": (
        "a sideways walk/run locomotion cycle moving to the LEFT: the character "
        "faces and travels left with clear directional steps (the mirror of the "
        "right-facing run)"
    ),
    "waving": "a friendly greeting: raising a paw/hand/limb to wave, clear up-and-down gesture",
    "jumping": "a happy celebration jump: anticipation, lift off the ground, peak, and land",
    "failed": "a sad or deflated reaction: slumped, dejected, small frown — readable but not noisy",
    "waiting": (
        "an expectant 'waiting on you' pose: looking up/out as if asking for input "
        "or approval — distinct from idle and review"
    ),
    "running": (
        "focused active work, staying IN PLACE (NOT walking or foot-running): "
        "leaning in, concentrating, busy 'thinking / processing / typing' energy"
    ),
    "review": "careful inspection: a focused lean, head tilt, studying something intently",
}

_STYLE_HINTS: dict[str, str] = {
    # Default to the popular petdex look: crisp 16-bit PIXEL ART, not the smooth
    # 2D illustration (let alone 3D render) gpt-image reaches for by default.
    "auto": (
        " Style: crisp 16-bit PIXEL-ART game sprite — visible square pixels, a small "
        "limited palette, clean dark outline, flat cel shading, chunky chibi "
        "proportions, like a classic SNES/JRPG party member or a petdex.dev mascot. "
        "Absolutely NOT 3D-rendered, NOT a smooth painted or vector illustration, "
        "NOT photorealistic — no soft gradients, no realistic lighting, no figurine look."
    ),
    "pixel": " Render in clean 16-bit pixel-art style with visible square pixels and a limited palette.",
    "plush": " Render as a soft plush toy.",
    "clay": " Render as a claymation / soft 3D clay figure.",
    "sticker": " Render as a glossy die-cut sticker.",
    "flat-vector": " Render in flat vector mascot style.",
    "3d-toy": " Render as a glossy 3D toy.",
    "painterly": " Render in a soft painterly style.",
}

_BACKGROUND = (
    "Center one full-body character on a flat, uniform, high-contrast chroma-key "
    "background (prefer pure hot magenta #FF00FF unless that color appears on "
    "the character). The background must completely surround the character: one "
    "even color with NO gradient, vignette, texture, pattern, scenery, shadow, "
    "ground line, frame, or border, so it keys out cleanly. The background color "
    "must not appear anywhere on the character itself. No text, no labels."
)


def style_hint(style: str | None) -> str:
    return _STYLE_HINTS.get((style or "auto").strip().lower(), "")


# Per-draft nudges so the 4 base options are actually distinct — gpt-image returns
# near-duplicates for a single prompt. We vary the *look* (palette, build,
# expression, accents), NOT the pose, so the chosen base still grounds clean,
# consistent animation rows.
BASE_VARIATIONS: tuple[str, ...] = (
    "",
    "a distinctly different colour palette and markings",
    "rounder, chunkier chibi proportions and a bigger head",
    "a different face and expression, with unique accent/accessory details",
    "a leaner, taller build and an alternate colour scheme",
    "bolder, more saturated colours and a playful expression",
)


def build_base_prompt(concept: str, *, style: str | None = "auto", variation: str = "") -> str:
    """The base look: a single, clean, centered full-body mascot.

    *variation* differentiates one draft from the next (see :data:`BASE_VARIATIONS`).
    """
    concept = (concept or "a cute friendly mascot creature").strip()
    nudge = f" Make this design distinct: {variation}." if variation else ""
    return (
        f"A cute, characterful mascot pet: {concept}. "
        "Compact, whole-body silhouette that reads clearly at small size, "
        "appealing face, simple consistent palette. "
        # A neutral, symmetric, at-rest stance makes the cleanest identity anchor
        "Neutral front-facing standing pose, upright and symmetric, arms/limbs "
        "relaxed at the sides, feet together on the ground, any cape/accessories "
        "hanging straight and still."
        f"{nudge} "
        f"{_BACKGROUND}{style_hint(style)}"
    )


def build_row_prompt(state: str, frame_count: int, concept: str, *, style: str | None = "auto") -> str:
    """A row strip: *frame_count* poses of the SAME character, left→right.

    The attached base image is the identity source of truth; the prompt locks
    species, palette, face, and props to it.
    """
    action = STATE_ACTIONS.get(state, "a simple idle pose")
    concept = (concept or "the mascot").strip()
    return (
        f"Using the attached reference image as the exact same character "
        f"(same species, face, colors, markings, proportions, and props), "
        f"draw a single horizontal strip of {frame_count} animation frames showing {action}. "
        f"The {frame_count} poses must be evenly spaced left to right, each fully separated "
        "by clear empty chroma-key gutters; silhouettes must NEVER touch, overlap, "
        "share a shadow, share a ground line, share motion trails, or merge into "
        "one connected shape. "
        # Registration: a clean sprite sheet keeps the character locked in place
        # so only the action moves — this is what stops the loop sliding/pulsing.
        "REGISTRATION (critical): the character is the SAME height and SAME width "
        "in every frame, drawn at the SAME scale, centered over the SAME point, "
        "with all feet resting on ONE shared horizontal ground line across the "
        "whole strip. Keep the body's center, size, and stance fixed frame to "
        "frame — ONLY the limbs/features the action needs may move. Capes, cloaks, "
        "bags, and scarves stay in the SAME place and shape every frame (no "
        "swinging, flowing, or drifting) unless the action itself requires it. No "
        "pose is cropped at the strip edges. "
        f"{_BACKGROUND}{style_hint(style)}"
    )
