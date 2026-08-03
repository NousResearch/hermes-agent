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
    "Center the character on a SINGLE flat, uniform, high-contrast chroma-key "
    "background — pure hot magenta #FF00FF (only if magenta appears on the "
    "character, use pure green #00FF00 instead). The background is ONE continuous "
    "even color that completely surrounds the character with NO gradient, "
    "vignette, texture, pattern, scenery, shadow, ground line, frame, border, "
    "panel, comic cell, gutter line, grid, or divider of any kind, so it keys out "
    "cleanly. The background color must not appear anywhere on the character. "
    "No text, no labels, no speech bubbles, no UI."
)


def style_hint(style: str | None) -> str:
    return _STYLE_HINTS.get((style or "auto").strip().lower(), "")


# Row strips are generated on the wider landscape canvas (see imagegen.generate /
# orchestrate). The extra width is what lets each pose stay a healthy size AND
# leave a real gutter — used here only to cite concrete pixel numbers.
_ASSUMED_STRIP_WIDTH = 1536


def _spacing_spec(frame_count: int) -> tuple[int, int]:
    """(per-pose width px, gap px) for a row of *frame_count* poses.

    Pixel counts alone don't hold — the model fills each slot edge-to-edge with
    the full wingspan, so neighbors touch even when bodies are spaced. The lever
    that works is proportional containment on a wide canvas: give each pose its
    own equal cell and keep the ENTIRE silhouette (wings/tail/halo included)
    inside it. On the 1536px landscape strip ~70% occupancy still leaves a
    generous gutter, so the pet stays a normal, good-looking size — no shrinking.
    """
    slots = max(1, frame_count)
    slot_w = _ASSUMED_STRIP_WIDTH / slots
    pose_px = round(slot_w * 0.7)
    gap_px = max(48, round(slot_w * 0.3))
    return pose_px, gap_px


# Per-draft nudges so the 4 base options are actually distinct — gpt-image returns
# near-duplicates for a single prompt. We vary the *look* (palette, build,
# expression, accents), NOT the pose, so the chosen base still grounds clean,
# consistent animation rows.
BASE_VARIATIONS: tuple[str, ...] = (
    "",
    "a distinctly different colour palette and markings",
    "a heavier, broader silhouette with sturdier proportions",
    "a different facial structure and expression matching the concept tone, with unique accent/accessory details",
    "a leaner, taller build and an alternate colour scheme",
    "bolder, more saturated colours and a stronger expression matching the concept tone",
)


def build_base_prompt(concept: str, *, style: str | None = "auto", variation: str = "") -> str:
    """The base look: a single, clean, centered full-body mascot.

    *variation* differentiates one draft from the next (see :data:`BASE_VARIATIONS`).
    """
    concept = (concept or "a distinctive mascot creature").strip()
    nudge = f" Make this design distinct: {variation}." if variation else ""
    return (
        f"A stylized mascot pet character: {concept}. "
        "Honor the requested tone and mood exactly (cute, eerie, scary, menacing, whimsical, etc.) "
        "while staying non-graphic. "
        "Compact, whole-body silhouette that reads clearly at small size, "
        "clear readable facial features, simple consistent palette. "
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
    pose_px, gap_px = _spacing_spec(frame_count)
    return (
        f"Using the attached reference image as the exact same character "
        f"(same species, face, colors, markings, proportions, and props), "
        "preserving the same emotional tone/mood (e.g., scary stays scary, cute stays cute), "
        f"draw a single WIDE horizontal strip of {frame_count} animation frames showing {action}. "
        f"LAYOUT: arrange {frame_count} poses in ONE horizontal row at equal spacing, "
        "each pose centered in its own imaginary equal region. Draw NO panel borders, "
        "NO comic cells, NO boxes, NO vertical divider/gutter lines, NO grid, NO frame "
        "outlines between poses — the backdrop is one unbroken flat field behind all of them. "
        "Fill the WHOLE strip with the SAME single flat chroma-key color as the attached "
        "reference image's background (identical hue in every frame, no per-pose color shifts). "
        f"SPACING (critical): draw each pose at a consistent, healthy, clearly "
        f"visible size (roughly {pose_px}px wide on a {_ASSUMED_STRIP_WIDTH}px "
        f"strip) — do NOT shrink it tiny — but keep its ENTIRE silhouette "
        f"(wings, tail, halo, horns, cape, every appendage) fully INSIDE its own "
        f"cell. Leave at least {gap_px}px of empty chroma-key background between "
        f"neighboring silhouettes at their closest point (wingtip to wingtip), and "
        f"the same empty margin before the first pose and after the last. If a wing, "
        f"cape, or tail would reach into a neighbor, FOLD or angle it inward rather "
        f"than letting it cross the gap. Silhouettes must NEVER touch, overlap, "
        f"share a shadow, share a ground line, share motion trails, or merge into "
        f"one connected shape. "
        # Registration: a clean sprite sheet keeps the character locked in place
        # so only the action moves — this is what stops the loop sliding/pulsing.
        "REGISTRATION (critical): the character is the SAME height and SAME width "
        "in every frame, drawn at the SAME scale, centered over the SAME point, "
        "with all feet aligned to the SAME invisible horizontal baseline across the "
        "whole strip — this baseline is conceptual ONLY: draw NO ground line, floor, "
        "platform, horizon, or contact shadow beneath the feet. Keep the body's center, size, and stance fixed frame to "
        "frame — ONLY the limbs/features the action needs may move. Capes, cloaks, "
        "bags, and scarves stay in the SAME place and shape every frame (no "
        "swinging, flowing, or drifting) unless the action itself requires it. No "
        "pose is cropped at the strip edges. "
        f"{_BACKGROUND}{style_hint(style)}"
    )
