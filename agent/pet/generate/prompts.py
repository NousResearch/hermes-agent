"""Prompt builders for quality-first pet generation.

The user selects a canonical base look, then reference-grounded one/two-pose
production assets supply the animation. Prompts stay concise and sprite-oriented;
identity, exact subject count, complete silhouettes, and empty isolation gutters
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

# Exact animation phases used by the quality-first hatch pipeline.  Large
# 4–8-pose strips are unreliable across providers: subjects are omitted, merged,
# or allowed to cross their imaginary cell boundaries.  Generating these phases
# in pairs keeps the provider's layout task small and lets local extraction
# require one subject in each half before any pixels reach the atlas.
POSE_PHASES: dict[str, tuple[str, ...]] = {
    "idle": (
        "neutral relaxed stance",
        "gentle inhale with the chest slightly raised",
        "breath crest with a tiny ear, tail, or wing response",
        "soft exhale",
        "small blink and subtle tail response",
        "return to the exact neutral stance",
    ),
    "running-right": (
        "right-facing run contact pose, front foot landing",
        "right-facing run recoil pose",
        "right-facing low passing pose",
        "right-facing airborne pose",
        "opposite foot contact pose",
        "opposite recoil pose",
        "opposite passing pose",
        "second airborne pose that loops cleanly to the first",
    ),
    "waving": (
        "friendly ready stance",
        "one paw, hand, or wing raised to begin a wave",
        "wave at its highest cheerful peak",
        "relaxed return toward neutral",
    ),
    "jumping": (
        "anticipation crouch",
        "strong takeoff",
        "high celebratory apex",
        "soft landing compression",
        "upright recovery",
    ),
    "failed": (
        "noticing a harmless mistake",
        "surprised reaction",
        "small backward wobble",
        "disappointed slump",
        "brief disappointed hold",
        "steadying the body",
        "recovering confidence",
        "calm return with no injury or gore",
    ),
    "waiting": (
        "patient neutral stance",
        "quiet glance to the side",
        "tiny weight shift",
        "single relaxed blink",
        "soft breath",
        "return to patient neutral, awake and calm",
    ),
    "running": (
        "focused work-in-place stance",
        "energetic forward lean without travelling",
        "active tool-free processing gesture",
        "peak energetic work pose",
        "small recoil",
        "return to the focused stance on the same pivot",
    ),
    "review": (
        "focused review stance",
        "eyes scanning slightly left",
        "eyes scanning slightly right",
        "thoughtful consideration",
        "small approving nod",
        "return to focused review with no external prop",
    ),
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
        "CANONICAL CHARACTER IDENTITY SOURCE. "
        f"Exactly one cohesive stylized mascot pet based on: {concept}. "
        "Interpret comma-separated animals as traits of one hybrid, never as separate creatures. "
        "Honor the requested tone and mood exactly (cute, eerie, scary, menacing, whimsical, etc.) "
        "while staying non-graphic. "
        "Compact, whole-body silhouette that reads clearly at small size, "
        "clear readable facial features, symmetrical anatomy, simple consistent palette, "
        "and no duplicate anatomy or extra creature. "
        # A directional but quiet anchor gives locomotion edits useful anatomy
        # while remaining stable for front-facing reaction poses.
        "Neutral standing pose facing right in a clean side or slight three-quarter game view, "
        "arms/limbs relaxed, feet together on the same invisible pivot, and any cape/accessories "
        "hanging straight and still. Show the entire body with generous empty padding."
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


def build_pose_segment_prompt(
    state: str,
    phases: tuple[str, ...],
    concept: str,
    *,
    style: str | None = "auto",
    retry_reason: str = "",
) -> str:
    """Build a strict one/two-pose reference edit prompt.

    The selected draft is the immutable identity anchor.  A pair is deliberately
    specified as two independent halves rather than as a sprite sheet: this
    produces a real empty center gutter that local QA can verify without ever
    guessing where one pose ends and its neighbour begins.
    """
    if not 1 <= len(phases) <= 2:
        raise ValueError("pose segment must contain one or two phases")

    identity = (
        "Use the attached reference as the exact same single character: preserve "
        "its species, face, anatomy, proportions, palette, markings, accessories, "
        "camera, emotional tone, and pixel-art treatment. "
    )
    restrictions = (
        "Show every limb, wing, horn, ear, tail, cape, and foot completely. Keep "
        "the whole silhouette inside its assigned area with generous empty padding. "
        "Use one perfectly flat, uniform chroma-key field behind everything, matching "
        "the reference background. Draw no floor, ground line, cast shadow, glow, "
        "motion trail, scenery, prop, text, label, panel, border, divider, duplicate "
        "anatomy, or extra creature. The background color must not appear on the character."
    )
    retry = f" CORRECTION: {retry_reason.strip()}" if retry_reason.strip() else ""
    concept_lock = (
        f" The character represents one cohesive pet based on {concept.strip() or 'the selected concept'}; "
        "comma-separated animals are traits of that one hybrid, never separate creatures."
    )

    if len(phases) == 1:
        layout = (
            "Render exactly ONE complete full-body pose, centered in the image with "
            f"generous empty space on every side. Motion phase: {phases[0]}."
        )
    else:
        layout = (
            "Render exactly TWO complete full-body poses total in one horizontal image. "
            "Pose A is centered wholly inside the LEFT half; pose B is centered wholly "
            "inside the RIGHT half. Leave a wide, completely empty chroma-key gutter "
            "through the center. The two silhouettes must never touch, overlap, share an "
            f"effect, or cross the midpoint. Pose A: {phases[0]}. Pose B: {phases[1]}."
        )

    return f"POSE PRODUCTION ASSET. {identity}{layout}{concept_lock} {restrictions}{style_hint(style)}{retry}"
