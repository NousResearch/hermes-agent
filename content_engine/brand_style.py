"""Central brand visual system for the content engine.

Single source of truth for how each brand looks: scene art-direction, ink
palette, the deterministic post-process "treatment", fonts and text colours.
Both prompt_templates.py (generation) and postprocess.py (finishing) read
from here so a brand's look stays consistent across the pipeline.

Deliberately NOT the nous-branding xerox/risograph underground aesthetic --
these treatments are tuned per brand (warm editorial, clean monochrome,
matchday punch, restrained professional).
"""
from __future__ import annotations

_FONT_DIR = "/usr/share/fonts/opentype/inter"

# Inter faces (installed via fonts-inter). PIL loads .otf fine.
FONTS = {
    "display": f"{_FONT_DIR}/InterDisplay-Black.otf",   # headlines
    "bold": f"{_FONT_DIR}/Inter-Bold.otf",
    "semibold": f"{_FONT_DIR}/Inter-SemiBold.otf",
    "medium": f"{_FONT_DIR}/Inter-Medium.otf",
    "regular": f"{_FONT_DIR}/Inter-Regular.otf",
}

# Per-brand visual identity.
#   scene:      art-direction phrase appended to every generation prompt
#   palette:    ordered ink list (hex) for optional gentle palette grade
#   treatment:  effect->params for postprocess.apply() (brand-appropriate, subtle)
#   text:       overlay colours (headline / sub / badge over a dark scrim)
BRAND_VISUALS = {
    "plenishd": {
        "scene": (
            "warm editorial food & kitchen photography, golden natural morning "
            "light, soft shallow depth of field, appetising, clean modern home, "
            "amber and warm-neutral tones"
        ),
        "palette": ["#2C2A28", "#FBBF24", "#F7EDE3", "#10B981", "#8A6D3B", "#1A1816"],
        "treatment": {
            "warm_grade": 0.5, "film_grain": 0.4, "vignette": 0.35,
            "paper_texture": 0.25, "contrast": 0.10,
        },
        "text": {"headline": "#FFFFFF", "sub": "#FBBF24", "badge": "#FBBF24", "scrim": "#2C2A28"},
        "badge": "Plenishd",
        "tone": "warm",
    },
    "coachos": {
        "scene": (
            "premium editorial sports photography, grassroots football, near-"
            "monochrome desaturated grade with a single restrained green accent, "
            "overcast natural light, clean grid composition, high craft"
        ),
        "palette": ["#111827", "#22C55E", "#E8E8E8", "#374151", "#0A0F1A"],
        "treatment": {
            "desaturate": 0.45, "film_grain": 0.45, "vignette": 0.4,
            "contrast": 0.18, "cool_grade": 0.25,
        },
        "text": {"headline": "#FFFFFF", "sub": "#22C55E", "badge": "#22C55E", "scrim": "#111827"},
        "badge": "CoachOS",
        "tone": "editorial",
    },
    "matchdaymaestro": {
        "scene": (
            "bold matchday-night football energy, dramatic stadium floodlights, "
            "high-contrast cinematic grade, deep blue night palette with electric "
            "crimson accents, punchy dynamic composition"
        ),
        "palette": ["#0F0F1A", "#E11D48", "#FBBF24", "#FFFFFF", "#1E1E3A"],
        "treatment": {
            "contrast": 0.28, "vignette": 0.5, "film_grain": 0.5,
            "chromatic": 0.5, "halftone": 0.18, "saturate": 0.15,
        },
        "text": {"headline": "#FFFFFF", "sub": "#FBBF24", "badge": "#E11D48", "scrim": "#0F0F1A"},
        "badge": "MatchdayMaestro",
        "tone": "punchy",
    },
    "sahil_twitter": {
        "scene": (
            "moody build-in-public developer aesthetic, dark editorial desk scene, "
            "warm key light against deep near-black, subtle screen glow, "
            "cinematic, minimal, high craft"
        ),
        "palette": ["#0A0A0A", "#C0C0C0", "#8B0000", "#E8E8E8", "#1A1A1A"],
        "treatment": {
            "contrast": 0.2, "vignette": 0.45, "film_grain": 0.5,
            "scanlines": 0.18, "cool_grade": 0.15,
        },
        "text": {"headline": "#FFFFFF", "sub": "#C0C0C0", "badge": "#8B0000", "scrim": "#0A0A0A"},
        "badge": "Sahil Saghir",
        "tone": "terminal",
    },
    "sahil_linkedin": {
        "scene": (
            "restrained professional editorial photography, clean modern workspace, "
            "soft neutral light, muted grey-tone palette with a single subtle accent, "
            "premium, uncluttered, business-appropriate"
        ),
        "palette": ["#0A0A0A", "#C0C0C0", "#8B0000", "#F2F2F2", "#2A2A2A"],
        "treatment": {
            "contrast": 0.10, "vignette": 0.25, "film_grain": 0.25, "cool_grade": 0.10,
        },
        "text": {"headline": "#FFFFFF", "sub": "#C0C0C0", "badge": "#C0C0C0", "scrim": "#0A0A0A"},
        "badge": "Sahil Saghir",
        "tone": "restrained",
    },
}

# Shared negative-prompt discipline (lifted in spirit from nous-branding's
# constraints): keep the base image textless and photographic so text never
# becomes gibberish -- the headline is added later as a crisp Pillow overlay.
NEGATIVE_PROMPT = (
    "text, letters, words, captions, typography, watermark, logo, signature, "
    "UI, user interface, buttons, charts, gibberish, distorted text, "
    "extra fingers, deformed, low quality, blurry, jpeg artifacts"
)

DEFAULT = BRAND_VISUALS["sahil_twitter"]


def for_brand(brand: str) -> dict:
    return BRAND_VISUALS.get((brand or "").lower(), DEFAULT)
