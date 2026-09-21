"""Load baoyu-style template blocks and universal construction rules for the
dimensional prompt system.

Source: github.com/JimLiu/baoyu-skills v1.57.0, vendored verbatim from the
Hermes skills install at ~/.hermes/skills/creative/baoyu-article-illustrator
(where PORT_NOTES.md attributes them as verbatim copies). See
baoyu_refs/SOURCE.md for full provenance.

The dimensional method: every prompt = Type (infographic|comparison|framework|
flowchart|timeline|scene|hero) x Style (23 options) x optional Palette (4
options). Styles and palettes are short markdown reference files that the
prompt engine embeds verbatim — never paraphrased — so the visual output
matches the canonical baoyu look.
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

_REFS = Path(__file__).resolve().parent / "baoyu_refs" / "skills" / "baoyu-article-illustrator" / "references"
_STYLES = _REFS / "styles"
_PALETTES = _REFS / "palettes"

# On-box skill: the source of truth at runtime. The on-box SKILL.md
# frontmatter declares MIT; the 23 style files and 4 palette files under
# `references/` are upstream copies re-read at runtime. Vendored refs
# remain the offline fallback when the on-box dir is absent.
_ON_BOX_SKILL = Path(os.path.expanduser(
    "~/.hermes/skills/creative/baoyu-article-illustrator"
))
_ON_BOX_REFS = (_ON_BOX_SKILL / "references") if _ON_BOX_SKILL.exists() else None

DEFAULT_STYLE = "notion"  # upstream default per references/styles.md
DEFAULT_PALETTE = "warm"  # only used when a brand explicitly asks for a palette

# Universal rules from baoyu references/prompt-construction.md — apply to ALL
# prompts. These are the rules the model needs to (a) keep the layout clean,
# (b) NOT render palette labels / hex codes as visible text, and (c) respect
# the requested aspect ratio.
UNIVERSAL_RULES = (
    "Clean composition with generous white space. Simple or no background. "
    "Main elements centered or positioned by content needs. "
    "Color values (#hex) and color names are rendering guidance only -- "
    "do NOT display color names, hex codes, or palette labels as visible text "
    "in the image. "
    "Human figures: simplified stylized silhouettes, not photorealistic. "
    "End respecting the requested aspect ratio."
)

# -- Canonical preset map (from references/style-presets.md) used to translate
#    a content_type into the upstream's recommended (type, style, palette)
#    combination. A brand can still override any dimension explicitly.
PRESETS = {
    "tech-explainer":        {"type": "infographic", "style": "blueprint"},
    "system-design":         {"type": "framework",   "style": "blueprint"},
    "dark-hud":              {"type": "infographic", "style": "dark-cyberpunk-hud"},
    "architecture":          {"type": "framework",   "style": "vector-illustration"},
    "science-paper":         {"type": "infographic", "style": "scientific"},
    "knowledge-base":        {"type": "infographic", "style": "vector-illustration"},
    "saas-guide":            {"type": "infographic", "style": "notion"},
    "tutorial":              {"type": "flowchart",   "style": "vector-illustration"},
    "process-flow":          {"type": "flowchart",   "style": "notion"},
    "warm-knowledge":        {"type": "infographic", "style": "vector-illustration", "palette": "warm"},
    "edu-visual":            {"type": "infographic", "style": "vector-illustration", "palette": "macaron"},
    "hand-drawn-edu":        {"type": "flowchart",   "style": "sketch-notes",         "palette": "macaron"},
    "ink-notes-compare":     {"type": "comparison",  "style": "ink-notes",            "palette": "mono-ink"},
    "ink-notes-flow":        {"type": "flowchart",   "style": "ink-notes",            "palette": "mono-ink"},
    "ink-notes-framework":   {"type": "framework",   "style": "ink-notes",            "palette": "mono-ink"},
    "data-report":           {"type": "infographic", "style": "editorial"},
    "versus":                {"type": "comparison",  "style": "vector-illustration"},
    "business-compare":      {"type": "comparison",  "style": "elegant"},
    "storytelling":          {"type": "scene",       "style": "warm"},
    "lifestyle":             {"type": "scene",       "style": "watercolor"},
    "history":               {"type": "timeline",    "style": "elegant"},
    "evolution":             {"type": "timeline",    "style": "warm"},
    "opinion-piece":         {"type": "scene",       "style": "screen-print"},
    "editorial-poster":      {"type": "comparison",  "style": "screen-print"},
    "cinematic":             {"type": "scene",       "style": "screen-print"},
    "hero":                  {"type": "hero",        "style": "warm"},
}


# -- Style names (24: 23 verbatim from upstream + 1 custom dark-cyberpunk-hud) -
STYLE_NAMES = [
    "blueprint", "chalkboard", "dark-cyberpunk-hud", "editorial", "elegant",
    "fantasy-animation", "flat-doodle", "flat", "ink-notes",
    "intuition-machine", "minimal", "nature", "notion", "pixel-art", "playful",
    "retro", "scientific", "screen-print", "sketch", "sketch-notes",
    "vector-illustration", "vintage", "warm", "watercolor",
]

# -- Palette names (4, verbatim from upstream) ---------------------------------
PALETTE_NAMES = ["macaron", "mono-ink", "neon", "warm"]

# -- Content types (7, from upstream prompt-construction) ---------------------
TYPE_NAMES = ["infographic", "comparison", "framework", "flowchart",
              "timeline", "scene", "hero"]


@lru_cache(maxsize=64)
def _read(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _on_box_path(name: str):
    """Path under the on-box skill refs (or None when absent)."""
    if not _ON_BOX_REFS:
        return None
    return _ON_BOX_REFS / name


def _vendored_path(name: str) -> Path:
    return _REFS / name


@lru_cache(maxsize=128)
def _read_first_available(name: str, kind: str = "ref") -> str:
    """Read a ref file: on-box skill first, vendored fallback.

    `kind` is "ref" (under references/) or "styles"/"palettes" (subdirs).
    Used by style_block / palette_block / prompt_construction_block.
    """
    if kind == "styles":
        on_box = _on_box_path(f"styles/{name}.md")
        vendored = _STYLES / f"{name}.md"
    elif kind == "palettes":
        on_box = _on_box_path(f"palettes/{name}.md")
        vendored = _PALETTES / f"{name}.md"
    else:
        on_box = _on_box_path(f"{name}.md") if _ON_BOX_REFS else None
        vendored = _REFS / f"{name}.md"
    if on_box is not None and on_box.exists():
        return on_box.read_text(encoding="utf-8").strip()
    if vendored.exists():
        return vendored.read_text(encoding="utf-8").strip()
    return ""


def universal_rules() -> str:
    return UNIVERSAL_RULES


def style_block(style: str) -> str:
    """Return the verbatim content of references/styles/<style>.md.

    Reads the on-box skill refs first (MIT, source of truth), falling back
    to the vendored refs when the on-box dir is absent. Falls back to the
    default style (notion) if the requested style is not found. Returns
    an empty string only if even the default is missing.
    """
    txt = _read_first_available(style, "styles")
    if not txt and style != DEFAULT_STYLE:
        txt = _read_first_available(DEFAULT_STYLE, "styles")
    return txt


def palette_block(palette: str) -> str:
    """Return the verbatim content of references/palettes/<palette>.md.

    On-box first, vendored fallback. Returns empty string if not found
    anywhere. Palettes are optional in the dimensional method.
    """
    return _read_first_available(palette, "palettes")


def prompt_construction_block() -> str:
    """Return the verbatim content of references/prompt-construction.md.

    On-box first, vendored fallback. Used by the article illustrator to
    build prompt-file bodies per the upstream construction rules.
    """
    return _read_first_available("prompt-construction", "ref")


def style_presets_block() -> str:
    """Return the verbatim content of references/style-presets.md.

    On-box first, vendored fallback. Lets callers parse the upstream
    preset table at runtime when needed.
    """
    return _read_first_available("style-presets", "ref")


def preset(name: str) -> dict:
    """Canonical (type, style, palette) for a named preset.

    Returns a copy so callers can mutate freely. Empty dict when unknown.
    """
    return PRESETS.get(name, {}).copy()


def list_presets() -> list[str]:
    """All canonical preset names from the in-code PRESETS table."""
    return list(PRESETS.keys())


def preset_for(name: str) -> dict:
    """Backwards-compatible alias for `preset(name)`."""
    return preset(name)
