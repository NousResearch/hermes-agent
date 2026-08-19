"""Extended style menu for the canonical image pipeline.

This package provides the ~9,000-style trait registry (plus deterministic
blending) as an *extension* of the existing style/design/skill capabilities.
It is a read-only consumer of the exported registry artifact; it never
modifies the source, never emits protected source names (artists,
photographers, films, titles) and never emits SREF codes.  Generation stays
on the existing canonical path (native Codex) — this layer only supplies
neutral trait language to enrich the prompt.

Design rules (locked by Sahil):
- No individual Hermes skills; registry + blend index only.
- No protected names / no SREF in any emitted fragment or registry surface.
- Backend authority unchanged (Codex primary; local explicit toggle only).
- Fail fast on unknown styles / blends — never a silent fallback.
"""

from __future__ import annotations

from pathlib import Path

_REGISTRY_FILE = Path(__file__).resolve().parent / "registry-full.json"


class RegistryError(ValueError):
    """Raised when a style/blend cannot be resolved from the registry."""


def _load_registry() -> dict:
    import json

    with _REGISTRY_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


_REGISTRY: dict | None = None


def get_registry() -> dict:
    """Return the loaded registry (cached at import-time on first call)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_registry()
    return _REGISTRY


def resolve_style(slug: str) -> dict:
    """Resolve a single style slug into its neutral registry entry.

    Raises RegistryError for an unknown slug.  The returned entry carries
    neutral ``style_label`` and ``prompt_fragment`` only — never the
    protected source ``name`` and never SREF.
    """
    registry = get_registry()
    styles = registry.get("styles") or {}
    entry = styles.get(slug)
    if entry is None:
        raise RegistryError(f"unknown registry style: {slug!r}")
    return entry


def search(tags: list[str], limit: int = 10) -> list[dict]:
    """Search the registry by trait tags (case-insensitive substring)."""
    from .registry import search as _search

    return _search(get_registry(), tags)[:limit]


def blend(style_slugs: list[str], seed: int | None = None) -> dict:
    """Deterministically blend two or more styles into a neutral fragment.

    Returns a dict with ``prompt_fragment`` (neutral trait language),
    ``tags``, ``medium`` and ``resolution_notes``.  Raises RegistryError if
    any slug is unknown or fewer than two are supplied.
    """
    # Import the submodule under a non-colliding name: ``from .blend import
    # ...`` would set the package attribute ``style_registry.blend`` to the
    # submodule, clobbering the module-level ``blend`` function below on the
    # first call (import machinery overwrites the attribute unconditionally).
    from . import blend_engine as _blend_module  # type: ignore[import-not-found]

    if not style_slugs or len(style_slugs) < 2:
        raise RegistryError("blend requires at least two style slugs")
    resolved = [resolve_style(slug) for slug in style_slugs]
    result = _blend_module.blend(resolved, seed=seed if seed is not None else 42)
    # Normalise to a dict contract (blend returns a dict with
    # prompt_fragment / tags / medium / resolution_notes).
    return result


# ---------------------------------------------------------------------------
# Neutral-trait enforcement (the seam's guarantee)
# ---------------------------------------------------------------------------

# Tokens that must never appear in any emitted prompt fragment.
_FORBIDDEN_PATTERNS = (
    "--sref",
    "sref",
)
_PROTECTED_NAME_HINTS = (
    # identity-bearing proper names that have leaked historically; the
    # registry surface is already neutral, this is defence in depth.
    "stanley kubrick",
    "pablo picasso",
    "käthe kollwitz",
    "claude monet",
    "ansel adams",
    "alvin lustig",
    "carole feuerman",
    "peter mohrbacher",
    "rineke dijkstra",
    "zao wou-ki",
    "2001: a space odyssey",
)


def assert_neutral(fragment: str) -> str:
    """Assert a fragment contains no protected name and no SREF token.

    Returns the fragment unchanged on success; raises RegistryError on any
    forbidden token.  This is the enforcement point for the seam guarantee:
    the final prompt must never carry protected identity or SREF into the
    generator.
    """
    lowered = fragment.lower()
    for token in _FORBIDDEN_PATTERNS:
        if token in lowered:
            raise RegistryError(f"non-neutral token in fragment: {token!r}")
    for name in _PROTECTED_NAME_HINTS:
        if name in lowered:
            raise RegistryError(f"protected name leaked into fragment: {name!r}")
    return fragment


def resolve_fragment(style_slug: str | None, blend_slugs: list[str] | None, seed: int | None = None) -> str | None:
    """Resolve the neutral fragment for a style and/or blend.

    Returns ``None`` when neither is supplied.  Single style → its neutral
    ``prompt_fragment``; blend → deterministic blended fragment.  Both are
    passed through :func:`assert_neutral`.
    """
    if style_slug:
        return assert_neutral(resolve_style(style_slug)["prompt_fragment"])
    if blend_slugs:
        return assert_neutral(blend(blend_slugs, seed=seed)["prompt_fragment"])
    return None


__all__ = [
    "RegistryError",
    "get_registry",
    "resolve_style",
    "search",
    "blend",
    "assert_neutral",
    "resolve_fragment",
]


# ── Deterministic variation selector ────────────────────────────────────────
# Extended style menu for existing callers (blog / X / LinkedIn / workflows).
# The selector keeps a post visually coherent (one style + optional blend per
# job) while varying deliberately across posts. Selection is seeded and
# deterministic: the same (context, seed) always picks the same style, so
# tests and re-runs are reproducible.

_KNOWN_EXAMPLE_SLUGS = (
    "steampunk", "synthwave", "linocut-print", "risograph", "editorial",
    "high-contrast", "technical-diagram", "monochrome-documentary",
    "surreal-painterly", "minimalist-geometric", "fresco", "woodblock-print",
    "pompeian-fresco", "birds-eye-view", "wireframe", "retro-futurist",
    "dark-cyberpunk-hud", "saga-noir", "chromatic-institute", "ink-ember-studio",
)

def pick_variation(seed: int, *, registry=None) -> dict:
    """Deterministically pick a style slug (+ optional blend pair) from the
    extended menu for a given seed. Never returns protected names or SREF."""
    if registry is None:
        registry = get_registry()
    styles = registry.get("styles") or {}
    available = [s for s in _KNOWN_EXAMPLE_SLUGS if s in styles]
    if not available:
        available = sorted(styles.keys())
    import hashlib
    digest = hashlib.sha256(f"{seed}:midlib-var".encode()).hexdigest()
    idx = int(digest[:8], 16) % len(available)
    slug = available[idx]
    # Deterministic companion blend (50% of picks, always different slug).
    digest2 = hashlib.sha256(f"{seed}:midlib-blend".encode()).hexdigest()
    if int(digest2[:8], 16) % 2 == 0 and len(available) > 1:
        idx2 = (idx + 1 + (int(digest2[8:16], 16) % (len(available) - 1))) % len(available)
        blend_slug = available[idx2]
        if blend_slug != slug:
            return {"style_slug": slug, "blend_slugs": [slug, blend_slug]}
    return {"style_slug": slug, "blend_slugs": None}
