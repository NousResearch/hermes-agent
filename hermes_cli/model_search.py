"""Picker-only search aliases for model ids.

Wire IDs stay unchanged. Some providers report short or brand-less ids
(Kimi Coding's flagship is literally ``k3``) that users still search for by
the familiar ``kimi-…`` naming of sibling models.

Keep in sync with ``ui-tui/src/lib/model-search-text.ts`` and
``web/src/lib/model-search-text.ts``.
"""

from __future__ import annotations

# Lowercased wire id → extra tokens appended to the search haystack only.
_MODEL_SEARCH_ALIASES: dict[str, tuple[str, ...]] = {
    "k3": ("kimi-k3", "kimi"),
}


def model_search_text(model: str) -> str:
    """Return the haystack used for fuzzy/substring model search.

    Never changes the wire id passed to the provider.
    """
    mid = (model or "").strip()
    if not mid:
        return model or ""
    aliases = _MODEL_SEARCH_ALIASES.get(mid.lower())
    if not aliases:
        return mid
    return f"{mid} {' '.join(aliases)}"
