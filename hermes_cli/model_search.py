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

# Lowercased wire id → canonical public slug it aliases. Used by picker
# dedup so a live bare id and its curated public slug (``k3`` / ``kimi-k3``)
# don't render as two rows for the same model. Derived from the FIRST alias
# entry, which by convention is the full public slug.
_MODEL_ALIAS_CANONICAL: dict[str, str] = {
    wire_id: aliases[0].lower()
    for wire_id, aliases in _MODEL_SEARCH_ALIASES.items()
    if aliases
}


def model_alias_canonical(model: str) -> str:
    """Return the canonical public slug for a bare wire-id alias.

    Identity for ids with no alias entry. Lowercases the input so callers
    can use the result directly as a dedup key.
    """
    key = (model or "").strip().lower()
    return _MODEL_ALIAS_CANONICAL.get(key, key)


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
