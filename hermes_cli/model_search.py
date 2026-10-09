"""Picker-only search aliases for model ids."""

from __future__ import annotations
from models.catalog_projection import _MODEL_SEARCH_ALIASES

# Lowercased wire id → extra tokens appended to the search haystack only.

# Lowercased wire id → canonical public slug (the FIRST alias by convention), so picker dedup doesn't
# render a live bare id and its curated slug (``k3`` / ``kimi-k3``) as two rows.




def model_search_text(model: str) -> str:
    """Haystack for fuzzy/substring model search; never changes the wire id sent to the provider."""
    mid = (model or "").strip()
    if not mid:
        return model or ""
    aliases = _MODEL_SEARCH_ALIASES.get(mid.lower())
    return f"{mid} {' '.join(aliases)}" if aliases else mid
