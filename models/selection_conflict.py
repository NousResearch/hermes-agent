"""Network-free model/provider coherence gate over canonical static catalogues."""
from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

from models.catalog_static import _BORROWED_MODEL_PROVIDERS, _PROVIDER_MODELS, static_provider_model_ids
from models.identity import normalize_model_id
from providers import get_provider_label, is_aggregator, normalize_provider


_STATIC_FAMILY_PREFIXES = {
    "openai-codex": ("gpt-", "codex-", "o1", "o3", "o4"),
    "xai-oauth": ("grok-",),
}


def _family_head(model: str) -> str:
    return re.split(r"[-./:]", model.strip().lower(), maxsplit=1)[0]


def _catalog_owns(provider: str, wanted: str) -> bool:
    catalog = static_provider_model_ids(provider)
    if any(model.lower() == wanted for model in catalog):
        return True
    normalized = normalize_model_id(provider, wanted, known_ids=catalog)
    return any(model.lower() == normalized.lower() for model in catalog)


def static_model_provider_conflict(
    model_name: str, provider: str | None, *, limit: int = 5,
) -> dict[str, Any] | None:
    """Reject only definite native-vendor conflicts; hidden/unknown IDs remain permissive.

    OAuth families also reject a foreign-looking ID even when no other vendor
    currently lists it. Custom endpoints and aggregators cannot be judged offline.
    """
    requested = str(model_name or "").strip()
    canonical = normalize_provider(provider)
    catalog = list(static_provider_model_ids(canonical))
    if not requested or not catalog or canonical == "moa" or is_aggregator(canonical):
        return None
    wanted = requested.lower()
    if _catalog_owns(canonical, wanted):
        return None
    if _family_head(requested) in {_family_head(model) for model in catalog}:
        return None
    if canonical not in _STATIC_FAMILY_PREFIXES:
        foreign = any(
            _catalog_owns(other, wanted)
            for other in _PROVIDER_MODELS
            if other != canonical and not is_aggregator(other)
            and other not in _BORROWED_MODEL_PROVIDERS
        )
        if not foreign:
            foreign = any(
                _catalog_owns(other, wanted)
                for other in _BORROWED_MODEL_PROVIDERS if other != canonical
            )
        if not foreign:
            return None
    suggestions = get_close_matches(requested, catalog, n=limit, cutoff=0.4) or catalog[:limit]
    label = get_provider_label(canonical)
    return {
        "model": requested,
        "provider": canonical,
        "suggestions": suggestions,
        "message": (
            f"Model `{requested}` is not served by provider `{canonical}` ({label}). "
            f"Closest {label} models: " + ", ".join(f"`{item}`" for item in suggestions) + "."
        ),
    }
