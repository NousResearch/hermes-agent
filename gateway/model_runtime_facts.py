"""Gateway projection of lower-domain model facts for turn construction."""

from __future__ import annotations

from models import normalize_model_id
from models.catalog_static import (
    static_provider_default_preference,
    static_provider_model_ids,
)
from models.selection import select_default_model
from providers import is_aggregator, normalize_provider


def provider_default_model(provider: str) -> str:
    """Select a provider default through the canonical model-selection domain."""
    provider_id = normalize_provider(provider)
    preferred = static_provider_default_preference(provider_id)
    if preferred:
        try:
            from gateway.model_catalog_runtime import cached_default_model

            preferred = str(cached_default_model(provider_id) or preferred)
        except Exception:
            pass
    selection = select_default_model(
        provider_id,
        static_provider_model_ids(provider_id),
        preferred_model=preferred,
        allow_preferred_without_models=bool(preferred),
        purpose="provider_default",
    )
    return selection.selected.ref.model if selection.selected is not None else ""


def normalize_runtime_model(provider: str, model: str) -> str:
    """Match agent model identity to the canonical provider-specific wire ID."""
    provider_id = normalize_provider(provider)
    if not provider_id or is_aggregator(provider_id):
        return str(model or "").strip()
    return normalize_model_id(
        provider_id,
        model,
        known_ids=static_provider_model_ids(provider_id),
    )


__all__ = ["normalize_runtime_model", "provider_default_model"]
