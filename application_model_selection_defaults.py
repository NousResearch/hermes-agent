"""Application-side fact acquisition for canonical default-model selection."""

from __future__ import annotations

import application_model_pricing

from collections.abc import Iterable

from models.selection import ModelSelection, select_default_model, select_nous_default_model


def preferred_silent_default_model(provider: str = "openrouter") -> str:
    """Cache-only labelled default, else the curated global silent fallback."""

    from hermes_cli.catalog_context import catalog_cache_path
    from models.catalog_runtime import cached_default_model
    from models.catalog_static import PREFERRED_SILENT_DEFAULT_MODEL

    try:
        labeled = cached_default_model(catalog_cache_path(), provider)
    except Exception:
        labeled = None
    return str(labeled or PREFERRED_SILENT_DEFAULT_MODEL)


def select_silent_default(
    provider: str,
    model_ids: Iterable[str],
    *,
    deprioritized_model_ids: Iterable[str] = (),
) -> ModelSelection:
    return select_default_model(
        provider,
        model_ids,
        preferred_model=preferred_silent_default_model(provider),
        deprioritized_model_ids=deprioritized_model_ids,
        purpose="silent_default",
    )


def select_provider_default(provider: str) -> ModelSelection:
    """Default for a configured provider with no explicit model selection."""

    from models.catalog_static import (
        _PROVIDER_MODELS,
        _SILENT_DEFAULT_PROVIDERS,
    )

    provider_id = str(provider or "").strip().lower()
    model_ids = tuple(_PROVIDER_MODELS.get(provider_id, ()))
    silent = provider_id in _SILENT_DEFAULT_PROVIDERS
    return select_default_model(
        provider_id,
        model_ids,
        preferred_model=preferred_silent_default_model(provider_id) if silent else "",
        allow_preferred_without_models=silent,
        purpose="provider_default",
    )


def _recommended_ids(payload: object, key: str) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    block = payload.get(key)
    if not isinstance(block, list):
        return ()
    out: list[str] = []
    for entry in block:
        name = entry.get("modelName") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return tuple(out)


def select_nous_recommended_default() -> tuple[ModelSelection, bool]:
    """Gather Nous account/catalogue facts, then delegate all default policy."""

    from hermes_cli import models as catalog
    from application_nous_recommendations import fetch_recommended_models

    curated = catalog.get_curated_nous_model_ids()
    pricing = application_model_pricing.get_pricing_for_provider("nous") or {}
    free_tier = bool(catalog.check_nous_free_tier(force_fresh=True))
    try:
        payload = fetch_recommended_models()
    except Exception:
        payload = None
    portal_key = "freeRecommendedModels" if free_tier else "paidRecommendedModels"
    recommended = _recommended_ids(payload, portal_key)

    selection = select_nous_default_model(
        curated,
        portal_recommended_model_ids=recommended,
        pricing=pricing,
        policy_allowed_ids=application_model_pricing.nous_policy_allowed_ids(),
        free_tier=free_tier,
        preferred_model=preferred_silent_default_model("nous"),
    )
    return selection, free_tier


def selected_model_id(selection: ModelSelection) -> str:
    return selection.selected.ref.model if selection.selected is not None else ""


__all__ = [
    "preferred_silent_default_model",
    "select_nous_recommended_default",
    "select_provider_default",
    "select_silent_default",
    "selected_model_id",
]
