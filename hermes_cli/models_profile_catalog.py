"""Provider-owned catalog probing and live/curated selection for setup and pickers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def probe_profile_catalog(normalized: str, profile, api_key: str | None, base_url: str | None) -> list[str] | None:
    """Probe with the configured credentials, then apply the profile's catalog policy."""
    live = None
    if api_key:
        try:
            live = profile.fetch_models(api_key=api_key, base_url=base_url)
        except Exception:
            logger.debug("Catalog fetch failed for %s", normalized, exc_info=True)
    return merge_profile_catalog(normalized, profile, live)


def merge_profile_catalog(normalized: str, profile, live: list[str] | None) -> list[str] | None:
    """Authoritative profiles use live IDs exclusively; other profiles retain curated merging."""
    from hermes_cli import models

    if profile.authoritative_model_catalog:
        if live is not None:
            return live
        from agent.models_dev import list_agentic_models
        cached = list_agentic_models(normalized, allow_network=False)
        rows = cached or list(profile.fallback_models)
        return models.CuratedFallbackModels(rows) if rows else None
    if not live:
        rows = models.CuratedFallbackModels(profile.fallback_models) if profile.fallback_models else None
    else:
        curated = list(models._PROVIDER_MODELS.get(normalized, [])) or list(profile.fallback_models or ())
        if not curated:
            rows = live
        else:
            primary, secondary = (live, curated) if normalized in models._LIVE_FIRST_PICKER_PROVIDERS else (curated, live)
            rows = models._merge_unique(primary, secondary, key=models._model_dedup_key)
    return models._drop_delisted_opencode_models(normalized, rows)
