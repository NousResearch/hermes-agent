"""OpenRouter zero-priced model picker — a FILTER over the live catalog, not a provider.

Hermes' curated OpenRouter list (``models.fetch_openrouter_models``) is a hand-ranked
snapshot, so a newly-launched free model stays invisible until a release promotes it.
This module answers a different question: "what can I run right now for free?", straight
off ``/v1/models``, reusing the catalog fetch, the free-price predicate and the
tool-capability predicate the curated path already uses.

It is a *picker action*, not a provider slug: the row enters ``hermes model`` through the
trailing-action list in ``main_provider_setup._build_provider_picker_rows`` and dispatches in
``main.select_provider_and_model`` ahead of ``_PROVIDER_MODEL_FLOWS``. That is deliberate —
minting an ``openrouter-free`` provider would drag it through ``CANONICAL_PROVIDERS``,
``resolve_provider_full`` and every alias table for what is really a model list under the
existing ``openrouter`` provider.
"""

from __future__ import annotations

import time
from typing import Optional

# Filtered for an hour: a picker open that re-downloads the ~686KB catalog is a visible stall.
_FREE_CACHE_TTL = 3600.0

# Set to an empty list by a failed fetch so a second open inside the TTL doesn't re-hit the
# network; a successful fetch replaces it.
_free_cache: Optional[list[str]] = None
_free_cache_time: float = 0.0


def _free_models_from_live_catalog(timeout: float) -> list[str]:
    """Model ids that are BOTH zero-priced AND tool-capable in the live catalog.

    Tool-capability is not optional: Hermes is tool-calling-first, so a free model that
    can't take a ``tools`` parameter fails at the first tool call — far worse than being
    absent from a list.
    """
    from hermes_cli.models import (
        _OPENROUTER_CATALOG_URL,
        _fetch_live_catalog_index,
        _openrouter_model_is_free,
        _openrouter_model_supports_tools,
        _urlopen_model_catalog_request,
    )

    live = _fetch_live_catalog_index(_OPENROUTER_CATALOG_URL, timeout, _urlopen_model_catalog_request)
    if live is None:
        return []
    items, _by_id = live
    return sorted(
        str(item.get("id") or "").strip()
        for item in items
        if isinstance(item, dict)
        and _openrouter_model_supports_tools(item)
        and _openrouter_model_is_free(item.get("pricing"))
        and str(item.get("id") or "").strip()
    )


def openrouter_free_model_ids(*, force_refresh: bool = False, timeout: float = 8.0) -> list[str]:
    """Zero-priced, tool-capable OpenRouter model ids, cached an hour per process."""
    global _free_cache, _free_cache_time

    now = time.monotonic()
    if not force_refresh and _free_cache is not None and (now - _free_cache_time) < _FREE_CACHE_TTL:
        return list(_free_cache)

    found = _free_models_from_live_catalog(timeout)
    if not found and _free_cache is not None:
        # Network blip: keep serving the last good list rather than emptying the picker.
        return list(_free_cache)

    _free_cache = found
    _free_cache_time = now
    return list(found)


def _model_flow_openrouter_free(config, current_model=""):
    """Ensure the OpenRouter key, then pick from the live free-model list."""
    from hermes_constants import OPENROUTER_BASE_URL
    from hermes_cli.auth import ProviderConfig, _prompt_model_selection
    from hermes_cli.model_setup_flows_common import _ensure_flow_api_key, _finish_model

    # Same synthesized pconfig the curated openrouter flow uses — OpenRouter isn't in
    # PROVIDER_REGISTRY. The key is shared; this only narrows the MODEL list.
    pconfig = ProviderConfig(
        id="openrouter", name="OpenRouter", auth_type="api_key", api_key_env_vars=("OPENROUTER_API_KEY",)
    )
    existing_key, resolved, abort = _ensure_flow_api_key(
        "openrouter", pconfig, missing_hint=("Get one at: https://openrouter.ai/keys", "")
    )
    if abort:
        return

    models = openrouter_free_model_ids(force_refresh=True)
    if not models:
        print("  No free OpenRouter models returned right now (catalog unreachable or none published).")
        return

    selected = _prompt_model_selection(
        models,
        current_model=current_model,
        confirm_provider="openrouter",
        confirm_base_url=OPENROUTER_BASE_URL,
        confirm_api_key=resolved or existing_key,
    )
    _finish_model(
        selected,
        "openrouter",
        f"Default model set to: {selected} (free OpenRouter model)",
        base_url=OPENROUTER_BASE_URL,
        api_mode="chat_completions",
    )
