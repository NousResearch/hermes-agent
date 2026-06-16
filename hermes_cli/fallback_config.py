"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import os
from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def _fallback_promotions_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("fallback_promotions")
    if not isinstance(raw, dict):
        return {}
    return raw


def _fallback_promotions_enabled(config: dict[str, Any]) -> bool:
    """Return whether dynamic free-promotion fallbacks should be merged.

    The feature is opt-in through DEFAULT_CONFIG's ``fallback_promotions`` key.
    Tests and callers that pass a small ad-hoc config without that key keep the
    old static-only behavior.
    """
    env_override = os.getenv("HERMES_FALLBACK_PROMOTIONS", "").strip().lower()
    if env_override in {"0", "false", "no", "off"}:
        return False
    if env_override in {"1", "true", "yes", "on"}:
        return True

    promotions = _fallback_promotions_config(config)
    return promotions.get("enabled") is True


def _provider_promotions_enabled(config: dict[str, Any], provider: str) -> bool:
    promotions = _fallback_promotions_config(config)
    providers = promotions.get("providers", ["nous"])
    if isinstance(providers, str) and providers.strip().lower() in {"*", "all"}:
        return True
    if not isinstance(providers, list):
        return False
    wanted = {str(p).strip().lower() for p in providers if str(p).strip()}
    return provider.strip().lower() in wanted


def _promotion_position(config: dict[str, Any]) -> str:
    promotions = _fallback_promotions_config(config)
    position = str(promotions.get("position") or "prepend").strip().lower()
    if position not in {"prepend", "append"}:
        return "prepend"
    return position


def _provider_has_auth(provider: str) -> bool:
    """Cheap auth gate so startup does not hit remote promo endpoints uselessly."""
    provider = provider.strip().lower()
    if provider == "nous":
        if os.getenv("NOUS_API_KEY", "").strip():
            return True
        try:
            from hermes_cli.auth import get_provider_auth_state

            return bool(get_provider_auth_state("nous"))
        except Exception:
            return False
    return False


def _free_nous_promotion_entries() -> list[dict[str, Any]]:
    """Return free Nous Portal promo models currently advertised by Portal."""
    try:
        from hermes_cli.models import (
            _resolve_nous_portal_url,
            fetch_nous_recommended_models,
        )
    except Exception:
        return []

    try:
        payload = fetch_nous_recommended_models(
            _resolve_nous_portal_url(),
            timeout=1.5,
        )
    except Exception:
        return []

    free_models = payload.get("freeRecommendedModels") if isinstance(payload, dict) else None
    if not isinstance(free_models, list):
        return []

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in free_models:
        if not isinstance(item, dict):
            continue
        model = str(item.get("modelName") or "").strip()
        if not model or model.lower() in seen:
            continue
        seen.add(model.lower())
        entries.append(
            {
                "provider": "nous",
                "model": model,
                "supports_tools": True,
                "source": "dynamic-free-promotion",
            }
        )
    return entries


def _dynamic_free_promotion_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not _fallback_promotions_enabled(config):
        return []

    entries: list[dict[str, Any]] = []
    if _provider_promotions_enabled(config, "nous") and _provider_has_auth("nous"):
        entries.extend(_free_nous_promotion_entries())
    return entries


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    relative order. Legacy ``fallback_model`` entries are appended afterwards
    unless they target the same provider/model/base_url route as an earlier
    entry. When ``fallback_promotions.enabled`` is true, currently-free
    provider promotions are merged in memory so short-lived free models do not
    depend on a static YAML edit. Explicitly configured entries win on
    duplicates. The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    promotion_entries: list[dict[str, Any]] = []
    for entry in _dynamic_free_promotion_entries(config):
        identity = _entry_identity(entry)
        if identity in seen:
            continue
        seen.add(identity)
        promotion_entries.append(entry)

    if promotion_entries and _promotion_position(config) == "prepend":
        return promotion_entries + chain
    if promotion_entries:
        return chain + promotion_entries
    return chain
