"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _default_model_for_provider(provider: str) -> str:
    """Best-effort default model for a bare provider name.

    Lazy import keeps this module free of the heavier provider catalog
    machinery in ``hermes_cli.models`` (and sidesteps any import cycle).
    Returns an empty string when the provider is unknown.
    """
    try:
        from hermes_cli.models import get_default_model_for_provider

        return (get_default_model_for_provider(provider) or "").strip()
    except Exception:
        return ""


def _coerce_entry(entry: Any) -> dict[str, Any] | None:
    """Normalize one fallback entry to a ``{provider, model, ...}`` dict.

    Accepts either a full dict or a bare provider string (e.g. ``"deepseek"``),
    which is expanded to that provider's default model. A dict carrying a
    provider but no ``model`` is likewise filled with the provider default.
    Returns ``None`` when the entry can't be resolved to a (provider, model)
    pair so callers can skip it instead of silently dropping a bare string.
    """
    if isinstance(entry, str):
        provider = entry.strip()
        if not provider:
            return None
        model = _default_model_for_provider(provider)
        if not model:
            return None
        return {"provider": provider, "model": model}

    if not isinstance(entry, dict):
        return None

    provider = str(entry.get("provider") or "").strip()
    if not provider:
        return None
    model = str(entry.get("model") or "").strip()
    if not model:
        model = _default_model_for_provider(provider)
        if not model:
            return None

    normalized = dict(entry)
    normalized["provider"] = provider
    normalized["model"] = model

    base_url = _normalized_base_url(entry.get("base_url"))
    if base_url:
        normalized["base_url"] = base_url

    return normalized


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, (dict, str)):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        coerced = _coerce_entry(entry)
        if coerced is not None:
            entries.append(coerced)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.
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

    return chain
