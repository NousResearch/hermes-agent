"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

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


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. If the primary ``model.default`` names a configured ``model_aliases``
    entry, that alias' ``fallback_chain`` is appended next. Legacy
    ``fallback_model`` entries are appended afterwards unless they target the
    same provider/model/base_url route as an earlier entry. The returned list
    always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    alias_fallback_chain: Any = None
    try:
        from hermes_cli.model_aliases import get_model_alias_entry

        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            alias_name = model_cfg.get("default") or model_cfg.get("model")
        else:
            alias_name = model_cfg
        alias = get_model_alias_entry(config, alias_name)
        if alias is not None:
            alias_fallback_chain = alias.get("fallback_chain")
    except Exception:
        alias_fallback_chain = None

    sources = (
        config.get("fallback_providers"),
        alias_fallback_chain,
        config.get("fallback_model"),
    )
    for raw_entries in sources:
        for entry in _iter_fallback_entries(raw_entries):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
