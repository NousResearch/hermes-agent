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
        from hermes_cli.fallback_chain import is_dynamic_fallback_entry

        if not ((provider and model) or is_dynamic_fallback_entry(entry)):
            continue

        normalized = dict(entry)
        if provider:
            normalized["provider"] = provider
        if model:
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


def _primary_route_from_config(config: dict[str, Any]) -> tuple[str, str]:
    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        return "", ""
    from hermes_cli.models import normalize_provider

    provider = normalize_provider(model_cfg.get("provider") or "")
    model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    return provider, model


def _enrich_nvidia_rotation(
    raw_chain: list[dict[str, Any]],
    *,
    primary_provider: str,
    primary_model: str,
) -> list[dict[str, Any]]:
    """Ensure NVIDIA primaries rotate through other NIM models before leaving NVIDIA."""
    from hermes_cli import models as model_catalog
    from hermes_cli.fallback_chain import is_nvidia_auto_fallback_entry

    if primary_provider != "nvidia":
        return raw_chain

    nvidia_entries = [
        entry
        for entry in raw_chain
        if model_catalog.normalize_provider(entry.get("provider")) == "nvidia"
    ]
    if not nvidia_entries:
        return [{"provider": "nvidia", "model": "auto"}, *raw_chain]

    if any(is_nvidia_auto_fallback_entry(entry) for entry in nvidia_entries):
        return raw_chain

    primary_norm = primary_model.strip().lower()
    alternates = [
        str(entry.get("model") or "").strip().lower()
        for entry in nvidia_entries
        if str(entry.get("model") or "").strip().lower() not in {"", primary_norm}
    ]
    if alternates:
        return raw_chain

    trimmed = [
        entry
        for entry in raw_chain
        if not (
            model_catalog.normalize_provider(entry.get("provider")) == "nvidia"
            and str(entry.get("model") or "").strip().lower() == primary_norm
        )
    ]
    return [{"provider": "nvidia", "model": "auto"}, *trimmed]


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


def resolve_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the runtime fallback chain with NVIDIA/Nous dynamic entries expanded."""
    config = config or {}
    raw_chain = get_fallback_chain(config)
    primary_provider, primary_model = _primary_route_from_config(config)

    if not raw_chain and primary_provider == "nvidia":
        raw_chain = [
            {"provider": "nvidia", "model": "auto"},
            {"provider": "nous", "model": "auto-free"},
        ]
    else:
        raw_chain = _enrich_nvidia_rotation(
            raw_chain,
            primary_provider=primary_provider,
            primary_model=primary_model,
        )

    from hermes_cli.fallback_chain import normalize_fallback_entries

    return normalize_fallback_entries(raw_chain, exclude_model=primary_model or None)
