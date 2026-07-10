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


def _configured_tier(config: dict[str, Any], explicit_tier: str | None = None) -> str:
    if explicit_tier:
        return explicit_tier.strip()

    model_cfg = config.get("model") or {}
    if isinstance(model_cfg, dict):
        tier = str(model_cfg.get("tier") or "").strip()
        if tier:
            return tier

    agent_cfg = config.get("agent") or {}
    if isinstance(agent_cfg, dict):
        tier = str(agent_cfg.get("tier") or "").strip()
        if tier:
            return tier

    return ""


def _routing_entry(config: dict[str, Any], task_context: str | None) -> dict[str, Any]:
    if not task_context:
        return {}
    routing = config.get("routing") or {}
    if not isinstance(routing, dict):
        return {}
    entry = routing.get(task_context) or {}
    return entry if isinstance(entry, dict) else {}


def get_fallback_chain(
    config: dict[str, Any] | None,
    *,
    tier: str | None = None,
    task_context: str | None = None,
) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    When ``model.tier`` / ``agent.tier`` or an explicit ``tier`` is configured,
    ``fallback_tiers.<tier>`` replaces the global chain. ``routing.<context>``
    may also provide ``fallback_providers`` or ``tier`` for task-specific
    routing.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    route = _routing_entry(config, task_context)
    route_chain = _iter_fallback_entries(
        route.get("fallback_providers") or route.get("fallback_model")
    )
    if route_chain:
        return route_chain

    selected_tier = _configured_tier(
        config,
        explicit_tier=(tier or str(route.get("tier") or route.get("fallback_tier") or "")),
    )
    fallback_tiers = config.get("fallback_tiers") or {}
    if selected_tier and isinstance(fallback_tiers, dict):
        tier_chain = _iter_fallback_entries(fallback_tiers.get(selected_tier))
        if tier_chain:
            return tier_chain

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
