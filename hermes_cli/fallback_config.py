"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any


def _normalized_base_url(value: Any) -> str:
    return value.strip().rstrip("/") if isinstance(value, str) else ""


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``api_key_env`` accepted as alias); None when neither
    yields a value so ``resolve_runtime_provider`` falls through to standard credential resolution.
    ``key_env`` goes through ``agent.secret_scope.get_secret``, not raw ``os.getenv``: in a
    multiplexed gateway a bare env read ignores the active profile's scope and can return another
    profile's credential.
    """
    if not isinstance(entry, dict):
        return None
    if inline := str(entry.get("api_key") or "").strip():
        return inline
    if key_env := str(entry.get("key_env") or entry.get("api_key_env") or "").strip():
        from agent.secret_scope import get_secret
        return (get_secret(key_env) or "").strip() or None
    return None


def effective_runtime_provider(
    entry: dict[str, Any] | None, runtime: dict[str, Any] | None
) -> str:
    """Provider identity to persist/display for a resolved fallback entry.

    ``resolve_runtime_provider`` returns the bare billing class ``"custom"``
    for every named ``providers:`` / ``custom_providers:`` entry; the entry's
    configured id only survives in ``requested_provider``. Fallback resolvers
    that persist ``runtime["provider"]`` as the agent identity therefore label
    sessions/billing rows ``custom`` instead of the configured provider name —
    while the manual ``/model`` switch path correctly persists the named id
    (#98739). Same class as the delegation fix in ``tools/delegate_tool.py``.

    Returns the entry's requested identity when the resolved provider is the
    bare ``custom`` class; a genuinely ad-hoc endpoint (requested provider IS
    ``custom``) keeps the bare class unchanged.
    """
    runtime = runtime or {}
    resolved = str(runtime.get("provider") or "").strip()
    if resolved.lower() != "custom":
        return resolved
    requested = str(
        runtime.get("requested_provider")
        or (entry or {}).get("provider")
        or ""
    ).strip()
    if requested and requested.lower() != "custom":
        return requested
    return resolved



def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    candidates = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue
        normalized = {**entry, "provider": provider, "model": model}
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


def _route_match_key(route: dict[str, Any]) -> tuple[str, str]:
    """``(provider, model)`` a route's ``when`` matches, model ``""`` = any model on that provider."""
    when = route.get("when")
    if not isinstance(when, dict):
        return "", ""
    return (
        str(when.get("provider") or "").strip().lower(),
        str(when.get("model") or "").strip().lower(),
    )


def get_fallback_routes(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalized per-primary ``fallback_routes`` in config order.

    Each route is ``{"provider", "model", "fallback_providers"}`` (model ``""`` = any model on that
    provider). A route is dropped — never an error — when ``when.provider`` is missing, when
    ``fallback_providers`` is absent or not a list/dict, or when a non-empty list yields no usable
    entries (those degrade to the global chain). An explicitly empty ``fallback_providers: []`` is
    kept as an empty chain: "this primary never falls back".
    """
    raw = (config or {}).get("fallback_routes")
    if not isinstance(raw, list):
        return []
    routes: list[dict[str, Any]] = []
    for route in raw:
        if not isinstance(route, dict):
            continue
        provider, model = _route_match_key(route)
        if not provider:
            continue
        declared = route.get("fallback_providers")
        if isinstance(declared, list) and not declared:
            entries: list[dict[str, Any]] = []
        elif isinstance(declared, (list, dict)):
            entries = _iter_fallback_entries(declared)
            if not entries:
                continue
        else:
            continue
        routes.append({"provider": provider, "model": model, "fallback_providers": entries})
    return routes


def match_fallback_route(
    config: dict[str, Any] | None, provider: Any, model: Any
) -> list[dict[str, Any]] | None:
    """``fallback_providers`` declared for the primary route ``provider``/``model``.

    Returns ``None`` when no route matches (callers keep the global ``fallback_providers`` chain) and
    an empty list when the matching route opted out of fallback. First match in config order wins;
    matching is case-insensitive and a route without ``when.model`` matches any model on its provider.
    """
    primary_provider = str(provider or "").strip().lower()
    if not primary_provider:
        return None
    primary_model = str(model or "").strip().lower()
    for route in get_fallback_routes(config):
        if route["provider"] != primary_provider:
            continue
        if route["model"] and route["model"] != primary_model:
            continue
        return [dict(entry) for entry in route["fallback_providers"]]
    return None


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its order. Legacy
    ``fallback_model`` entries are appended afterwards unless they target the same
    provider/model/base_url route as an earlier entry. The returned list always contains fresh dict
    copies.
    """
    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity not in seen:
                seen.add(identity)
                chain.append(entry)
    return chain
