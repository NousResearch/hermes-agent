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


def pre_agent_fallback_notice(
    primary_provider: Any, primary_model: Any, fallback_provider: Any, fallback_model: Any
) -> str:
    """User-visible one-shot line for a provider switch made during credential resolution, before
    any AIAgent exists (#74349). Shared by the messaging gateway, the TUI/Desktop gateway and cron
    so the three pre-agent fallback paths cannot drift in wording."""
    primary_desc = "/".join(str(p).strip() for p in (primary_provider, primary_model) if p) or "primary"
    fallback_desc = "/".join(str(p).strip() for p in (fallback_provider, fallback_model) if p) or "fallback"
    return f"⚠️ Provider fallback: {primary_desc} unavailable; using {fallback_desc} for this response."



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
