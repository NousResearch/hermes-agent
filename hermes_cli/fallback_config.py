"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


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



def _fallback_entry_from_string(value: str) -> dict[str, Any] | None:
    """Parse one compact ``provider:model`` fallback entry."""
    provider, separator, model = value.strip().partition(":")
    if separator and provider.strip() and model.strip():
        return {"provider": provider.strip(), "model": model.strip()}
    return None


def normalize_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    """Normalize mapping, JSON, and compact ``provider:model`` fallback entries."""
    entries: list[dict[str, Any]] = []

    def append_candidate(entry: Any, location: str) -> None:
        if isinstance(entry, str):
            text = entry.strip()
            try:
                decoded = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                decoded = text
            if decoded != text or isinstance(decoded, (dict, list)):
                append_candidate(decoded, location)
                return
            entry = _fallback_entry_from_string(text)
        if isinstance(entry, list):
            for index, nested in enumerate(entry):
                append_candidate(nested, f"{location}.{index}")
            return
        if not isinstance(entry, dict):
            logger.warning(
                "Ignoring fallback entry %s: expected a provider:model string or mapping with provider and model",
                location,
            )
            return
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            logger.warning("Ignoring fallback entry %s: both provider and model are required", location)
            return
        normalized = {**entry, "provider": provider, "model": model}
        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url
        entries.append(normalized)

    if raw is None:
        return []
    if isinstance(raw, list):
        for index, candidate in enumerate(raw):
            append_candidate(candidate, str(index))
    elif isinstance(raw, (dict, str)):
        append_candidate(raw, "0")
    else:
        logger.warning("Ignoring fallback configuration with unsupported type %s", type(raw).__name__)
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
        for entry in normalize_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity not in seen:
                seen.add(identity)
                chain.append(entry)
    return chain
