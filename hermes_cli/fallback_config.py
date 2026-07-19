"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import json
import os
from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        return os.getenv(key_env, "").strip() or None
    return None


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    """Yield normalised ``{provider, model, base_url?}`` dicts from any of
    the supported ``fallback_providers`` payload shapes.

    Accepts the canonical list-of-dicts, the legacy single-dict, and a
    JSON-encoded string of either — the last form exists because earlier
    ``hermes config set fallback_providers [...]`` invocations wrote the
    value through a YAML serializer that round-tripped the list as a
    quoted JSON string, and downstream code was silently treating the
    string as "no chain configured" (hermes-agent #41590). Configs that
    carry an unparseable string fall through to an empty chain rather than
    raising — the failure surfaces later when the primary provider hits a
    quota wall, which is exactly the bug a startup-time crash would hide.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, str):
        stripped = raw.strip()
        # Heuristic: only attempt JSON parse for strings that look like
        # serialised JSON (start with ``[`` or ``{``). Anything else — an
        # accidental string literal, an env-var name, etc. — is treated as
        # not-a-chain rather than raising at config-load time.
        if not stripped or stripped[0] not in "[{":
            return []
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            return []
        if isinstance(parsed, dict):
            candidates = [parsed]
        elif isinstance(parsed, list):
            candidates = parsed
        else:
            return []
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
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.

    Both keys accept the legacy single-dict shape, the canonical
    list-of-dicts shape, and a JSON-encoded string of either — see
    ``_iter_fallback_entries``. The string form is what
    ``hermes config set fallback_providers [...]`` produced before the
    YAML serializer was made list-aware; the silent "no chain" failure on
    that path was the user-facing symptom for cron jobs hitting hard
    provider quota walls.
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
