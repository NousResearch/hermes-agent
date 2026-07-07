"""Helpers for attributing model-token usage to credential-pool labels."""
from __future__ import annotations

from typing import Any, Optional


def resolve_credential_label(agent: Any) -> Optional[str]:
    """Best-effort credential label for the current model call.

    The router selects a runtime token from the provider credential pool before
    the agent is initialized. We intentionally store only the non-secret label,
    never the token. If matching fails, return None and skip credential-level
    telemetry for that call.
    """
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    if not provider:
        return None
    api_key = str(getattr(agent, "api_key", "") or "").strip()
    pool = getattr(agent, "_credential_pool", None)
    if pool is None:
        try:
            from agent.credential_pool import load_pool

            pool = load_pool(provider)
        except Exception:
            return None
    try:
        entries = pool._available_entries(clear_expired=False, refresh=False)
    except Exception:
        try:
            entries = getattr(pool, "_entries", []) or []
        except Exception:
            return None
    for entry in entries:
        try:
            if api_key and str(getattr(entry, "runtime_api_key", "") or "").strip() == api_key:
                return str(getattr(entry, "label", "") or "").strip() or None
        except Exception:
            continue
    try:
        current = pool.current() if callable(getattr(pool, "current", None)) else None
        if current is not None:
            return str(getattr(current, "label", "") or "").strip() or None
    except Exception:
        return None
    return None
