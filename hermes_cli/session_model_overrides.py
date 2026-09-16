"""Operator surface for persisted gateway session /model overrides (#102658).

Uses the existing SessionStore.model_override field. Does not add a second persist store.
"""

from __future__ import annotations

from typing import Any


def overrides_from_store(store) -> list[dict[str, Any]]:
    """Return {session_id, session_key, model, provider} for entries with an override."""
    rows = []
    for entry in store.list_sessions():
        ov = getattr(entry, "model_override", None) or {}
        model = ov.get("model")
        if not model:
            continue
        rows.append({
            "session_id": entry.session_id,
            "session_key": entry.session_key,
            "model": model,
            "provider": ov.get("provider") or "",
        })
    return rows


def resolve_override_entry(store, needle: str):
    """Match session_id, session_key, or a unique prefix of either."""
    n = (needle or "").strip()
    if not n:
        return None
    hits = []
    for entry in store.list_sessions():
        sid = entry.session_id or ""
        key = entry.session_key or ""
        if n in (sid, key) or sid.startswith(n) or key.startswith(n):
            hits.append(entry)
    if len(hits) == 1:
        return hits[0]
    exact = [e for e in hits if n in (e.session_id, e.session_key)]
    return exact[0] if len(exact) == 1 else None


def clear_model_override(store, needle: str) -> dict[str, Any] | None:
    """Clear persisted override so the session falls back to config default. None if unresolved."""
    entry = resolve_override_entry(store, needle)
    if entry is None:
        return None
    store.set_model_override(entry.session_key, None)
    return {
        "session_id": entry.session_id,
        "session_key": entry.session_key,
        "cleared": True,
    }
