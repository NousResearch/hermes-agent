"""Restore durable messaging routes into a restarted session authority."""

from __future__ import annotations

from gateway.config import Platform
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError

_NON_MESSAGING_SOURCES = frozenset({
    "acp", "a2a", "api_server", "bot_room", "cli", "cron", "desktop", "gui", "kanban", "local",
    "oneshot", "subagent", "tool", "tui",
})


def is_messaging_session(row: dict) -> bool:
    """Whether *row* is a platform conversation rather than a local/API producer."""
    return str(row.get("source") or "").lower() not in _NON_MESSAGING_SOURCES


def restore_messaging_session(authority, session_id: str) -> SessionRef:
    """Recreate one platform conversation's live authority entry after a restart.

    Messaging rows already have a durable routing entry written at ingress.  Unlike local
    and API sessions, they have no separate policy or API-binding receipt to restore.
    """
    row = authority.db.get_session(session_id)
    route = row.get("session_key") if row else None
    entry = authority.runner.session_store.lookup_by_session_key(route)
    if not route or entry is None or entry.origin is None:
        raise RuntimeStoreError("not_found")
    source = entry.origin
    if (not is_messaging_session(row) or source.platform in {Platform.LOCAL, Platform.API_SERVER}
            or row.get("source") != source.platform.value):
        raise RuntimeStoreError("not_found")
    if route != authority.runner.session_store._generate_session_key(source):
        raise RuntimeStoreError("admission_conflict")
    if entry.session_id not in authority.db.get_compression_lineage(session_id):
        raise RuntimeStoreError("admission_conflict")
    restored_source = getattr(authority.runner, "_restored_source", None)
    if callable(restored_source):
        source = restored_source(entry)
    if source is None:
        raise RuntimeStoreError("storage_unavailable")
    from gateway.session_authority import LiveSession
    authority.sessions.setdefault(session_id, LiveSession(source, route))
    return SessionRef(authority.profile_id, session_id)
