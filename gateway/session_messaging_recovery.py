"""Rebuild a persisted messaging route when its owner has no live entry."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError


def restore_messaging_session(authority, session_id):
    """Materialize a durable non-local, non-API session for a read-only attach.

    The persisted origin is the routing authority.  Do not infer a source from the
    display row: doing so could reattach a session to a different chat or profile.
    """
    row = authority.db.get_session(session_id)
    try:
        source = SessionSource.from_dict(json.loads(row["origin_json"]))
        if source.platform in {Platform.LOCAL, Platform.API_SERVER}:
            raise RuntimeStoreError("not_found")
        route = authority.runner.session_store._generate_session_key(source)
        if (row["source"], row["session_key"], row["chat_id"], row["chat_type"], row["user_id"],
                row["thread_id"]) != (source.platform.value, route, source.chat_id, source.chat_type,
                                        source.user_id, source.thread_id):
            raise ValueError("stored routing identity mismatch")
    except RuntimeStoreError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeStoreError("storage_unavailable") from exc

    from gateway.session_authority import LiveSession
    from gateway.session_identity import restore_identity

    target = authority.physical_target(SessionRef(authority.profile_id, session_id))
    store = authority.runner.session_store
    transport_profile = row.get("transport_profile")
    restore_identity(source, runner=authority.runner, transport_profile=transport_profile)
    with store._lock:
        store._ensure_loaded_locked()
        current = store._entries.get(route)
        if current is not None and current.session_id not in authority.db.get_compression_lineage(session_id):
            raise RuntimeStoreError("admission_conflict")
        if current is None:
            now = datetime.now(UTC)
            store._entries[route] = SessionEntry(route, target, now, now, origin=source,
                                                  platform=source.platform, chat_type=source.chat_type,
                                                  transport_profile=transport_profile)
    authority.sessions.setdefault(session_id, LiveSession(source, route))
    return SessionRef(authority.profile_id, session_id)
