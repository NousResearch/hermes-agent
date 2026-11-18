"""The built-in SQLite store serves no hosted activity on any execution path, not only through its holder: the
``'shared'`` hosted-room file, the health services, the resumable-subagent probes that open their own connection, and
a kept physical admin all refuse (``ScopePinRefused``) while a hosted scope is bound, before touching a file."""

from __future__ import annotations

import pytest

import persistence_assembly as pa
from persistence import ScopePinRefused, SessionId
from persistence.conversation.sessions import NewSession
from tests.contracts.hosted.test_hosted_activity_contract import _hosted


def _create(rooms, room_id):
    return rooms.create_room(placement="shared", room_id=room_id, name="private room", members=[],
                             authority_gateway_id="gateway-a", now=100.0)


def _shared_ids(rooms):
    return [room["room_id"] for room in rooms.list_rooms(placement="shared")]


_VERBS = {
    "shared-read": lambda kept: _shared_ids(kept["rooms"]),
    "shared-write": lambda kept: _create(kept["rooms"], "foreign-created-room"),
    "shared-probe": lambda kept: kept["rooms"].probe_hosted_room(placement="shared", room_id="local-secret-room"),
    "handle-health": lambda kept: kept["health"].probe(),
    "handle-health-latch": lambda kept: kept["health"].storage_state(),
    "assembly-health": lambda kept: kept["file_health"].probe(),
    "pinned-subagents": lambda kept: kept["coordination"].list_pinned_subagents(max_resumable=5),
    "owed-subagents": lambda kept: kept["coordination"].has_owed_subagents(),
    "admin": lambda kept: kept["admin"].session_row_count(),
    "admin-static": lambda kept: kept["admin"].has_runtime_ledger_table(kept["path"]),
}


@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_kept_local_services_refuse_hosted_activity_before_touching_a_file(tmp_path, monkeypatch, verb):
    """Each service is taken from the handle (or the assembly) before the hosted scope is bound and used under it."""
    import sqlite3

    from hermes_cli import sqlite_safe_read

    handle = pa.open_persistence(pa.profile_scope(home=tmp_path), eager=True)
    try:
        sessions = handle.sessions
        kept = {"rooms": handle.hosted_rooms, "health": handle.health, "coordination": handle.coordination,
                "file_health": pa.storage_health(handle.scope), "admin": pa.open_store_admin(handle.scope),
                "path": pa.store_location(handle.scope)}
        sessions.create(NewSession(session_id=SessionId("local-secret"), source="cli"))
        _create(kept["rooms"], "local-secret-room")
        assert _shared_ids(kept["rooms"]) == ["local-secret-room"]
        local_answer = None if verb == "shared-write" else _VERBS[verb](kept)
        opens = []
        for module, name in ((sqlite_safe_read, "connect_tracked"), (sqlite3, "connect")):
            original = getattr(module, name)
            monkeypatch.setattr(module, name, lambda *a, _original=original, **k: opens.append(a[0]) or _original(*a, **k))
        with _hosted("foreign-tenant", "agent-1"):
            with pytest.raises(ScopePinRefused):
                sessions.get(SessionId("local-secret"))  # the holder's refusal, for comparison
            with pytest.raises(ScopePinRefused):
                _VERBS[verb](kept)
        assert opens == []
        monkeypatch.undo()
        assert _shared_ids(kept["rooms"]) == ["local-secret-room"]
        if verb != "shared-write":
            assert _VERBS[verb](kept) == local_answer
    finally:
        handle.close()
