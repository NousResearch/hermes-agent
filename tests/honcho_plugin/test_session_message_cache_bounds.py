"""Bounds for Honcho's local post-sync message cache (#71461)."""

import threading

from plugins.memory.honcho import session as session_mod
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


def _session(messages):
    return HonchoSession(
        key="channel:1", user_peer_id="user", assistant_peer_id="assistant",
        honcho_session_id="session-1", messages=messages,
    )


def _message(index: int, *, synced: bool = True, size: int = 32):
    return {"role": "user", "content": f"{index}:" + "x" * size, "_synced": synced}


def test_trim_synced_cache_keeps_recent_tail_within_count_bound(monkeypatch):
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_COUNT", 4)
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_BYTES", 1_000_000)
    session = _session([_message(index) for index in range(8)])

    assert session.trim_synced_cache() == 4
    assert [message["content"].split(":", 1)[0] for message in session.messages] == ["4", "5", "6", "7"]


def test_trim_synced_cache_bounds_retained_heap_bytes(monkeypatch):
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_COUNT", 100)
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_BYTES", 2_000)
    session = _session([_message(index, size=700) for index in range(8)])

    assert session.trim_synced_cache() > 0
    assert sum(session_mod._retained_heap_bytes(message) for message in session.messages) <= 2_000


def test_trim_synced_cache_never_drops_unsynced_rows(monkeypatch):
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_COUNT", 1)
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_BYTES", 1)
    unsynced = _message(2, synced=False, size=20_000)
    session = _session([_message(1), unsynced, _message(3)])

    session.trim_synced_cache()

    assert unsynced in session.messages
    assert all(not message.get("_synced") for message in session.messages)


def test_flush_without_new_messages_still_trims_old_durable_history(monkeypatch):
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_COUNT", 2)
    monkeypatch.setattr(session_mod, "_SYNCED_MESSAGE_CACHE_MAX_BYTES", 1_000_000)
    session = _session([_message(index) for index in range(6)])
    manager = HonchoSessionManager.__new__(HonchoSessionManager)
    manager._cache_lock = threading.RLock()
    manager._cache = {session.key: session}

    assert manager._flush_session(session) is True
    assert [message["content"].split(":", 1)[0] for message in session.messages] == ["4", "5"]
