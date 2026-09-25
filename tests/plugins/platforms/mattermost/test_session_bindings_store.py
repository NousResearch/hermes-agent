"""Persistent Mattermost session binding store contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from plugins.platforms.mattermost.session_bindings import (
    BindingValidationError,
    MattermostSessionBindingStore,
    SCHEMA_VERSION,
)


def _store(tmp_path):
    return MattermostSessionBindingStore(tmp_path / "bindings.db")


def test_store_creation_and_persistence_after_reopen(tmp_path):
    store = _store(tmp_path)
    created = store.replace("session-1", "channel1", "root1")

    reopened = _store(tmp_path)
    assert reopened.get_by_session("session-1") == created
    assert reopened.resolve("channel1", "root1") == created

    import sqlite3

    conn = sqlite3.connect(tmp_path / "bindings.db")
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def test_create_read_resolve_and_delete(tmp_path):
    store = _store(tmp_path)
    created = store.replace("session-1", "channel1", "root1")

    assert created.session_id == "session-1"
    assert store.get_by_session(" session-1 ") == created
    assert store.resolve("channel1", "root1") == created
    assert store.delete("session-1") is True
    assert store.delete("session-1") is False
    assert store.get_by_session("session-1") is None


def test_list_bindings_is_recent_first_and_paginated(tmp_path, monkeypatch):
    store = _store(tmp_path)
    times = iter((10.0, 20.0, 30.0))
    monkeypatch.setattr("plugins.platforms.mattermost.session_bindings.time.time", lambda: next(times))
    store.replace("session-a", "channel1", "root1")
    store.replace("session-b", "channel2", "root2")
    store.replace("session-c", "channel3", "root3")

    assert [item.session_id for item in store.list_bindings(limit=2)] == ["session-c", "session-b"]
    assert [item.session_id for item in store.list_bindings(limit=2, offset=2)] == ["session-a"]


def test_replace_enforces_both_unique_relationships(tmp_path):
    store = _store(tmp_path)
    store.replace("session-a", "channel1", "root1")
    store.replace("session-a", "channel2", "root2")
    assert store.resolve("channel1", "root1") is None
    assert store.resolve("channel2", "root2").session_id == "session-a"

    store.replace("session-b", "channel2", "root2")
    assert store.get_by_session("session-a") is None
    assert store.get_by_session("session-b").root_post_id == "root2"


def test_parallel_replacements_leave_one_valid_owner(tmp_path):
    path = tmp_path / "bindings.db"
    MattermostSessionBindingStore(path)

    def write(index: int):
        return MattermostSessionBindingStore(path).replace(
            f"session-{index}", "sharedchannel", "sharedroot"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(24)))

    store = MattermostSessionBindingStore(path)
    winner = store.resolve("sharedchannel", "sharedroot")
    assert winner is not None
    assert store.get_by_session(winner.session_id) == winner


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", ""),
        ("session_id", "bad/session"),
        ("session_id", "s" * 257),
        ("channel_id", "bad channel"),
        ("root_post_id", "r" * 65),
    ],
)
def test_invalid_identifiers_fail_closed(tmp_path, field, value):
    values = {"session_id": "session-1", "channel_id": "channel1", "root_post_id": "root1"}
    values[field] = value

    with pytest.raises(BindingValidationError):
        _store(tmp_path).replace(**values)
