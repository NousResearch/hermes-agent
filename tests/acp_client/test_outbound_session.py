"""Tests for acp_client.outbound_session — OutboundSessionManager.

Mirror of ``tests/acp/test_session.py`` for the client (Hermes-as-client) side.
No real external CLI is launched; sessions are exercised against a temp
SessionDB so persistence/reconnect is covered without touching the user's real
``~/.hermes/state.db``.
"""

import pathlib

import pytest

from hermes_state import SessionDB
from acp_client.outbound_session import (
    SOURCE,
    OutboundSessionManager,
    OutboundSessionState,
)


@pytest.fixture()
def db(tmp_path):
    return SessionDB(pathlib.Path(tmp_path) / "state.db")


@pytest.fixture()
def manager(db):
    return OutboundSessionManager(db=db)


class TestRegister:
    def test_register_returns_state(self, manager):
        state = manager.register("ext-1", cwd="/tmp/work", backend="claude")
        assert isinstance(state, OutboundSessionState)
        assert state.session_id == "ext-1"
        assert state.cwd == "/tmp/work"
        assert state.backend == "claude"
        assert state.history == []
        assert state.is_running is False

    def test_register_persists_to_db_with_acp_client_source(self, manager, db):
        manager.register("ext-2", cwd="/w", backend="codex")
        row = db.get_session("ext-2")
        assert row is not None
        assert row["source"] == SOURCE

    def test_get_returns_in_memory_session(self, manager):
        state = manager.register("ext-3")
        assert manager.get("ext-3") is state

    def test_get_unknown_returns_none(self, manager):
        assert manager.get("nope") is None


class TestHistory:
    def test_record_history_appends_and_persists(self, manager, db):
        manager.register("ext-h", cwd="/w", backend="claude")
        manager.record_history("ext-h", "user", "hello")
        manager.record_history("ext-h", "assistant", "hi there")
        assert manager.get("ext-h").history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        # Derived state is persisted, not just in memory.
        assert db.get_messages_as_conversation("ext-h") == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]


class TestCancel:
    def test_cancel_sets_event_and_clears_running(self, manager):
        state = manager.register("ext-c", backend="claude")
        state.is_running = True
        assert manager.cancel("ext-c") is True
        assert state.cancel_event.is_set() is True
        assert state.is_running is False

    def test_cancel_unknown_session_returns_false(self, manager):
        assert manager.cancel("missing") is False


class TestReconnect:
    def test_get_restores_from_db_after_eviction(self, db):
        # First manager registers + records history, then "crashes".
        m1 = OutboundSessionManager(db=db)
        m1.register("ext-r", cwd="/work", backend="claude")
        m1.record_history("ext-r", "user", "task please")
        m1.record_history("ext-r", "assistant", "on it")

        # Fresh manager (simulated worker restart) has nothing in memory.
        m2 = OutboundSessionManager(db=db)
        assert m2._sessions == {}

        restored = m2.get("ext-r")
        assert restored is not None
        assert restored.cwd == "/work"
        assert restored.backend == "claude"
        assert restored.history == [
            {"role": "user", "content": "task please"},
            {"role": "assistant", "content": "on it"},
        ]

    def test_restore_ignores_non_acp_client_sessions(self, db):
        db.create_session(session_id="other", source="acp", model="m")
        m = OutboundSessionManager(db=db)
        assert m.get("other") is None

    def test_set_stop_reason_persists(self, manager):
        manager.register("ext-s", backend="claude")
        manager.set_stop_reason("ext-s", "end_turn")
        assert manager.get("ext-s").last_stop_reason == "end_turn"
        assert manager.get("ext-s").is_running is False


class TestForkAndList:
    def test_fork_deep_copies_history_under_new_id(self, manager):
        manager.register("ext-f", cwd="/w", backend="claude")
        manager.record_history("ext-f", "user", "original")
        forked = manager.fork("ext-f")
        assert forked is not None
        assert forked.session_id != "ext-f"
        assert forked.history == [{"role": "user", "content": "original"}]
        # Deep copy: mutating the fork doesn't touch the original.
        forked.history.append({"role": "user", "content": "extra"})
        assert manager.get("ext-f").history == [{"role": "user", "content": "original"}]

    def test_fork_unknown_returns_none(self, manager):
        assert manager.fork("missing") is None

    def test_list_sessions_includes_registered(self, manager):
        manager.register("ext-l1", cwd="/a", backend="claude")
        manager.register("ext-l2", cwd="/b", backend="codex")
        ids = {row["session_id"] for row in manager.list_sessions()}
        assert {"ext-l1", "ext-l2"} <= ids

    def test_remove_drops_from_memory_and_db(self, manager, db):
        manager.register("ext-d", backend="claude")
        assert manager.remove("ext-d") is True
        assert manager.get("ext-d") is None
        assert db.get_session("ext-d") is None
