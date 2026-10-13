"""Tests for SessionStore.restore_session — the gateway /redo primitive.

/undo soft-deletes (rows stay in state.db with active=0), which is what makes
a redo possible at all: restore_session reactivates exactly the rows the undo
archived. These tests drive the real SessionStore against a temp state.db.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import hermes_undo
from hermes_state import SessionDB
from gateway.config import GatewayConfig
from gateway.session import SessionStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    s = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    s._db = db
    hermes_undo.clear_state()
    hermes_undo._session_db = db
    yield s
    hermes_undo.clear_state()
    hermes_undo._session_db = None


def _seed(store, sid, source="telegram", turns=3):
    store._db.create_session(sid, source=source)
    for i in range(1, turns + 1):
        store._db.append_message(sid, "user", f"q{i}")
        store._db.append_message(sid, "assistant", f"a{i}")
    return sid


def _roles(store, sid):
    return [m["role"] for m in store.load_transcript(sid)]


class TestRoundTrip:
    def test_undo_then_redo_restores_the_transcript(self, store):
        sid = _seed(store, "gw-redo-1")
        before = _roles(store, sid)

        store.rewind_session(sid)
        assert _roles(store, sid) == before[:-2]

        result = store.restore_session(sid)
        assert result["reactivated_count"] == 2
        assert _roles(store, sid) == before

    def test_multi_turn_undo_then_redo(self, store):
        sid = _seed(store, "gw-redo-2")
        before = _roles(store, sid)

        store.rewind_session(sid, 2)
        assert _roles(store, sid) == before[:-4]

        result = store.restore_session(sid, 1)
        assert result["reactivated_count"] == 4
        assert _roles(store, sid) == before

    def test_redo_reactivates_rather_than_reinserting(self, store):
        """No duplicate rows: the same physical ids come back."""
        sid = _seed(store, "gw-redo-3", turns=2)
        all_before = [
            m["id"] for m in store._db.get_messages(sid, include_inactive=True)
        ]
        store.rewind_session(sid)
        store.restore_session(sid)
        all_after = [
            m["id"] for m in store._db.get_messages(sid, include_inactive=True)
        ]
        assert all_before == all_after


class TestNothingToRedo:
    def test_without_an_undo_reports_nothing(self, store):
        sid = _seed(store, "gw-redo-4")
        result = store.restore_session(sid)
        assert result["reactivated_count"] == 0
        assert "nothing to redo" in result["message"]

    def test_no_db_returns_none(self, store):
        sid = _seed(store, "gw-redo-5")
        store._db = None
        assert store.restore_session(sid) is None

    def test_lost_stack_reports_the_restart_reason(self, store):
        sid = _seed(store, "gw-redo-6")
        store.rewind_session(sid)
        hermes_undo.clear_state(sid)  # as a gateway restart would
        result = store.restore_session(sid)
        assert result["reactivated_count"] == 0
        assert "restart" in result["message"]


class TestErrorHonesty:
    def test_transient_lock_reports_busy_not_nothing(self, store, monkeypatch):
        """A momentary lock must never read as 'your redo branch is gone'."""
        sid = _seed(store, "gw-redo-7")
        store.rewind_session(sid)

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(hermes_undo, "redo", boom)
        assert store.restore_session(sid) == {"status": "busy"}

    def test_real_fault_reports_error_not_nothing(self, store, monkeypatch):
        sid = _seed(store, "gw-redo-8")
        store.rewind_session(sid)

        def boom(*a, **k):
            raise RuntimeError("genuine bug")

        monkeypatch.setattr(hermes_undo, "redo", boom)
        assert store.restore_session(sid) == {"status": "error"}

    def test_non_lock_db_error_is_error_not_busy(self, store, monkeypatch):
        sid = _seed(store, "gw-redo-9")
        store.rewind_session(sid)

        def boom(*a, **k):
            raise sqlite3.OperationalError("no such column: bogus")

        monkeypatch.setattr(hermes_undo, "redo", boom)
        assert store.restore_session(sid) == {"status": "error"}


class TestRedoInvalidation:
    def test_new_user_message_kills_the_redo_branch(self, store):
        """Typing after an undo makes the undone turns unreachable."""
        sid = _seed(store, "gw-redo-10", turns=2)
        store.rewind_session(sid)
        after_undo = _roles(store, sid)

        store.append_to_transcript(sid, {"role": "user", "content": "new ask"})

        result = store.restore_session(sid)
        assert result["reactivated_count"] == 0
        # The undone rows stayed archived; only the new message is present.
        assert _roles(store, sid) == after_undo + ["user"]

    def test_assistant_append_does_not_kill_the_branch(self, store):
        """Only USER input ends the redo branch, mirroring an editor."""
        sid = _seed(store, "gw-redo-11", turns=2)
        store.rewind_session(sid)
        store.append_to_transcript(sid, {"role": "assistant", "content": "note"})
        assert hermes_undo.has_redoable(sid) is True


class TestRetryDoesNotBankARedo:
    def test_retry_rewind_is_not_redoable(self, store):
        """/retry replaces the turn; it is not a branch to return to."""
        sid = _seed(store, "gw-redo-12", turns=2)
        # The composite-carrier retry path returns None for a plain user row,
        # which is fine — the contract under test is that it banks nothing.
        store.rewind_session(sid, 1, require_retryable_composite=True)
        assert hermes_undo.has_redoable(sid) is False
