"""Tests for session handoff (CLI to gateway platform).

The handoff state machine lives on the ``sessions`` table:

    None  → "pending" → "running" → ("completed" | "failed")

CLI side calls ``request_handoff`` and poll-waits on ``get_handoff_state``.
Gateway side iterates ``list_pending_handoffs``, calls ``claim_handoff`` to
flip pending → running, and finishes with ``complete_handoff`` or
``fail_handoff``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import pytest

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages

class TestHandoffStateDB:
    """Test the handoff schema + the store's coordination verbs over it."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        with open_persistence(profile_scope(home=home)) as store:
            yield store

    def _make_session(self, db, session_id, source="cli", title=None):
        """Insert a session row directly for testing."""
        def _do(conn):
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, source, title, started_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, source, title, time.time()),
            )
        raw_holder(db)._execute_write(_do)

    def test_list_pending_handoffs_excludes_running_and_terminal(self, db):
        a, b, c, d = "sess-a", "sess-b", "sess-c", "sess-d"
        for sid in (a, b, c, d):
            self._make_session(db, sid)

        db.coordination.request_handoff(session_id=a, platform="telegram")
        db.coordination.request_handoff(session_id=b, platform="discord")
        db.coordination.request_handoff(session_id=c, platform="telegram")
        db.coordination.claim_handoff(session_id=c)  # c is now running, not pending
        db.coordination.request_handoff(session_id=d, platform="slack")
        db.coordination.claim_handoff(session_id=d)
        db.coordination.complete_handoff(session_id=d)  # d is terminal

        pending = db.coordination.list_pending_handoffs()
        ids = [r.session_id for r in pending]
        assert set(ids) == {a, b}

    def test_complete_handoff_clears_error(self, db):
        sid = "sess-complete"
        self._make_session(db, sid)
        db.coordination.request_handoff(session_id=sid, platform="telegram")
        db.coordination.claim_handoff(session_id=sid)
        db.coordination.fail_handoff(session_id=sid, error="transient")
        # User retries; mock the watcher path
        db.coordination.request_handoff(session_id=sid, platform="telegram")
        db.coordination.claim_handoff(session_id=sid)
        db.coordination.complete_handoff(session_id=sid)

        state = db.coordination.get_handoff(session_id=sid)
        assert state.state == "completed"
        assert state.error is None

    def test_full_pending_to_completed_flow(self, db):
        """End-to-end sequence the CLI + gateway watcher follow."""
        sid = "sess-flow"
        self._make_session(db, sid, title="my session")
        seed_messages(db, sid, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"})

        # CLI: request handoff
        assert db.coordination.request_handoff(session_id=sid, platform="telegram") is True
        assert db.coordination.get_handoff(session_id=sid).state == "pending"

        # Gateway watcher: discover + claim
        pending = db.coordination.list_pending_handoffs()
        assert len(pending) == 1
        assert pending[0].session_id == sid
        assert db.coordination.claim_handoff(session_id=sid) is True
        assert db.coordination.get_handoff(session_id=sid).state == "running"

        # Gateway reads the transcript (real flow uses
        # session_store.switch_session which reads the same table).
        messages = transcript_rows(db, sid)
        assert [m["role"] for m in messages] == ["user", "assistant"]

        # Gateway: mark completed
        db.coordination.complete_handoff(session_id=sid)
        assert db.coordination.get_handoff(session_id=sid).state == "completed"
        assert db.coordination.list_pending_handoffs() == ()

    def test_handoff_reads_use_read_context_during_reconnect(self, db, monkeypatch):
        """Handoff polling must not borrow the reconnectable writer handle."""
        sid = "sess-read-path"
        self._make_session(db, sid)
        db.coordination.request_handoff(session_id=sid, platform="discord")

        entered = []
        holder = raw_holder(db)
        original = holder._read_ctx

        @contextmanager
        def traced_read_ctx():
            entered.append(True)
            with original() as conn:
                yield conn

        monkeypatch.setattr(holder, "_read_ctx", traced_read_ctx)
        assert db.coordination.get_handoff(session_id=sid).state == "pending"
        assert [row.session_id for row in db.coordination.list_pending_handoffs()] == [sid]
        assert len(entered) == 2
