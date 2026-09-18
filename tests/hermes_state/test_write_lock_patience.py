"""Write-lock patience for the shared state.db (#74478).

A shared state.db is legitimately held for multi-second stretches by
sibling Hermes processes (VACUUM after auto-prune, TRUNCATE checkpoint at
close on a large WAL, a long FTS pass from an older still-running
install).  The old attempt-counted retry budget (15 x <=150ms jitter)
gave up in ~1-2s of retrying, so:

- ``append_message`` failed -> the conversation loop aborted the turn as
  ``session_persistence_failed`` ("No reply: ... session storage could
  not be written") even though the store was healthy and merely busy;
- ``SessionDB()`` open failed -> the CLI disabled persistence for the
  whole run ("Failed to initialize SessionDB ... database is locked").

These tests lock the DB from a second connection for a bounded window and
assert the three contracts: transcript writes ride out long holds, open
rides out long holds, and exhausted patience raises an error that names
the real cause instead of reading like disk damage.
"""

import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB


def _hold_write_lock(db_path, hold_s, started_evt):
    """Hold the SQLite write lock on *db_path* for *hold_s* seconds."""
    conn = sqlite3.connect(str(db_path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        started_evt.set()
        time.sleep(hold_s)
        conn.execute("COMMIT")
    finally:
        conn.close()


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class TestTranscriptWritePatience:
    def test_append_message_survives_multi_second_lock_hold(self, db, tmp_path):
        """A transcript append must ride out a lock held well past the old
        ~1-2s attempt-counted budget instead of aborting the turn."""
        db.create_session("s1", "cli")

        started = threading.Event()
        # 3s hold: comfortably beyond the old worst-case retry budget,
        # comfortably inside _TRANSCRIPT_WRITE_PATIENCE_S.
        holder = threading.Thread(
            target=_hold_write_lock, args=(db.db_path, 3.0, started)
        )
        holder.start()
        try:
            assert started.wait(5.0)
            msg_id = db.append_message(
                session_id="s1", role="user", content="survived the lock"
            )
        finally:
            holder.join(timeout=10.0)
        assert not holder.is_alive()
        assert isinstance(msg_id, int)
        msgs = db.get_messages("s1")
        assert any(m["content"] == "survived the lock" for m in msgs)

    def test_transcript_patience_outlasts_routine_patience(self, db):
        """append_message must be given MORE patience than routine writes —
        the invariant that lets background writers give up while the
        turn-critical append keeps waiting."""
        assert db._TRANSCRIPT_WRITE_PATIENCE_S > db._WRITE_PATIENCE_S
        # Both budgets must comfortably exceed the old ~2.25s worst case
        # (15 attempts x 150ms) that lost races against real maintenance.
        assert db._WRITE_PATIENCE_S >= 10.0
        assert db._TRANSCRIPT_WRITE_PATIENCE_S >= 30.0

    def test_exhausted_patience_names_the_real_cause(self, db, monkeypatch):
        """When patience genuinely runs out, the error must say the lock was
        held by another process — not read like disk/permission damage."""
        monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.2)

        started = threading.Event()
        holder = threading.Thread(
            target=_hold_write_lock, args=(db.db_path, 2.0, started)
        )
        holder.start()
        try:
            assert started.wait(5.0)
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                db.set_meta("k", "v")  # routine write, short patience
        finally:
            holder.join(timeout=10.0)
        assert not holder.is_alive()
        text = str(excinfo.value)
        assert "another Hermes process" in text
        assert "healthy" in text

    def test_write_succeeds_immediately_when_uncontended(self, db):
        """Patience must cost nothing when there is no contention."""
        db.create_session("s2", "cli")
        t0 = time.monotonic()
        db.append_message(session_id="s2", role="user", content="fast")
        assert time.monotonic() - t0 < 5.0  # loose: no patience-length stall


class TestOpenLockPatience:
    def test_open_survives_multi_second_lock_hold(self, tmp_path):
        """SessionDB() open must wait out a sibling's lock hold instead of
        disabling persistence for the whole run."""
        db_path = tmp_path / "state.db"
        # Create + close so the schema exists (open still runs reconcile DDL
        # through the same 1s-timeout connection).
        SessionDB(db_path=db_path).close()

        started = threading.Event()
        holder = threading.Thread(
            target=_hold_write_lock, args=(db_path, 3.0, started)
        )
        holder.start()
        try:
            assert started.wait(5.0)
            db = SessionDB(db_path=db_path)  # must NOT raise
        finally:
            holder.join(timeout=10.0)
        assert not holder.is_alive()
        try:
            db.create_session("s-open", "cli")
            db.append_message(session_id="s-open", role="user", content="ok")
            assert len(db.get_messages("s-open")) == 1
        finally:
            db.close()

    def test_open_propagates_non_lock_errors_immediately(self, tmp_path):
        """A non-lock open failure must not sit in the patience loop."""
        # A directory is not openable as a database file — raises an
        # OperationalError that is NOT the locked/busy class.
        bad_path = tmp_path / "state.db"
        bad_path.mkdir()
        t0 = time.monotonic()
        with pytest.raises(sqlite3.Error):
            SessionDB(db_path=bad_path)
        # Must fail well before a full patience window (loose bound).
        assert time.monotonic() - t0 < 15.0
