"""Tests for fail-closed state.db NOTADB handling and journal-mode EIO retries.

Covers:

* fail closed when a live write connection reports ``file is not a database``;
* the write-path SQLITE_IOERR retry boundary: admitted only when the callback
  has provably not run, never by closing and replaying;
* transient ``disk i/o error`` retry in ``_on_disk_journal_mode`` so a
  one-shot EIO doesn't push callers onto the fail-closed unknown-mode branch.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB, _on_disk_journal_mode


class _NotADbOnce:
    """Connection proxy that raises 'file is not a database' on execute."""

    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestFailClosedAfterNotADb:
    def test_write_does_not_reopen_after_connection_identity_breaks(
        self, tmp_path, monkeypatch
    ):
        """One connection cannot safely heal a shared DB identity change."""
        db = SessionDB(db_path=tmp_path / "state.db")
        real_conn = db._conn
        try:
            db.create_session(session_id="s1", source="cli", model="test")
            reopen = MagicMock()
            monkeypatch.setattr("hermes_state._connect_tracked_db", reopen)
            db._conn = _NotADbOnce(real_conn)
            with pytest.raises(sqlite3.DatabaseError, match="not a database"):
                db.create_session(session_id="s2", source="cli", model="test")
            reopen.assert_not_called()
        finally:
            db._conn = real_conn
            db.close()


class TestWriteIoerrRetryBoundary:
    """IOERR retry is admitted by EFFECT POSITION, not error spelling.

    ``_execute_write`` owns non-idempotent transcript/counter mutations, so
    replaying its callback is only safe when the first attempt provably did
    nothing. SQLite does not define ``SQLITE_IOERR`` as pre-effect-only (an
    IOERR at fsync/commit may or may not have landed), so the admission gate
    is "did the callback start", not "does the message say disk I/O".
    """

    def test_ioerr_on_begin_retries_because_the_callback_never_ran(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        real_conn = db._conn
        try:

            class _BeginIoerrOnce:
                def __init__(self, conn):
                    self._real = conn
                    self.begins = 0

                def execute(self, sql, *args, **kwargs):
                    if str(sql).strip().upper().startswith("BEGIN") and self.begins == 0:
                        self.begins += 1
                        raise sqlite3.OperationalError("disk I/O error")
                    return self._real.execute(sql, *args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._real, name)

            proxy = _BeginIoerrOnce(real_conn)
            db._conn = proxy
            db.create_session(session_id="s1", source="cli", model="test")
            assert proxy.begins == 1
        finally:
            db._conn = real_conn

        rows = db.list_sessions_rich(limit=10, compact_rows=True)
        assert [row["id"] for row in rows] == ["s1"]
        db.close()

    def test_ioerr_after_the_callback_mutates_does_not_replay(self, tmp_path):
        """Settlement is unknown once the callback has run — surface, don't rerun."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            calls = []

            def mutate_then_fail(conn):
                calls.append(1)
                conn.execute(
                    "INSERT INTO sessions (id, started_at, source) VALUES (?, ?, ?)",
                    (f"row-{len(calls)}", 1.0, "cli"),
                )
                raise sqlite3.OperationalError("disk I/O error")

            with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
                db._execute_write(mutate_then_fail)

            assert calls == [1], "a started write must not be replayed"
            assert db.list_sessions_rich(limit=10, compact_rows=True) == []
        finally:
            db.close()

    def test_write_ioerr_never_closes_the_connection(self, tmp_path, monkeypatch):
        """close() cancels this process's POSIX locks for every sibling fd."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            closed = []
            monkeypatch.setattr(
                type(db._conn), "close", lambda self: closed.append(1), raising=False
            )

            def always_ioerr(conn):
                raise sqlite3.OperationalError("disk I/O error")

            with pytest.raises(sqlite3.OperationalError):
                db._execute_write(always_ioerr)

            assert closed == []
            assert db._conn is not None
        finally:
            db.close()


class TestOnDiskJournalModeEioRetry:
    def _conn_raising_then(self, failures, result_rows):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = result_rows
        conn.execute.side_effect = list(failures) + [cursor]
        return conn

    def test_transient_eio_clears_on_retry(self):
        conn = self._conn_raising_then(
            [sqlite3.OperationalError("disk i/o error")] * 2, ("wal",)
        )
        assert _on_disk_journal_mode(conn) == "wal"

    def test_persistent_eio_returns_none(self):
        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("disk i/o error")
        assert _on_disk_journal_mode(conn) is None
        # Bounded: retried a handful of times, not forever.
        assert conn.execute.call_count == 4

    def test_non_eio_operational_error_fails_fast(self):
        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        assert _on_disk_journal_mode(conn) is None
        assert conn.execute.call_count == 1
