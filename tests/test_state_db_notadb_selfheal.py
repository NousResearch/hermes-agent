"""Tests for the state.db runtime connection self-heal (PR #82280 remainder).

Covers the two independently-valuable pieces salvaged from the state.db
hardening rollup:

* one-shot reconnect when a live write connection reports
  ``file is not a database`` (backing file replaced/truncated by a sibling
  process — the connection is broken, the on-disk file may be healthy);
* transient ``disk i/o error`` retry in ``_on_disk_journal_mode`` so a
  one-shot EIO doesn't push callers onto the fail-closed unknown-mode branch.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB, _is_not_a_database_error, _on_disk_journal_mode


class _NotADbOnce:
    """Connection proxy that raises 'file is not a database' on execute."""

    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestIsNotADatabaseError:
    def test_matches_sqlite_message(self):
        assert _is_not_a_database_error(
            sqlite3.DatabaseError("file is not a database")
        )

    def test_rejects_other_database_errors(self):
        assert not _is_not_a_database_error(
            sqlite3.DatabaseError("database disk image is malformed")
        )

    def test_rejects_non_sqlite_exceptions(self):
        assert not _is_not_a_database_error(ValueError("file is not a database"))


class TestReconnectAfterNotADb:
    def test_write_self_heals_when_connection_breaks(self, tmp_path):
        """A broken connection over a healthy file reconnects and retries."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session(session_id="s1", source="cli", model="test")
            # Simulate the runtime corruption class: the connection starts
            # raising 'file is not a database' while the on-disk file is
            # perfectly healthy (sibling replaced/truncated the old inode).
            db._conn = _NotADbOnce(db._conn)

            db.create_session(session_id="s2", source="cli", model="test")

            assert db._notadb_reconnect_attempted is True
            assert db.get_session("s2") is not None
            # The pre-existing row survived (same on-disk file).
            assert db.get_session("s1") is not None
        finally:
            db.close()

    def test_reconnect_is_one_shot(self, tmp_path):
        """A second 'file is not a database' propagates instead of looping."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db._notadb_reconnect_attempted = True
            db._conn = _NotADbOnce(db._conn)
            with pytest.raises(sqlite3.DatabaseError, match="not a database"):
                db.create_session(session_id="s3", source="cli", model="test")
        finally:
            db._conn = None
            db.close()

    def test_failed_reconnect_returns_false_and_original_error_propagates(
        self, tmp_path, monkeypatch
    ):
        """If the reopen itself fails, the original write error surfaces."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            monkeypatch.setattr(
                "hermes_state._connect_tracked_db",
                MagicMock(side_effect=sqlite3.DatabaseError("file is not a database")),
            )
            db._conn = _NotADbOnce(db._conn)
            with pytest.raises(sqlite3.DatabaseError, match="not a database"):
                db.create_session(session_id="s4", source="cli", model="test")
            assert db._notadb_reconnect_attempted is True
        finally:
            db._conn = None
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
