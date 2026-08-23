"""Tests for fail-closed state.db NOTADB handling and journal-mode EIO retries.

Covers the two independently-valuable pieces salvaged from the state.db
hardening rollup:

* fail closed when a live write connection reports ``file is not a database``;
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
