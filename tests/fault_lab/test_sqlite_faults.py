"""Fault-injection test for real cross-process SQLite write-lock contention.

Companion to #54964: a REAL separate process holds the exclusive lock —
no mocked connection, no monkeypatched cursor — so this proves the actual
``sqlite3.OperationalError`` shape a concurrent writer hits, which is what
retry/backoff code in ``hermes_state.py`` (``SessionDB``) must handle.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.fault_lab.sqlite_faults import LockHolder


def test_write_during_exclusive_lock_raises_real_operational_error(tmp_path):
    db_path = tmp_path / "fault_lab.db"
    setup_conn = sqlite3.connect(db_path)
    setup_conn.execute("CREATE TABLE t (id INTEGER)")
    setup_conn.commit()
    setup_conn.close()

    with LockHolder(db_path):
        writer = sqlite3.connect(db_path, timeout=0.5)
        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                writer.execute("INSERT INTO t VALUES (1)")
                writer.commit()
        finally:
            writer.close()

    # After the holder releases, a normal write must succeed — the fault
    # was transient contention, not a corrupted database.
    recovered = sqlite3.connect(db_path)
    recovered.execute("INSERT INTO t VALUES (2)")
    recovered.commit()
    row_count = recovered.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    recovered.close()
    assert row_count == 1
