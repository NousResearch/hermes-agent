"""Concurrent gateway/worker opens must not replay a stale delivery schema."""

import sqlite3
import subprocess
import sys

import pytest

from cron import delivery_queue as queue


@pytest.mark.parametrize("failure_lane", [False, True])
def test_concurrent_migration_preserves_suppressed_outcome(tmp_path, monkeypatch, failure_lane):
    table = "deliveries"
    db = tmp_path / "deliveries.db"
    monkeypatch.setattr(queue, "DELIVERY_DB", db)
    # Cover upgrades both before and after the failure-delivery column existed.
    with sqlite3.connect(db) as conn:
        queue._initialize_schema(conn)
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        conn.execute(f"DROP TABLE {table}")
        conn.execute(schema.replace(",\u0027suppressed\u0027", ""))

        if not failure_lane:
            conn.execute("ALTER TABLE deliveries DROP COLUMN for_failure")

    child = """
import sys
from pathlib import Path
from cron import delivery_queue as queue
queue.DELIVERY_DB = Path(sys.argv[1])
queue.enqueue('muted', {}, 'diagnostic')
def suppress(job, content, for_failure):
    job['_notification_all_targets_suppressed'] = True
assert queue.drain(suppress) == 1
assert queue.get_status('muted')['status'] == 'suppressed'
"""
    connect = sqlite3.connect
    interleaved = []

    class InterleavedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith("SELECT sql FROM sqlite_master") and parameters == (table,) and not interleaved:
                interleaved.append(True)
                result = subprocess.run(
                    [sys.executable, "-c", child, str(db)],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                assert result.returncode == 0, result.stderr
            return cursor

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=InterleavedConnection))
        queue.enqueue("new", {}, "result")

    assert interleaved
    assert queue.get_status("muted")["status"] == "suppressed"
    sends = []
    assert queue.drain(lambda *args: sends.append(args)) == 1
    assert len(sends) == 1
    assert sends[0][1] == "result"
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_failed_rebuild_rolls_back_legacy_table(tmp_path, monkeypatch):
    db = tmp_path / "deliveries.db"
    monkeypatch.setattr(queue, "DELIVERY_DB", db)
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE delivery_tombstones (
            execution_id TEXT PRIMARY KEY,
            terminal_status TEXT CHECK(terminal_status IN ('delivered','failed','unknown')),
            finished_at TEXT)""")
        conn.execute("INSERT INTO delivery_tombstones VALUES ('done','delivered','2026-01-01')")

    connect = sqlite3.connect

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("INSERT INTO delivery_tombstones SELECT"):
                raise sqlite3.OperationalError("simulated rebuild failure")
            return super().execute(sql, parameters)

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=FailingConnection))
        with pytest.raises(sqlite3.OperationalError, match="simulated rebuild failure"):
            queue.get_status("done")

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT terminal_status FROM delivery_tombstones").fetchone()[0] == "delivered"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='delivery_tombstones_old'").fetchone() is None
    assert queue.get_status("done")["status"] == "delivered"
