"""A delivery must stay terminal when retention races a repeated enqueue."""

import sqlite3
import subprocess
import sys


def test_enqueue_during_retention_never_recreates_delivered_request(tmp_path, monkeypatch):
    from cron import delivery_queue as queue

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "deliveries.db"
    monkeypatch.setattr(queue, "DELIVERY_DB", db)
    queue.enqueue("execution", {"id": "job"}, "original")
    sends = []
    assert queue.drain(lambda *args: sends.append(args)) == 1
    connect = sqlite3.connect
    prunes = []

    class InterleavedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith("SELECT terminal_status, finished_at FROM delivery_tombstones"):
                # Pause after the real tombstone lookup, before enqueue can
                # INSERT. A separate process exercises SQLite's actual lock
                # boundary rather than the module's process-local RLock.
                result = subprocess.run(
                    [sys.executable, "-c", """
import sqlite3
import sys
from cron import delivery_queue as queue
queue.MAX_TERMINAL_DELIVERIES = 0
try:
    with sqlite3.connect(sys.argv[1], timeout=0) as conn:
        queue._prune_terminal_unlocked(conn)
except sqlite3.OperationalError as exc:
    if 'locked' not in str(exc):
        raise
    sys.exit(75)
""", str(db)], capture_output=True, text=True, timeout=30,
                )
                assert result.returncode in (0, 75), result.stderr
                prunes.append(result.returncode)
            return cursor

    with monkeypatch.context() as patch:
        patch.setattr(
            queue.sqlite3, "connect",
            lambda *args, **kwargs: connect(*args, **kwargs, factory=InterleavedConnection),
        )
        replay = queue.enqueue("execution", {"id": "job"}, "duplicate")

    assert prunes, "the concurrent retention path must actually be attempted"
    # If enqueue held the SQLite writer lock, complete the deferred pruning now.
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", 0)
    with queue._transaction() as conn:
        queue._prune_terminal_unlocked(conn)

    assert replay["status"] == "delivered"
    assert queue.get_status("execution")["status"] == "delivered"
    assert queue.drain(lambda *args: sends.append(args)) == 0
    assert len(sends) == 1
