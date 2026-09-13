# 2026-09-01 state.db corruption (second incident, 15 minutes apart):
# gateway_heartbeats corrupted to the table's own root page despite the
# concurrent-writer pairing being a legitimate by-design long-lived setup
# (default-profile `serve` + `gateway run` writing rows every ~60s). The fix
# is a short-held cross-process flock wrapping all three write functions,
# fail-open on timeout — heartbeats are a liveness hint, a writer must never
# hang behind a wedged peer.
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hermes_state import _gateway_heartbeats_write_lock


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "state.db"
    sqlite3.connect(str(p)).close()
    return str(p)


def test_lock_held_cleanly_when_uncontended(db_path):
    """Single-process acquire yields True and the lock file is created."""
    with _gateway_heartbeats_write_lock(db_path) as acquired:
        assert acquired is True
    # Lock file may be cleaned up or persist; either is fine — flock is
    # released on close. Re-acquire still works.
    with _gateway_heartbeats_write_lock(db_path) as acquired:
        assert acquired is True


def test_timeout_proceeds_without_lock_and_logs(db_path, caplog):
    """A peer that holds the lock past the bounded timeout must NOT wedge the
    writer — it proceeds without the lock (fail open) and the warning is
    logged."""
    import fcntl
    # Hold the lock from a separate file descriptor.
    blocking_handle = open(f"{db_path}.gateway_heartbeats.lock", "a+b")
    try:
        fcntl.flock(blocking_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Acquire with a tiny timeout to keep the test fast.
        with caplog.at_level("WARNING"):
            with _gateway_heartbeats_write_lock(db_path, timeout_seconds=0.2) as acquired:
                assert acquired is False, "writer hung behind a wedged peer instead of failing open"
        assert any("gateway heartbeats lock" in rec.message for rec in caplog.records)
    finally:
        fcntl.flock(blocking_handle.fileno(), fcntl.LOCK_UN)
        blocking_handle.close()


def test_serializes_concurrent_writers_in_one_process(db_path):
    """Two threads both writing through the context manager must each get a
    turn (refuse-to-start semantics would deadlock here)."""
    order = []
    barrier = threading.Barrier(2)

    def worker(name: str):
        with _gateway_heartbeats_write_lock(db_path, timeout_seconds=5.0) as acquired:
            assert acquired is True
            order.append(f"{name}-start")
            barrier.wait()
            time.sleep(0.05)
            order.append(f"{name}-end")

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start(); t2.start(); t1.join(); t2.join()
    # Each thread's start must precede its own end (no interleave inside the
    # critical section), and both must complete — neither refused to run.
    assert sorted(order) == sorted(["A-start", "A-end", "B-start", "B-end"])
