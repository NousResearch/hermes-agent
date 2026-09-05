"""Regression tests for #102827 — self-heal reopen must serialize with writers.

A teardown owner can close() the writer while a worker still has a flush to
land; the next write reopens the connection. The reopen's setup must not
interleave with another thread's active WAL append on the same file
(structural corruption). Per-database RLock serializes reopen vs writes.
"""
import threading
from pathlib import Path

from hermes_state import SessionDB, _get_db_reopen_lock


def test_reopen_lock_identity_per_path(tmp_path):
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    assert _get_db_reopen_lock(a) is _get_db_reopen_lock(a)
    assert _get_db_reopen_lock(a) is not _get_db_reopen_lock(b)


def test_reopen_lock_is_reentrant(tmp_path):
    lock = _get_db_reopen_lock(tmp_path / "r.db")
    assert lock.acquire(blocking=False)
    try:
        # _execute_write holds it across the nested reopen call.
        assert lock.acquire(blocking=False)
    finally:
        lock.release()
        lock.release()


def test_write_after_close_still_heals(tmp_path):
    """Self-heal (#94736) keeps working under the new lock."""
    db = SessionDB(db_path=tmp_path / "heal.db")
    try:
        db.create_session("s1", "cli")
        db.append_message("s1", "user", content="hello")
        db.close()  # teardown wins the race
        assert db._conn is None
        db.append_message("s1", "assistant", content="flushed after teardown")
        rows = db.get_messages("s1")
        assert [r["role"] for r in rows] == ["user", "assistant"]
    finally:
        db.close()


def test_concurrent_close_during_flush_loses_no_writes(tmp_path):
    """Teardown close() racing workers mid-flush: every append lands."""
    db = SessionDB(db_path=tmp_path / "conc.db")
    db.create_session("s1", "cli")
    n_writes = 20
    start = threading.Event()
    errors: list = []

    def _worker(n):
        start.wait()
        for i in range(n_writes):
            try:
                db.append_message("s1", "tool", content=f"w{n} result {i}")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    start.set()
    db.close()
    db.close()
    for t in threads:
        t.join(timeout=60)
    try:
        assert not any(t.is_alive() for t in threads)
        assert errors == [], f"appends died during teardown race: {errors!r}"
        assert len(db.get_messages("s1")) == 2 * n_writes
    finally:
        db.close()
