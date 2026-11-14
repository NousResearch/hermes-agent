"""A process killed inside the settle transaction publishes nothing; the retry settles the whole composite exactly
once."""

from __future__ import annotations

import multiprocessing
import sqlite3
import time
from pathlib import Path

import pytest

from persistence import FenceStale
from tests.persistence_sqlite.authority._admission_support import EPOCH, admit, claim, register

SESSION = "s1"


def _settle_child(path: str, profile_dir: str, name: str, admission_id: str, generation: int, signal_path: str) -> None:
    """Hold the settle transaction open at its terminal admission UPDATE (after the worker retirement ran in it),
    signal the parent, and wait to be killed."""
    from persistence import ProfileScope
    from persistence.sqlite import SQLiteBackend

    scope = ProfileScope(profile_dir=Path(profile_dir), name=name)
    store = SQLiteBackend({scope.hermes_home_key: Path(path)}).open(scope)

    def trace(statement: str) -> None:
        if "UPDATE session_admissions SET status='terminal'" in statement:
            Path(signal_path).write_text("in-transaction")
            time.sleep(600)  # health: allow HX013 -- the child holds the transaction open until the parent kills it

    store.holder()._conn.set_trace_callback(trace)
    store.admissions.settle(epoch=EPOCH, admission_id=admission_id, generation=generation, outcome="completed",
                            result={"answer": 42})


def _row(path, sql, params=()):
    with sqlite3.connect(path) as conn:
        return conn.execute(sql, params).fetchone()


def test_process_crash_inside_settle_rolls_back_then_settles_exactly_once(seeded, scope_a, tmp_path):
    def seed(db, epoch):
        db.create_session(SESSION, source="cli")
        admit(db, SESSION, "req-1")
        claimed = claim(db, SESSION)
        register(db, SESSION, claimed["generation"], execution_id="exec-r30c")
        return claimed

    path, store, claimed = seeded(seed)
    revision = _row(path, "SELECT runtime_revision FROM sessions WHERE id=?", (SESSION,))[0]
    signal_path = tmp_path / "settle-signal"
    child = multiprocessing.get_context("spawn").Process(target=_settle_child, args=(
        str(path), str(scope_a.profile_dir), scope_a.name, claimed["admission_id"], claimed["generation"],
        str(signal_path)))
    child.start()
    try:
        deadline = time.monotonic() + 120
        while not signal_path.exists():
            assert child.is_alive(), "the child exited before reaching the terminal update"
            assert time.monotonic() < deadline, "the child never reached the terminal update"
            time.sleep(0.05)
        # The worker retirement already ran inside the child's open transaction.
    finally:
        child.kill()
        child.join(timeout=30)
    # Rolled back: the admission still started, its worker still live, no revision bump, no stored result.
    row = store.admissions.get(admission_id=claimed["admission_id"])
    assert row.status == "started"
    assert _row(path, "SELECT status FROM worker_executions WHERE execution_id='exec-r30c'")[0] == "registered"
    assert _row(path, "SELECT runtime_revision FROM sessions WHERE id=?", (SESSION,))[0] == revision
    assert store.admissions.settled_result(admission_id=claimed["admission_id"]) is None
    # The retry settles the whole composite once; a second settle is a stale claim token.
    settled = store.admissions.settle(epoch=EPOCH, admission_id=claimed["admission_id"],
                                      generation=claimed["generation"], outcome="completed", result={"answer": 42})
    assert (settled.status, settled.outcome) == ("terminal", "completed")
    assert _row(path, "SELECT status FROM worker_executions WHERE execution_id='exec-r30c'")[0] == "terminal"
    assert _row(path, "SELECT runtime_revision FROM sessions WHERE id=?", (SESSION,))[0] == revision + 1
    assert store.admissions.settled_result(admission_id=claimed["admission_id"]).result == {"answer": 42}
    with pytest.raises(FenceStale):
        store.admissions.settle(epoch=EPOCH, admission_id=claimed["admission_id"],
                                generation=claimed["generation"], outcome="completed", result={"answer": 42})
    assert _row(path, "SELECT runtime_revision FROM sessions WHERE id=?", (SESSION,))[0] == revision + 1
    # The queue reopened: the next input is claimable.
    store.admissions.admit(epoch=EPOCH, principal_id="p", session_id=SESSION, request_id="req-after",
                           payload={"text": "req-after"})
    assert store.admissions.claim(epoch=EPOCH, session_id=SESSION) is not None
