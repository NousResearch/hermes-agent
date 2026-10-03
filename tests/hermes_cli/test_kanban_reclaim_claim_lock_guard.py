"""Tests: reclaim paths are claim-lock-aware so they can't desync a re-claimed
task (issue #36910).

A stale crash/stale-claim/max-runtime reclaim, computed from a snapshot of an
OLD worker, used to reset ``tasks.status`` back to ``ready`` with only a
``WHERE status='running'`` guard. If the task had since been reclaimed AND
re-claimed by a NEW worker (new run, new claim_lock, live pid), that stale
UPDATE clobbered the live task: ``tasks.status='ready'`` while the new
``task_runs.status='running'`` and the worker kept executing — the board showed
the task in the Ready lane and the dispatcher could treat live work as
available. The reset is now gated on the snapshot's ``claim_lock`` (and pid),
so it only fires when the task is still owned by the worker the reclaim was
computed for.
"""

from __future__ import annotations

import secrets
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def test_stale_crash_reset_rejected_for_reclaimed_task(conn):
    """A reset carrying an OLD worker's claim_lock must NOT clobber a task
    that has since been re-claimed by a new worker."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="desync", assignee="w")

    # Worker A claims, then dies.
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    old = conn.execute(
        "SELECT claim_lock, worker_pid FROM tasks WHERE id=?", (tid,)
    ).fetchone()

    # Reclaim + re-claim by worker B (alive).
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    kb.claim_task(conn, tid, claimer=f"{host}:B")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, sleeper.pid)

        # The stale reset for worker A — same shape as the guarded UPDATE in
        # detect_crashed_workers — must reject (rowcount 0) because B owns it.
        cur = conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL "
            "WHERE id=? AND status='running' AND worker_pid=? AND claim_lock IS ?",
            (tid, old["worker_pid"], old["claim_lock"]),
        )
        conn.commit()
        assert cur.rowcount == 0, "stale reclaim wrongly clobbered the re-claimed task"

        final = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert final["status"] == "running"
        assert final["claim_lock"] == f"{host}:B"
    finally:
        sleeper.terminate()


def test_genuine_crash_still_reclaims(conn):
    """When the claim_lock still matches the dead worker, the crash reclaim
    fires normally — the guard must not break the legitimate path."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="legit", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    # Rewind started_at so the launch grace window doesn't skip the check.
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,)
    )
    conn.commit()
    kbd._record_worker_exit(dead.pid, 1 << 8)  # nonzero exit → crash

    crashed = kbd.detect_crashed_workers(conn)
    assert tid in crashed
    final = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
    assert final["status"] in ("ready", "blocked", "todo")


def _stamp_running_claim_with_run(conn, *, title, assignee, worker_pid, claim_lock=None):
    """Create a running task + open task_runs row and stamp current_run_id."""
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    lock = claim_lock or f"{kb._host_prefix()}{secrets.token_hex(8)}"
    future = int(time.time()) + 3600
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, "
        "claim_expires=?, worker_pid=? WHERE id=?",
        (lock, future, worker_pid, task_id),
    )
    cur = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, status, claim_lock, claim_expires, worker_pid, started_at) "
        "VALUES (?, 'running', ?, ?, ?, ?)",
        (task_id, lock, future, worker_pid, int(time.time())),
    )
    run_id = cur.lastrowid
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))
    conn.commit()
    return task_id, run_id, lock


def _install_successor_claim(conn, task_id, *, worker_pid, claim_lock=None):
    """Install successor B without closing run A (ended_at stays NULL)."""
    lock = claim_lock or f"{kb._host_prefix()}{secrets.token_hex(8)}"
    future = int(time.time()) + 3600
    cur = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, status, claim_lock, claim_expires, worker_pid, started_at) "
        "VALUES (?, 'running', ?, ?, ?, ?)",
        (task_id, lock, future, worker_pid, int(time.time())),
    )
    run_b = cur.lastrowid
    conn.execute(
        "UPDATE tasks SET claim_lock=?, claim_expires=?, worker_pid=?, "
        "current_run_id=? WHERE id=?",
        (lock, future, worker_pid, run_b, task_id),
    )
    conn.commit()
    return run_b, lock


def _task_claim_row(conn, task_id):
    return conn.execute(
        "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()


def test_reclaim_task_expected_run_id_refuses_successor(conn):
    """reclaim_task(..., expected_run_id=A) after B is current must no-op."""
    signaled = []

    def signal_fn(pid, _sig):
        signaled.append(pid)

    tid, run_a, _lock_a = _stamp_running_claim_with_run(
        conn, title="cas-mismatch", assignee="w", worker_pid=991011,
    )
    run_b, lock_b = _install_successor_claim(conn, tid, worker_pid=991012)

    assert kb.reclaim_task(
        conn, tid, expected_run_id=run_a, signal_fn=signal_fn,
    ) is False

    row = _task_claim_row(conn, tid)
    assert row["status"] == "running"
    assert row["claim_lock"] == lock_b
    assert row["worker_pid"] == 991012
    assert int(row["current_run_id"]) == int(run_b)
    assert signaled == []


def test_reclaim_task_fail_open_omitted_expected_run_id_and_matching_id(conn):
    """Operator reclaim (no expected_run_id) still reclaims B; matching id works."""
    signaled = []

    def signal_fn(pid, _sig):
        signaled.append(pid)

    tid, _run_a, _lock_a = _stamp_running_claim_with_run(
        conn, title="fail-open-omit", assignee="w", worker_pid=991021,
    )
    _run_b, _lock_b = _install_successor_claim(conn, tid, worker_pid=991022)

    assert kb.reclaim_task(conn, tid, signal_fn=signal_fn) is True
    row = _task_claim_row(conn, tid)
    assert row["status"] in ("ready", "blocked", "todo")
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None
    assert 991022 in signaled

    signaled.clear()
    tid2, _run_a2, _lock_a2 = _stamp_running_claim_with_run(
        conn, title="fail-open-match", assignee="w", worker_pid=991023,
    )
    run_b2, _lock_b2 = _install_successor_claim(conn, tid2, worker_pid=991024)

    assert kb.reclaim_task(
        conn, tid2, expected_run_id=run_b2, signal_fn=signal_fn,
    ) is True
    row2 = _task_claim_row(conn, tid2)
    assert row2["status"] in ("ready", "blocked", "todo")
    assert row2["claim_lock"] is None
    assert row2["worker_pid"] is None
    assert 991024 in signaled
