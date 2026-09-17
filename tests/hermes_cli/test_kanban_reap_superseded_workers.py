"""Tests: the dispatcher reaps superseded runs' still-live worker processes
before spawning a replacement (goal-mode review/fix cycles used to stack
4-6 workers on one task, t_e46a5fea).

A run reaches a terminal outcome (review_requested / changes_requested / ...)
in the DB while its worker process is still alive. The respawn guard must find
that PID (via the append-only ``spawned`` event — ``task_runs.worker_pid`` is
cleared when the run closes) and terminate it, so two workers never mutate one
card concurrently. The reap is cmdline-scoped to avoid PID recycling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _spawned_event(conn, task_id, run_id, pid):
    """Record a spawned event the way _set_worker_pid does (append-only)."""
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, 'spawned', ?, ?)",
        (task_id, run_id, json.dumps({"pid": pid}), int(time.time())),
    )
    conn.commit()


def test_reap_skips_open_run_and_unmatched_pid(conn, monkeypatch):
    """Only CLOSED runs' workers are reaped; an innocent PID (no kanban worker
    cmdline) is never signalled. PID recycling is the risk that makes the
    cmdline check load-bearing: a closed run's recorded PID may now belong to
    something else entirely."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="reap", assignee="w")

    # Old run: claimed, spawned, then closed by its terminal handoff.
    kb.claim_task(conn, tid, claimer=f"{host}:old")
    old_run = conn.execute(
        "SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()["id"]
    assert kb.request_review(conn, tid, summary="impl done", expected_run_id=old_run)

    # Innocent live process with a PID that must NOT be killed: ourselves.
    _spawned_event(conn, tid, old_run, os.getpid())

    # The review lane claims the task, opening a new (still open) run.
    claimed = kb.claim_review_task(conn, tid, claimer=f"{host}:new")
    assert claimed is not None
    new_run = conn.execute(
        "SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()["id"]
    assert new_run != old_run
    _spawned_event(conn, tid, new_run, os.getpid())

    def _must_not_signal(*_args, **_kwargs):
        pytest.fail("reap signalled a PID that is not a worker of this task")

    monkeypatch.setattr(kbd, "_terminate_reclaimed_worker", _must_not_signal)
    # Our own cmdline is not a kanban worker for this task → not reaped.
    assert kbd._reap_superseded_run_workers(conn, tid, keep_run_id=new_run) == 0
    assert kb._pid_alive(os.getpid())


def test_reap_terminates_superseded_live_worker(conn, monkeypatch):
    """A closed run whose spawned PID is a live kanban worker for THIS task is
    SIGTERMed before the replacement spawns."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="reap-live", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:old")
    old_run = conn.execute(
        "SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()["id"]
    assert kb.request_review(conn, tid, summary="impl done", expected_run_id=old_run)

    # Simulate a lingering worker: a real sleep child whose cmdline we will
    # fake-match as a kanban worker for tid.
    linger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _spawned_event(conn, tid, old_run, linger.pid)
        # cmdline guard sees through the fake process only if we patch the
        # identity check to confirm this pid belongs to our task.
        monkeypatch.setattr(
            kbd, "_cmdline_belongs_to_kanban_worker", lambda pid, task: pid == linger.pid
        )
        assert kb._pid_alive(linger.pid)
        assert kbd._reap_superseded_run_workers(conn, tid) == 1
        # Give the SIGTERM a moment to land.
        linger.wait(timeout=10)
        assert linger.returncode is not None
    finally:
        if linger.poll() is None:
            linger.kill()
            linger.wait()


def test_reap_noop_without_spawned_events(conn):
    tid = kb.create_task(conn, title="clean", assignee="w")
    assert kbd._reap_superseded_run_workers(conn, tid) == 0


def _sleep_worker_cmd(tid):
    """argv whose cmdline reads as a kanban worker scoped to ``tid``."""
    return [sys.executable, "-c", "import time; time.sleep(120)", f"work kanban task {tid}"]


def test_full_review_fix_cycle_keeps_single_live_worker(
    conn, all_assignees_spawnable,
):
    """Replay the t_e46a5fea cycle (implement -> review -> fix -> review ->
    complete) through the real dispatcher. Old workers deliberately do NOT exit
    after their terminal tool (the bug); the respawn reap must SIGTERM each
    superseded process before the next spawn, so at every instant at most one
    worker process is alive for the task."""
    import time as _time

    tid = kb.create_task(
        conn, title="cycle", assignee="dev",
        body="acceptance: step done", goal_mode=True,
    )
    live: dict[int, subprocess.Popen] = {}

    def spawn_fn(task, workspace, board=None):
        p = subprocess.Popen(_sleep_worker_cmd(tid))
        live[p.pid] = p
        return p.pid

    def live_pids():
        return [pid for pid, p in list(live.items()) if p.poll() is None]

    def run_id():
        row = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()
        return row["current_run_id"]

    try:
        # Tick 1 — implementation run spawns (no prior runs).
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert any(row[0] == tid for row in res.spawned)
        first_run = run_id()
        assert len(live_pids()) == 1

        # Implementer's terminal tool: request_review. Process stays alive (bug).
        kb.request_review(conn, tid, summary="impl done", expected_run_id=first_run)
        assert len(live_pids()) == 1  # orphan still alive — the bug

        # Tick 2 — review lane spawns; the reap must kill run 1's worker first.
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert any(row[0] == tid for row in res.spawned)
        second_run = run_id()
        assert second_run != first_run
        # Old worker reaped before/around spawn; exactly one remains.
        assert len(live_pids()) == 1, f"expected 1 live worker, got {live_pids()}"

        # Reviewer's terminal tool: request_changes.
        kb.request_changes(conn, tid, reason="fix needed", expected_run_id=second_run)

        # Tick 3 — fixer lane spawns; reap run 2's worker.
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert any(row[0] == tid for row in res.spawned)
        third_run = run_id()
        assert third_run != second_run
        assert len(live_pids()) == 1

        # Fixer requests review again.
        kb.request_review(conn, tid, summary="fixed", expected_run_id=third_run)

        # Tick 4 — review lane spawns; reap run 3's worker.
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert any(row[0] == tid for row in res.spawned)
        fourth_run = run_id()
        assert fourth_run != third_run
        assert len(live_pids()) == 1

        # Final reviewer completes the task.
        kb.complete_task(
            conn, tid, summary="approved", created_cards=[],
            expected_run_id=fourth_run,
        )
        # Even the final worker is eventually gone (or at most one remains).
        assert len(live_pids()) <= 1
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "done"
    finally:
        for p in live.values():
            if p.poll() is None:
                p.kill()
                p.wait()
