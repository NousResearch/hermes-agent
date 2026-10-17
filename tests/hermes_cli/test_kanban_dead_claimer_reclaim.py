"""A pid-less claim is safe to reclaim only when its local claimer is gone."""

import os
import psutil
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with kbc.connect(tmp_path / "kanban.db") as conn:
        yield conn


def _claim(conn, lock, *, expired=False, age=None):
    """A pid-less running claim. ``age`` backdates the run start (seconds);
    default is past the dead-claimer launch bound."""
    tid = kb.create_task(conn, title="pid-less claim", assignee="worker")
    kb.claim_task(conn, tid)
    age = kbd.DEAD_CLAIMER_LAUNCH_BOUND_SECONDS + 5 if age is None else age
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_lock=?, claim_expires=?, started_at=? WHERE id=?",
            (lock, int(time.time()) - 10 if expired else int(time.time()) + 600,
             int(time.time()) - age, tid),
        )
        conn.execute(
            "UPDATE task_runs SET claim_lock=?, started_at=? "
            "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
            (lock, int(time.time()) - age, tid),
        )
    return tid


def _dead_pid():
    c = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    c.wait(timeout=10)
    assert not psutil.pid_exists(c.pid)
    return c.pid


def _release(conn, tid, path):
    if path == "manual":
        return kb.reclaim_task(conn, tid)
    if path == "ttl":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (int(time.time()) - 1, tid))
        return kb.release_stale_claims(conn) == 1
    if path == "stale":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_heartbeat_at=? WHERE id=?", (int(time.time()) - 7200, tid))
        return kbd.detect_stale_running(conn, stale_timeout_seconds=30) == [tid]
    raise AssertionError(path)


@pytest.mark.parametrize("expired", [False, True])
def test_dead_local_claimer_releases_claim_and_records_reclaimed(board, monkeypatch, expired):
    dead_pid = 999991
    lock = f"{kb._host_prefix()}{dead_pid}"
    tid = _claim(board, lock, expired=expired)

    def dead(pid):
        assert pid == dead_pid
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", dead)
    if expired:
        assert kb.release_stale_claims(board) == 1
    else:
        assert kb.reclaim_task(board, tid) is True
    assert kb.get_task(board, tid).status == "ready"
    assert kb.get_task(board, tid).claim_lock is None
    events = [e for e in kb.list_events(board, tid) if e.kind == "reclaimed"]
    assert len(events) == 1
    assert events[0].payload["claimer_pid_dead"] == dead_pid
    assert events[0].payload["terminated"] is True
    assert not [e for e in kb.list_events(board, tid) if e.kind == "reclaim_deferred"]


@pytest.mark.parametrize("expired", [False, True])
def test_live_local_claimer_keeps_pidless_claim(board, monkeypatch, expired):
    lock = f"{kb._host_prefix()}{os.getpid()}"
    tid = _claim(board, lock, expired=expired)
    seen = []
    real_process = psutil.Process

    def alive(pid):
        seen.append(pid)
        return real_process(pid)

    monkeypatch.setattr(psutil, "Process", alive)
    if expired:
        assert kb.release_stale_claims(board) == 0
    else:
        assert kb.reclaim_task(board, tid) is False
    assert seen == [os.getpid()]
    assert kb.get_task(board, tid).status == "running"
    assert kb.get_task(board, tid).claim_lock == lock
    assert not [e for e in kb.list_events(board, tid) if e.kind == "reclaimed"]


def test_foreign_pidless_claimer_is_not_probed(board, monkeypatch):
    tid = _claim(board, "foreign-host:999991")
    monkeypatch.setattr(psutil, "Process", lambda *_: pytest.fail("foreign PID was probed"))
    assert kb.reclaim_task(board, tid) is True  # preserve foreign-host release policy
    assert kb.get_task(board, tid).status == "ready"


@pytest.mark.parametrize("path", ["manual", "ttl", "stale"])
def test_unstamped_orphan_heartbeat_prevents_second_worker(board, monkeypatch, path):
    """Popen succeeds, gateway dies before PID stamp, detached worker heartbeats."""
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    claimer = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    claimer.wait(timeout=10)
    assert not psutil.pid_exists(claimer.pid)
    lock = f"{kb._host_prefix()}{claimer.pid}"
    tid = _claim(board, lock, expired=False)
    run_id = kb._current_run_id(board, tid)
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        assert kb.heartbeat_claim(board, tid, claimer=lock)
        assert kbd.heartbeat_worker(board, tid, note="unstamped worker alive")
        assert kb.get_task(board, tid).worker_pid is None
        if path == "ttl":
            with kb.write_txn(board):
                board.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (int(time.time()) - 1, tid))
            assert kb.release_stale_claims(board) == 0
        elif path == "stale":
            old = int(time.time()) - 7200
            with kb.write_txn(board):
                board.execute("UPDATE task_runs SET started_at=? WHERE id=?", (old, run_id))
                board.execute("UPDATE tasks SET last_heartbeat_at=? WHERE id=?", (old, tid))
            assert kbd.detect_stale_running(board, stale_timeout_seconds=60) == []
        else:
            assert kb.reclaim_task(board, tid) is False
        spawned = []
        kbd.dispatch_once(board, spawn_fn=lambda task, workspace, board=None: spawned.append(task.id) or 424242)
        assert tid not in spawned
        assert orphan.poll() is None
        assert kb.get_task(board, tid).status == "running"
        assert kb.get_task(board, tid).claim_lock == lock
        assert not [e for e in kb.list_events(board, tid) if e.kind == "reclaimed"]
    finally:
        orphan.kill()
        orphan.wait(timeout=10)


def test_previous_run_heartbeat_does_not_hold_dead_claimer(board, monkeypatch):
    dead_pid = 999995
    tid = _claim(board, f"{kb._host_prefix()}{dead_pid}")
    run_id = kb._current_run_id(board, tid)
    assert run_id is not None
    with kb.write_txn(board):
        kb._append_event(board, tid, "heartbeat", run_id=run_id - 1)
    def dead(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, "Process", dead)
    assert kb.reclaim_task(board, tid) is True


@pytest.mark.parametrize("path", ["manual", "ttl", "stale"])
def test_orphan_before_first_heartbeat_keeps_claim(board, monkeypatch, path):
    """A detached orphan that has not heartbeated YET is still a worker.

    Claim is 60 s old: past a 45 s sweep grace, inside the launch bound.
    """
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    lock = f"{kb._host_prefix()}{_dead_pid()}"
    tid = _claim(board, lock, age=60)
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        assert _release(board, tid, path) is False
        spawned = []
        kbd.dispatch_once(board, spawn_fn=lambda task, workspace, board=None: spawned.append(task.id) or 424242)
        assert tid not in spawned
        assert orphan.poll() is None
        assert kb.get_task(board, tid).claim_lock == lock
    finally:
        orphan.kill()
        orphan.wait(timeout=10)


@pytest.mark.parametrize("path", ["manual", "ttl", "stale"])
def test_orphan_that_heartbeated_then_died_is_released_once_stale(board, path):
    lock = f"{kb._host_prefix()}{_dead_pid()}"
    stale = kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
    tid = _claim(board, lock, age=stale + 10)
    assert kbd.heartbeat_worker(board, tid, note="alive once")
    with kb.write_txn(board):
        board.execute(
            "UPDATE task_events SET created_at=? WHERE task_id=? AND kind='heartbeat'",
            (int(time.time()) - stale - 5, tid),
        )
    assert _release(board, tid, path) is True
    assert kb.get_task(board, tid).claim_lock is None


def test_fresh_heartbeat_holds_dead_claimer(board):
    lock = f"{kb._host_prefix()}{_dead_pid()}"
    tid = _claim(board, lock)
    assert kbd.heartbeat_worker(board, tid, note="recent")
    assert kb.reclaim_task(board, tid) is False
    assert kb.get_task(board, tid).claim_lock == lock


def test_missing_run_context_fails_closed():
    info = kbd._terminate_reclaimed_worker(None, f"{kb._host_prefix()}{_dead_pid()}")
    assert kbd._worker_survived_termination(info) is True
    assert info["unstamped_worker_check"] == "skipped_no_run_context"
