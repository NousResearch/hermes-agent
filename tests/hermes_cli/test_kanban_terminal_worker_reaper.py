"""Invariant: a worker that outlives its own terminal transition is still
reachable and gets reaped by the dispatcher (issue #111791).

``complete_task`` wipes ``tasks.worker_pid`` and the ``running``-only reclaim
sweeps never look at a ``done`` card again, so a worker that called
``kanban_complete`` and then hung (holding deleted ``state.db`` sidecar fds)
was invisible to every command. The closed ``task_runs`` row now keeps the pid
and its spawn fingerprint, and ``reap_terminal_workers`` ends such a worker on
the next tick — never a recycled PID, never a legacy row without a fingerprint.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _completed_card_with_worker(conn, proc, *, ended_ago: int = 600) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, proc.pid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    # Default: the run closed long enough ago that the reaper's grace window has passed.
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def test_worker_alive_after_completion_is_reaped_on_dispatch_tick(conn):
    proc = _sleeper()
    try:
        tid, run_id = _completed_card_with_worker(conn, proc)
        # The evidence the running-only sweeps lost lives on the closed run row.
        run = conn.execute("SELECT worker_pid, worker_started_at, ended_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["ended_at"] is not None and run["worker_pid"] == proc.pid and run["worker_started_at"] is not None

        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

        assert result.reaped_terminal_workers == [tid]
        assert proc.wait(timeout=10) is not None
        run = conn.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] is None and run["worker_started_at"] is None
        kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
        assert "terminal_worker_reaped" in kinds
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_recycled_pid_and_legacy_row_are_never_signalled(conn):
    stranger = _sleeper()
    legacy = _sleeper()
    try:
        _tid, run_id = _completed_card_with_worker(conn, stranger)
        # PID reuse: the recorded fingerprint belongs to a process that no longer exists.
        conn.execute("UPDATE task_runs SET worker_started_at = worker_started_at - 1000000 WHERE id=?", (run_id,))
        _, legacy_run = _completed_card_with_worker(conn, legacy)
        conn.execute("UPDATE task_runs SET worker_started_at = NULL WHERE id=?", (legacy_run,))

        assert kbd.reap_terminal_workers(conn) == []

        assert stranger.poll() is None and legacy.poll() is None
        # Stale evidence is dropped so the row is not rescanned; the legacy row is left alone.
        assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()["worker_pid"] is None
        assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (legacy_run,)).fetchone()["worker_pid"] == legacy.pid
    finally:
        for p in (stranger, legacy):
            p.kill()
            p.wait()


def test_fresh_terminal_run_is_left_alone_until_grace_passes(conn):
    """A worker is still alive for a moment after its own kanban_complete returns
    (final turn, session persistence): a just-closed run is not signalled."""
    proc = _sleeper()
    signals = []

    def signal_fn(pid, sig):
        signals.append((pid, sig))
        os.kill(pid, sig)

    try:
        tid, run_id = _completed_card_with_worker(conn, proc, ended_ago=0)

        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == []

        assert signals == [] and proc.poll() is None
        run = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] == proc.pid  # evidence kept for a later tick
        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
            (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id),
        )
        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == [tid]
        assert [pid for pid, _ in signals] == [proc.pid]
    finally:
        proc.kill()
        proc.wait()


def test_one_failing_row_does_not_abort_the_sweep(conn):
    """A signal failure on one run is logged and skipped; the other rows are still reaped."""
    broken, healthy = _sleeper(), _sleeper()
    try:
        _completed_card_with_worker(conn, broken)
        tid, _ = _completed_card_with_worker(conn, healthy)

        def signal_fn(pid, sig):
            if pid == broken.pid:
                raise RuntimeError("boom")
            healthy.kill()

        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == [tid]
    finally:
        for p in (broken, healthy):
            p.kill()
            p.wait()


# --- the workspace a dead worker left behind -----------------------------------
#
# Completion cleanup runs inside the worker, which the dispatcher started with
# the workspace as its cwd. ``_force_rmtree`` moves the worker out before the
# removal, but a run can leave another child process holding the directory and
# Windows keeps refusing the ``rmdir`` until that child is gone. Once the
# dispatcher has confirmed the worker itself is gone it runs the cleanup again.


def _completed_card_with_leftover_workspace(conn, pid: int) -> tuple[str, int, Path]:
    """A finished scratch card whose workspace is still on disk — the empty
    shell Windows leaves when the worker's own ``rmdir`` was refused."""
    tid, run_id = _completed_card_with_worker(conn, type("Proc", (), {"pid": pid})())
    ws = kb.workspaces_root() / tid
    ws.mkdir(parents=True)
    (ws / "scratch.txt").write_text("left over", encoding="utf-8")
    conn.execute(
        "UPDATE tasks SET workspace_kind='scratch', workspace_path=? WHERE id=?",
        (str(ws), tid),
    )
    return tid, run_id, ws


def _exited_pid() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    proc.wait(timeout=60)
    return proc.pid


def test_leftover_workspace_of_an_exited_worker_is_removed_on_the_sweep(conn):
    tid, run_id, ws = _completed_card_with_leftover_workspace(conn, _exited_pid())

    # Nothing to terminate: the pid is gone, so the row is not a reap — only
    # its stale evidence is dropped, and the shell with it.
    assert kbd.reap_terminal_workers(conn) == []
    run = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert run["worker_pid"] is None
    assert not ws.exists()
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
    assert "terminal_worker_reaped" not in kinds


def test_leftover_workspace_waits_for_a_live_worker(conn, monkeypatch):
    proc = _sleeper()
    try:
        _tid, run_id, ws = _completed_card_with_leftover_workspace(conn, proc.pid)
        # Still running and refusing to die: nothing about this card may move.
        monkeypatch.setattr(kbd, "_worker_alive", lambda pid, started_at=None: True)
        monkeypatch.setattr(kbd, "_terminate_reclaimed_worker", lambda *a, **k: {"terminated": False})

        assert kbd.reap_terminal_workers(conn) == []

        assert ws.is_dir() and (ws / "scratch.txt").exists()
        run = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] == proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_leftover_workspace_of_a_respawned_card_is_kept(conn):
    tid, _run_id, ws = _completed_card_with_leftover_workspace(conn, _exited_pid())
    # Requeued after a failed/cancelled run: the next worker reuses this directory.
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))

    kbd.reap_terminal_workers(conn)

    assert ws.is_dir() and (ws / "scratch.txt").exists()


def test_one_failed_leftover_cleanup_does_not_abort_the_sweep(conn, monkeypatch):
    first, second = _exited_pid(), _exited_pid()
    broken_tid, _run, _broken_ws = _completed_card_with_leftover_workspace(conn, first)
    tid, _run, second_ws = _completed_card_with_leftover_workspace(conn, second)
    real = kbd._kbw._retry_workspace_cleanup_after_exit
    calls = []

    def flaky(c, task_id):
        calls.append(task_id)
        if task_id == broken_tid:
            raise RuntimeError("boom")
        return real(c, task_id)

    monkeypatch.setattr(kbd._kbw, "_retry_workspace_cleanup_after_exit", flaky)

    assert kbd.reap_terminal_workers(conn) == []
    assert sorted(calls) == sorted([broken_tid, tid])
    assert not second_ws.exists()


def test_retry_reports_a_workspace_that_still_refuses_to_go(conn, monkeypatch, caplog):
    tid, _run, ws = _completed_card_with_leftover_workspace(conn, _exited_pid())
    monkeypatch.setattr(kbd._kbw, "_cleanup_workspace", lambda c, task_id: None)  # refused again

    with caplog.at_level(logging.WARNING):
        assert kbd._kbw._retry_workspace_cleanup_after_exit(conn, tid) is False

    assert ws.is_dir()
    assert any("still present" in r.getMessage() for r in caplog.records)


def test_retry_is_a_no_op_once_the_workspace_is_gone(conn):
    tid, _run, _ws = _completed_card_with_leftover_workspace(conn, _exited_pid())
    shutil.rmtree(_ws)

    assert kbd._kbw._retry_workspace_cleanup_after_exit(conn, tid) is False

    assert not _ws.exists()
