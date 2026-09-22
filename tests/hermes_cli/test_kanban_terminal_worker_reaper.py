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
        tid, run_id = _completed_card_with_worker(conn, stranger)
        # PID reuse: the recorded fingerprint belongs to a process that no longer exists.
        # A valid different legacy start time, not arithmetic on a boot-witness
        # string (SQLite coerces that to a malformed negative timestamp).
        conn.execute("UPDATE task_runs SET worker_started_at = 1 WHERE id=?", (run_id,))
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
