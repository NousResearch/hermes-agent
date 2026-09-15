"""Regression coverage for terminal Kanban workers that survive their task."""

from __future__ import annotations

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
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_dispatch_reaps_a_completed_worker_but_not_a_reused_pid(kanban_home, monkeypatch):
    """Terminal runs retain a start-time-fenced identity for dispatcher recovery."""
    sidecar = kanban_home / "state.db-wal"
    sidecar.write_bytes(b"stale sidecar")
    child = subprocess.Popen(
        [
            sys.executable, "-c",
            "import os, sys, time; fd = os.open(sys.argv[1], os.O_RDONLY); "
            "os.unlink(sys.argv[1]); print('held', flush=True); time.sleep(30)",
            str(sidecar),
        ],
        stdout=subprocess.PIPE, text=True,
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "held"
    assert not sidecar.exists()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="sidecar holder", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        kbd._set_worker_pid(conn, tid, child.pid)
        assert kb.complete_task(conn, tid, result="finished")
        run = kb.latest_run(conn, tid)
        assert run is not None and run.worker_pid == child.pid

        # A mismatched fingerprint is PID reuse: fail closed and leave it alone.
        monkeypatch.setattr(kbd, "_worker_start_time", lambda _pid: 0)
        assert kbd.reap_terminal_workers(conn) == []
        assert child.poll() is None

        monkeypatch.setattr(kbd, "_worker_start_time", lambda _pid: conn.execute(
            "SELECT worker_started_at FROM task_runs WHERE id = ?", (run.id,)
        ).fetchone()[0])
        assert kbd.dispatch_once(conn, spawn_fn=lambda *_args: None).reaped_terminal_workers == [tid]
        child.wait(timeout=5)
        assert conn.execute("SELECT worker_pid FROM task_runs WHERE id = ?", (run.id,)).fetchone()[0] is None
    finally:
        conn.close()
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
