"""Dispatcher failure budgets survive crash/timeout reaping (PR #97991).

Ported to the split dispatcher; test both strict and lenient limits and the
per-task override through real dispatcher ticks against isolated SQLite state.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.mark.parametrize("failure_kind", ["crash", "timeout"])
@pytest.mark.parametrize(
    "dispatcher_limit,task_limit", [(1, None), (3, None), (3, 1), (1, 3)]
)
def test_reaper_honours_effective_failure_budget(
    tmp_path,
    monkeypatch,
    failure_kind,
    dispatcher_limit,
    task_limit,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Timeouts must reach the timeout reaper, not the earlier crash sweep.
    monkeypatch.setenv(
        "HERMES_KANBAN_CRASH_GRACE_SECONDS",
        "86400" if failure_kind == "timeout" else "0",
    )
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    # Fake PIDs must never cause real signals, including on the red baseline.
    signals = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    kbc.init_db()
    effective_limit = task_limit if task_limit is not None else dispatcher_limit
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="bounded reaper",
            assignee="worker",
            max_runtime_seconds=1 if failure_kind == "timeout" else None,
            max_retries=task_limit,
        )
        for attempt in range(1, effective_limit + 1):
            assert kb.claim_task(conn, tid)
            pid = 991_100 + attempt
            kbd._set_worker_pid(conn, tid, pid)
            if failure_kind == "crash":
                kbd._record_worker_exit(pid, 256)
            old = int(time.time()) - 30
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
                conn.execute(
                    "UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                    (old, tid),
                )
            result = kbd.dispatch_once(
                conn, max_spawn=0, failure_limit=dispatcher_limit
            )
            assert tid in (
                result.crashed if failure_kind == "crash" else result.timed_out
            )
            assert result.spawned == []
            task = kb.get_task(conn, tid)
            assert task.consecutive_failures == attempt
            gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
            if attempt < effective_limit:
                assert task.status == "ready"
                assert gave_up == []
            else:
                assert task.status == "blocked"
                assert len(gave_up) == 1
                assert gave_up[0].payload["effective_limit"] == effective_limit
                assert gave_up[0].payload["limit_source"] == (
                    "task" if task_limit is not None else "dispatcher"
                )
        if failure_kind == "timeout":
            assert signals == [
                (991_100 + n, signal.SIGTERM) for n in range(1, effective_limit + 1)
            ]
        else:
            assert signals == []
