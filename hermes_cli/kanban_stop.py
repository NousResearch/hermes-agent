"""Operator stop: park a card, then confirm its retained workers have exited."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import sqlite3


@dataclass(frozen=True)
class StopResult:
    task_id: str
    status: str
    stopped: bool
    pending_runs: tuple[int, ...]


def stop_task(conn: sqlite3.Connection, task_id: str, reason: str) -> StopResult:
    """Use the terminal-worker reaper without its self-handoff grace period.

    Blocking prevents dispatch while termination runs outside the write transaction.
    Unconfirmed workers retain their run evidence; repeating stop retries those runs.
    Never archive the workspace or unblock dependent cards as a side effect of stop.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _reap_terminal_worker_row
    from hermes_cli.kanban_db_connect import _main_db_file

    kb._assert_not_delegated_child_mutation(_main_db_file(conn))
    if os.environ.get("HERMES_KANBAN_TASK"):
        raise ValueError("kanban stop is operator-only; workers must hand off with kanban block")

    task = kb.get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task: {task_id}")
    if task.status in {"running", "ready"}:
        run_id = kb._current_run_id(conn, task_id)
        if not kb.block_task(conn, task_id, reason=reason, kind="capability", expected_run_id=run_id):
            raise ValueError(f"task changed while stopping: {task_id}; retry stop")
    elif task.status not in {"blocked", "triage"}:
        raise ValueError(f"cannot stop a {task.status} task: {task_id}")

    rows = conn.execute(
        "SELECT id, task_id, worker_pid, worker_started_at, claim_lock FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL AND worker_pid IS NOT NULL",
        (task_id,),
    ).fetchall()
    for row in rows:
        if row["worker_started_at"] is not None:
            _reap_terminal_worker_row(conn, row, kb._host_prefix(), None, [])
    pending = tuple(row["id"] for row in conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND worker_pid IS NOT NULL ORDER BY id",
        (task_id,),
    ))
    landed = kb.get_task(conn, task_id)
    stopped = not pending and landed.worker_pid is None and landed.status in {"blocked", "triage"}
    return StopResult(task_id, landed.status, stopped, pending)


def cmd_stop(args) -> int:
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        result = stop_task(conn, args.task_id, args.reason or "operator requested stop")
    if args.json:
        print(json.dumps(asdict(result)))
    elif result.stopped:
        print(f"Stopped {result.task_id}; task remains {result.status}")
    else:
        print(f"Stop unconfirmed for {result.task_id}; retained runs: {result.pending_runs}; retry stop")
    return 0 if result.stopped else 1
