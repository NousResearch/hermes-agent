"""Atomic operator-only execution controls for stopped Kanban tasks.

No lifecycle transition lives here: editing budgets must not restart a task or
rewrite the limits/clocks of an earlier run.
"""
from __future__ import annotations

import argparse
import os

CONTROL_FIELDS = ("goal_mode", "goal_max_turns", "max_retries", "max_runtime_seconds")
STOPPED_STATUSES = frozenset({"blocked", "todo", "triage", "scheduled", "done"})
MAX_CONTROL_INT = 2**63 - 1


def validate_controls(values: dict) -> dict:
    """Validate without coercion. Omitted keys stay omitted; nullable limits clear."""
    unknown = values.keys() - set(CONTROL_FIELDS)
    if unknown:
        raise ValueError(f"unknown execution controls: {', '.join(sorted(unknown))}")
    for field, value in values.items():
        if field == "goal_mode":
            if type(value) is not bool:
                raise ValueError("goal_mode must be a boolean (not null)")
        elif value is not None and (type(value) is not int or not 1 <= value <= MAX_CONTROL_INT):
            raise ValueError(f"{field} must be a positive integer or null")
    return dict(values)


def assert_operator_context() -> None:
    """A worker, child or non-owned execution cannot change its own budgets."""
    from agent.delegation_context import (
        is_delegated_child_process_context, is_dispatcher_owned_worker_context,
    )

    if (os.environ.get("HERMES_KANBAN_TASK") or is_delegated_child_process_context()
            or not is_dispatcher_owned_worker_context()):
        raise PermissionError("execution controls require an operator main context, not a worker or delegate")


def edit_execution_controls(conn, task_id: str, values: dict, *, board=None) -> bool:
    from hermes_cli import kanban_db as kb

    assert_operator_context()
    values = validate_controls(values)
    if not values:
        return False
    # write_txn retains the native path fence and takes BEGIN IMMEDIATE BEFORE
    # eligibility is read; concurrent claims cannot slip between read and write.
    with kb.write_txn(conn):
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return False
        active_run = conn.execute(
            "SELECT 1 FROM task_runs WHERE task_id = ? AND ended_at IS NULL LIMIT 1", (task_id,)
        ).fetchone()
        if (row["status"] not in STOPPED_STATUSES or active_run is not None
                or any(row[key] is not None for key in
                       ("claim_lock", "claim_expires", "worker_pid", "worker_started_at", "current_run_id"))):
            raise ValueError("execution controls require a stopped task without a claim or active run")
        old = {key: bool(row[key]) if key == "goal_mode" else row[key] for key in CONTROL_FIELDS}
        new = {**old, **values}
        conn.execute(
            f"UPDATE tasks SET {', '.join(f'{key} = ?' for key in CONTROL_FIELDS)} WHERE id = ?",
            (*[new[key] for key in CONTROL_FIELDS], task_id),
        )
        kb._append_event(conn, task_id, "edited", {
            "fields": list(values), "execution_controls": {"old": old, "new": new},
            "actor": kb._hook_profile_name(),
        })
    kb.notify_task_updated(conn, task_id, list(values), board=board)
    return True


def parse_control_bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def parse_control_limit(value: str) -> int | None:
    if value == "clear":
        return None
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or clear") from exc
    if not 1 <= number <= MAX_CONTROL_INT:
        raise argparse.ArgumentTypeError("expected a positive SQLite integer or clear")
    return number


def cli_edit_controls(args) -> int:
    """Narrow branch of the existing edit command; do not combine mutation verbs."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli.kanban import _ok_or_err
    from hermes_cli.kanban_output import _err

    if any(getattr(args, key, None) is not None for key in
           ("title", "body", "priority", "result", "summary", "metadata")):
        return _err("execution controls cannot be combined with other edits", 2)
    values = {key: getattr(args, key) for key in CONTROL_FIELDS if hasattr(args, key)}
    try:
        assert_operator_context()
        with kbc.connect_closing() as conn:
            ok = kb.edit_task(conn, args.task_id, **values)
    except (ValueError, PermissionError) as exc:
        return _err(str(exc), 2)
    return _ok_or_err(ok, f"unknown task {args.task_id}", f"Edited {args.task_id}")
