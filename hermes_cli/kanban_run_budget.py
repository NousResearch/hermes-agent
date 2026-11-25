"""Absolute per-task run-ledger budgets, independent of failure retry policy.

Every allocated run ID spends a slot, including neutral exits and failed launch
attempts. Reserving at claim (before spawning) is deliberately conservative:
crashing between allocation and spawn must never refund permission implicitly.
Historical/synthetic ledger rows also count; history is never reset or deleted.
"""
from __future__ import annotations

import json
import sqlite3


def validate_max_worker_runs(value):
    """NULL is unlimited; zero explicitly forbids further claims."""
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("max_worker_runs must be a non-negative integer or null (unlimited)")
    return value


def budget_state(conn: sqlite3.Connection, task_id: str):
    return conn.execute(
        "SELECT max_worker_runs, status, claim_lock, "
        "(SELECT COUNT(*) FROM task_runs WHERE task_id = tasks.id) AS worker_runs_used "
        "FROM tasks WHERE id = ? AND max_worker_runs IS NOT NULL", (task_id,),
    ).fetchone()


def exhausted(row) -> bool:
    return bool(row is not None and row["max_worker_runs"] is not None
                and row["worker_runs_used"] >= row["max_worker_runs"])


def pending_budget_exhaustion(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    """Latest exhaustion not acknowledged by an explicit owner unblock.

    Kept separate from ``blocked`` so an existing human/provider blocker and
    its reason are never replaced by a second blocker when both limits trip.
    """
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'run_budget_exhausted' "
        "AND id > COALESCE((SELECT MAX(id) FROM task_events "
        "WHERE task_id = ? AND kind = 'unblocked'), 0) ORDER BY id DESC LIMIT 1",
        (task_id, task_id),
    ).fetchone()


def enforce_run_budget(conn: sqlite3.Connection, task_id: str) -> bool:
    """Park an exhausted runnable task INSIDE its mutation/claim transaction.

    Called at the common event-write seam, so rate limits, auth, crashes,
    timeouts, stale claims, orphan recovery, review, manual requeues and future
    lifecycle paths all obey the same invariant. Never interrupts a running
    worker or unblocks a parked task. Already blocked tasks receive the budget
    diagnostic without overwriting their blocker. The claim gate also checks
    independently.
    """
    row = budget_state(conn, task_id)
    if not exhausted(row):
        return False
    if row["status"] not in ("ready", "review", "blocked") or row["claim_lock"] is not None:
        return True
    parking = row["status"] != "blocked"
    if not parking:
        prior = pending_budget_exhaustion(conn, task_id)
        if prior is not None:
            prior_payload = json.loads(prior["payload"])
            if (prior_payload.get("max_worker_runs") == row["max_worker_runs"]
                    and prior_payload.get("worker_runs_used") == row["worker_runs_used"]):
                return True  # comments/ticks must not duplicate the diagnostic
    from hermes_cli import kanban_db as kb
    resume_status = row["status"] if parking else kb._resume_status_from_events(conn, task_id)
    reason = (f"run-budget-exhausted: {row['worker_runs_used']}/{row['max_worker_runs']} "
              "worker run IDs used; owner must raise or clear max_worker_runs, then unblock")
    if parking:
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
    # Raw inserts avoid recursively invoking the common event seam.
    import time
    payload = json.dumps({"reason": reason, "sticky": True,
                          "resume_status": resume_status,
                          "max_worker_runs": row["max_worker_runs"],
                          "worker_runs_used": row["worker_runs_used"]})
    for kind in (("run_budget_exhausted", "blocked") if parking else ("run_budget_exhausted",)):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (task_id, kind, payload, int(time.time())),
        )
    return True


def assert_budget_edit_allowed():
    """An executing worker cannot grant itself (or another card) more runs."""
    from agent.delegation_context import owned_kanban_task

    if owned_kanban_task():
        raise ValueError("max_worker_runs is owner-managed; workers cannot edit run budgets")
