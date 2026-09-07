"""Read-only, timestamped views of the existing per-attempt runtime limit.

These snapshots neither renew claims nor authorize more work. The dispatcher
still owns hard enforcement; other limits apply even without a per-task cap.
"""

from dataclasses import dataclass, replace
import sqlite3
import time


@dataclass(frozen=True)
class RuntimeBudget:
    task_id: str
    run_id: int | None
    observed_at: int
    state: str
    limit_seconds: int | None = None
    started_at: int | None = None
    start_basis: str | None = None
    deadline_at: int | None = None
    elapsed_seconds: int | None = None
    remaining_seconds: int | None = None
    reason: str = ""


def runtime_budget_snapshot(
    conn: sqlite3.Connection, task_id: str, *, observed_at: int,
) -> RuntimeBudget:
    """Use one joined read and the caller's clock, never a new policy/default.

    Match enforcement's current run start, task-start fallback and task cap.
    A stale/cross-task run or invalid clock is reported unknown, not silently
    presented as permission to continue. No board or runtime state is written.
    """
    row = conn.execute(
        "SELECT t.status, t.current_run_id, t.max_runtime_seconds, t.started_at, "
        "r.task_id, r.started_at, r.ended_at FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id WHERE t.id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown task {task_id}")
    status, run_id, limit, task_start, run_task, run_start, run_end = row
    result = RuntimeBudget(task_id, run_id, observed_at, "unknown")
    if status != "running":
        return replace(result, state="not_running", reason="no running attempt")
    if run_task is not None and (run_task != task_id or run_end is not None):
        return replace(result, reason="current run binding is inconsistent")
    if limit is None:
        return replace(result, state="unbounded", reason="no per-task runtime cap; other limits still apply")
    start = run_start if run_start is not None else task_start
    if start is None:
        return replace(result, reason="attempt start is unavailable")
    try:
        limit, start = int(limit), int(start)
    except (TypeError, ValueError, OverflowError):
        return replace(result, reason="invalid runtime cap or start")
    basis = "current_run" if run_start is not None else "task_fallback"
    result = replace(result, limit_seconds=limit, started_at=start, start_basis=basis)
    if start > observed_at:
        return replace(result, reason="observation clock precedes attempt start")
    deadline = start + limit
    return replace(
        result, state="active" if observed_at < deadline else "exhausted",
        deadline_at=deadline, elapsed_seconds=observed_at - start,
        remaining_seconds=max(0, deadline - observed_at),
    )


def _utc(timestamp: int) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
    except (OverflowError, OSError, ValueError):
        return f"{timestamp} Unix seconds"


def format_runtime_budget(snapshot: RuntimeBudget) -> str:
    """A snapshot label prevents old context from posing as a live countdown."""
    prefix = (f"Runtime budget snapshot: observed={_utc(snapshot.observed_at)}; "
              f"run={snapshot.run_id}; state={snapshot.state}")
    if snapshot.remaining_seconds is None:
        return f"{prefix}; {snapshot.reason}."
    return (f"{prefix}; {snapshot.remaining_seconds}s remaining; "
            f"deadline={_utc(snapshot.deadline_at)}; elapsed={snapshot.elapsed_seconds}s; "
            f"cap={snapshot.limit_seconds}s; start_basis={snapshot.start_basis}.")
