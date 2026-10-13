"""``block_task`` helpers: status routing for a block, and halting the worker a
supervisor blocked.

Split out of ``hermes_cli.kanban_db``; origin-resident names are reached
late-bound via ``_kb`` so monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def _route_block(
    kind: Optional[str], reason: Optional[str], source_status: str, *,
    prev_kind: Optional[str], prev_recurrences: int,
) -> tuple[str, str, str, tuple, dict]:
    """``(new_status, event_kind, set_sql, params, payload)`` for :func:`block_task`.

    ``dependency`` never enters the human ``blocked`` bucket: it waits in
    ``todo`` for ``recompute_ready``, so a cron never sees a dependency-wait
    as something to "unblock". Callers that pass ``dependency`` with no
    incomplete parent are re-kinded to ``needs_input`` before this runs
    (see :func:`block_task`). Every other kind counts unblock-loop
    recurrences: block_task only fires from running/ready (AFTER an unblock
    returned the task to the pool), so a stored ``block_kind`` equal to the
    incoming one means blocked -> unblocked -> re-block for the same cause
    (un-typed None compares equal to a prior un-typed block). At
    ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
    """
    payload = {"reason": reason, "kind": kind, "source_status": source_status}
    if kind == "dependency":
        return "todo", "dependency_wait", "block_kind    = ?", (kind,), payload
    recurrences = prev_recurrences + 1 if prev_kind == kind else 1
    set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
    payload = {"reason": reason, "kind": kind, "recurrences": recurrences, "source_status": source_status}
    if recurrences >= _kb.BLOCK_RECURRENCE_LIMIT:
        payload["limit"] = _kb.BLOCK_RECURRENCE_LIMIT
        return "triage", "block_loop_detected", set_sql, (kind, recurrences), payload
    return "blocked", "blocked", set_sql, (kind, recurrences), payload


def halt_blocked_worker(conn: sqlite3.Connection, run_id: Optional[int]) -> bool:
    """End the worker of a run that a supervisor (not the worker) just blocked.

    ``reap_terminal_workers`` waits ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` after a
    run ends because a worker is still finishing its own terminal call (final
    turn, session persistence). A block filed by a human or an integration (a
    Linear Stop: "must not perform any further actions") leaves no such work to
    wait for, so the same per-run reap runs now: same host check, same spawn
    fingerprint proof, SIGTERM then SIGKILL, same ``terminal_worker_reaped``
    event. A legacy run with no fingerprint, an UNVERIFIED spawn and a recycled
    PID are never signalled; a worker that survives keeps its evidence for the
    reaper's next tick. Call after the block's txn has committed: the reap polls
    for up to ~5 s and opens its own write txn. True when a live worker was ended.
    """
    if run_id is None:
        return False
    row = conn.execute(
        "SELECT id, task_id, worker_pid, worker_started_at, claim_lock FROM task_runs "
        "WHERE id = ? AND ended_at IS NOT NULL "
        "AND worker_pid IS NOT NULL AND worker_started_at IS NOT NULL",
        (int(run_id),),
    ).fetchone()
    if row is None:
        return False
    from hermes_cli.kanban_db_dispatch import _reap_terminal_worker_row

    reaped: list[str] = []
    try:
        _reap_terminal_worker_row(conn, row, _kb._host_prefix(), None, reaped)
    except Exception:
        # The block has committed; a failed halt leaves the reaper's grace path in charge.
        logger.exception("kanban: halting blocked worker failed for run %s (task %s)", row["id"], row["task_id"])
        return False
    return bool(reaped)


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
