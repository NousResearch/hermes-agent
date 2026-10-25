"""Kanban review change request state transition (extracted from the DB facade).

Keep the review transaction, durable event writes, and operator action together.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from hermes_cli.kanban_db import (
    _append_event,
    _canonical_assignee,
    _end_run,
    _json_dict,
    _landing_status_after_parents,
    _latest_event,
    _nonblank_str,
    _row_get,
    redact_review_value,
    write_txn,
)


def request_changes(
    conn: sqlite3.Connection, task_id: str, *, reason: str, expected_run_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Close an active reviewer run (claimed from ``review``) and hand the task
    back to the implementer from the latest ``review_requested`` event, parent
    gating reapplied. Returns ``(ok, implementer | reason)``."""
    reason = str(redact_review_value(reason or "")).strip()
    if not reason:
        return False, "reason is required"

    with write_txn(conn):
        task_row = conn.execute(
            "SELECT status, assignee, current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task_row is None:
            return False, "task not found"
        current_run_id = task_row["current_run_id"]
        if task_row["status"] != "running" or current_run_id is None:
            return False, "task is not in an active review run"
        if expected_run_id is not None and int(current_run_id) != int(expected_run_id):
            return False, "run_id mismatch"

        claimed_event = _latest_event(conn, task_id, "claimed", current_run_id)
        claimed_payload = _json_dict(_row_get(claimed_event, "payload"))
        if claimed_payload.get("source_status") != "review":
            return False, "active run was not claimed from review"

        requested_event = _latest_event(conn, task_id, "review_requested")
        if requested_event is None:
            return False, "no prior review_requested event"
        implementer = _nonblank_str(_json_dict(requested_event["payload"]).get("implementer"))
        if implementer is None:
            return False, "review handoff has no valid implementer provenance"
        reviewer = _canonical_assignee(_nonblank_str(task_row["assignee"]))

        from hermes_cli.kanban_review_rework_budget import max_review_rejections

        rework_limit = max_review_rejections()
        previous_rejections = 0
        if rework_limit is not None:
            previous_rejections = conn.execute(
                "SELECT COUNT(*) FROM task_events "
                "WHERE task_id = ? AND kind = 'changes_requested'",
                (task_id,),
            ).fetchone()[0]
        owner_action_required = (
            rework_limit is not None
            and previous_rejections + 1 >= rework_limit
        )
        # 'blocked' is an existing operator-recoverable state; unlike 'triage',
        # it does not depend on any unmerged triage-promotion features.
        new_status = (
            "blocked" if owner_action_required
            else _landing_status_after_parents(conn, task_id)
        )
        # consecutive_failures deliberately PRESERVED: a review transition is
        # not evidence the pathology cleared; only complete_task resets it.
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL, worker_started_at = NULL,
                   block_kind = CASE WHEN ? THEN 'needs_input' ELSE block_kind END
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, int(owner_action_required), task_id, int(current_run_id)),
        )
        if cur.rowcount != 1:
            return False, "task changed during review handoff"
        run_id = _end_run(
            conn, task_id, outcome="changes_requested", status=new_status, summary=reason,
        )
        _append_event(
            conn,
            task_id,
            "changes_requested",
            {
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
                "status": new_status,
            },
            run_id=run_id,
        )
        if owner_action_required:
            _append_event(
                conn, task_id, "blocked",
                {
                    "kind": "needs_input",
                    "reason": (
                        f"Review rework limit reached "
                        f"({previous_rejections + 1}/{rework_limit}); "
                        "operator decision required before further review"
                    ),
                },
                run_id=run_id,
            )
    return True, implementer
