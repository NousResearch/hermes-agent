"""Review-lane kernel primitives (moved out of the ``kanban_db`` and
``kanban_db_dispatch`` facades to keep both under their line ratchet).

Late-binds the ``kanban_db`` facade (same pattern as ``kanban_db_boards``):
these functions run inside a caller-owned txn and reach shared helpers via
``_kb`` at call time.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional, Tuple

# Late-bound origin namespace; imported LAST so this module is fully populated
# before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb


def _prior_reviewer(conn: sqlite3.Connection, task_id: str):
    """Reviewer recorded by the latest ``changes_requested`` run's event.
    ``None`` = first review (no such run); ``False`` = a run exists but its
    provenance is missing/malformed."""
    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if changes_run is None:
        return None
    changes_event = _kb._latest_event(conn, task_id, "changes_requested", changes_run["id"])
    reviewer = _kb._json_dict(_kb._row_get(changes_event, "payload")).get("reviewer")
    return reviewer if isinstance(reviewer, str) and reviewer.strip() else False


def request_changes(
    conn: sqlite3.Connection, task_id: str, *, reason: str, expected_run_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Close an active reviewer run (claimed from ``review``) and hand the task
    back to the implementer from the latest ``review_requested`` event, parent
    gating reapplied. Returns ``(ok, implementer | reason)``."""
    reason = str(_kb.redact_review_value(reason or "")).strip()
    if not reason:
        return False, "reason is required"

    with _kb.write_txn(conn):
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

        claimed_event = _kb._latest_event(conn, task_id, "claimed", current_run_id)
        claimed_payload = _kb._json_dict(_kb._row_get(claimed_event, "payload"))
        if claimed_payload.get("source_status") != "review":
            return False, "active run was not claimed from review"

        requested_event = _kb._latest_event(conn, task_id, "review_requested")
        if requested_event is None:
            return False, "no prior review_requested event"
        implementer = _kb._nonblank_str(_kb._json_dict(requested_event["payload"]).get("implementer"))
        if implementer is None:
            return False, "review handoff has no valid implementer provenance"
        reviewer = _kb._canonical_assignee(_kb._nonblank_str(task_row["assignee"]))

        new_status = _kb._landing_status_after_parents(conn, task_id)
        # consecutive_failures deliberately PRESERVED: a review transition is
        # not evidence the pathology cleared; only complete_task resets it.
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL, worker_started_at = NULL
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, task_id, int(current_run_id)),
        )
        if cur.rowcount != 1:
            return False, "task changed during review handoff"
        run_id = _kb._end_run(
            conn, task_id, outcome="changes_requested", status=new_status, summary=reason,
        )
        _kb._append_event(
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
    return True, implementer


def _review_lane_self_review_reason(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[str]:
    """``"self_review:<implementer>"`` when the review row's assignee (the
    profile the dispatcher would spawn) is the same profile that performed the
    latest implementation, else ``None``.

    Implementer provenance is the ``implementer`` field of the latest
    ``review_requested`` event — the same field the kernel already uses to
    route ``request_changes`` handoffs. When that event is missing or carries
    no valid implementer, the fallback is the profile that owns the card's
    latest terminal implementer run (completed / review_requested). Runs that
    ended in a crash/reclaim are deliberately NOT consulted — a reviewer that
    crashed mid-review owns its last run but is not an implementer, and must
    stay re-spawnable from the review lane. A card with no run and no handoff
    event (operator-created review row) has no implementer to conflict with
    and always passes. The guard is an escape hatch, not a routing decision:
    an operator reassigns an independent reviewer profile and the next tick
    dispatches normally.
    """
    row = conn.execute(
        "SELECT assignee FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None:
        return None
    assignee = _kb._canonical_assignee(row["assignee"])
    if not assignee:
        return None

    implementer: Optional[str] = None
    ev = _kb._latest_event(conn, task_id, "review_requested")
    payload = _kb._json_dict(_kb._row_get(ev, "payload")) if ev is not None else {}
    raw = payload.get("implementer")
    if isinstance(raw, str) and raw.strip():
        implementer = _kb._canonical_assignee(raw)
    if implementer is None:
        latest_run = conn.execute(
            "SELECT profile FROM task_runs WHERE task_id = ? AND profile IS NOT NULL "
            "AND outcome IN ('completed', 'review_requested') "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if latest_run is not None:
            implementer = _kb._canonical_assignee(latest_run["profile"])
    if implementer and implementer == assignee:
        return (
            f"self_review:{implementer} — the review lane would spawn the "
            "latest implementer's own profile as its reviewer (t_d3290431); "
            "reassign an independent reviewer profile explicitly"
        )
    return None


def _is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff. A no-op re-assign (dev→dev via CLI/dashboard/``reassign
    --reclaim``), an unassign, or the dispatcher's own
    ``kanban.default_assignee`` write would otherwise lift ``active_pr`` for
    the very implementer that opened the PR. Events without ``from`` (written
    before it was recorded) are not trusted as handoffs — fail closed."""
    if kind != "assigned":
        return True
    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to
