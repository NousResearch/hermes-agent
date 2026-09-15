"""Conservative recognition of separate review/finalize workflow cards.

First-class same-card review (``ready -> review -> running``) stays in
``kanban_db.request_changes``. This sibling covers Impulse-style graphs where
review is a distinct child of implementation and a finalize join depends on
both. Recognition uses the durable ``workflow_role`` column — never titles,
``current_step_key``, or graph topology. A review child targets exactly one
implementation-role parent; that unique explicit parent is the only durable
association. Join membership and sorted link order cannot choose among two
implementation parents, so ``request_changes`` fails closed without mutating
the review run.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

_REVIEW_ROLE = "review"
_FINALIZE_ROLE = "finalize"
_IMPLEMENTATION_ROLE = "implementation"


def _task_workflow_role(conn: sqlite3.Connection, task_id: str) -> str:
    from hermes_cli import kanban_db as kb

    row = conn.execute(
        "SELECT workflow_role FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    return kb.normalize_workflow_role(row["workflow_role"] if row else None) or "ordinary"


def _explicit_implementation_parents(
    conn: sqlite3.Connection, task_id: str,
) -> list[str]:
    from hermes_cli import kanban_db as kb

    return [
        parent_id for parent_id in kb.parent_ids(conn, task_id)
        if _task_workflow_role(conn, parent_id) == _IMPLEMENTATION_ROLE
    ]


def implementation_parent_for_review_child(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[str]:
    """Implementation parent id when ``task_id`` is an explicit review child.

    Safe only when exactly one parent carries ``workflow_role=implementation``.
    Zero parents is not a review-child handoff. Two or more have no unique
    durable association — callers must fail closed rather than pick by link
    order or join topology.
    """
    if _task_workflow_role(conn, task_id) != _REVIEW_ROLE:
        return None
    impl_parents = _explicit_implementation_parents(conn, task_id)
    if len(impl_parents) != 1:
        return None
    return impl_parents[0]


def is_finalize_workflow_card(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when ``task_id`` is explicitly marked finalize."""
    return _task_workflow_role(conn, task_id) == _FINALIZE_ROLE


def is_review_or_finalize_workflow_card(conn: sqlite3.Connection, task_id: str) -> bool:
    """Domain gate: auto-decomposer must not fan these cards out."""
    return _task_workflow_role(conn, task_id) in {_REVIEW_ROLE, _FINALIZE_ROLE}


def apply_review_child_changes(
    conn: sqlite3.Connection, task_id: str, *, reason: str,
    current_run_id: int, reviewer: Optional[str],
) -> tuple[bool, Optional[str], list[tuple[Optional[int], Optional[str]]]]:
    """Caller holds ``write_txn``. Reopen the implementation parent, re-gate
    sibling reviews and finalize descendants, and land this review in ``todo``.

    Returns ``(ok, implementer, terminations)``. Only a ``done`` implementation
    may reopen; archived (or any other status) is left untouched.
    """
    from hermes_cli import kanban_db as kb

    if _task_workflow_role(conn, task_id) != _REVIEW_ROLE:
        return False, "active run was not claimed from review", []
    impl_parents = _explicit_implementation_parents(conn, task_id)
    if not impl_parents:
        return False, "active run was not claimed from review", []
    if len(impl_parents) > 1:
        return False, (
            "review child has multiple implementation parents; "
            "request_changes requires unambiguous provenance"
        ), []
    parent_id = impl_parents[0]

    parent_row = conn.execute(
        "SELECT assignee, status FROM tasks WHERE id = ?", (parent_id,),
    ).fetchone()
    if parent_row is None:
        return False, "review child has no implementation parent", []
    implementer = kb._canonical_assignee(kb._nonblank_str(parent_row["assignee"]))
    reviewer = kb._canonical_assignee(reviewer)
    now = int(time.time())

    cur = conn.execute(
        """
        UPDATE tasks
           SET status = 'todo',
               claim_lock = NULL,
               claim_expires = NULL,
               worker_pid = NULL
         WHERE id = ? AND status = 'running' AND current_run_id = ?
        """,
        (task_id, int(current_run_id)),
    )
    if cur.rowcount != 1:
        return False, "task changed during review handoff", []
    run_id = kb._end_run(
        conn, task_id, outcome="changes_requested", status="todo", summary=reason,
    )
    payload = {
        "reason": reason,
        "implementer": implementer,
        "reviewer": reviewer,
        "status": "todo",
        "parent": parent_id,
    }
    kb._append_event(conn, task_id, "changes_requested", payload, run_id=run_id)
    kb._insert_comment(
        conn, task_id, reviewer or "reviewer",
        f"Changes requested on implementation {parent_id}: {reason}", now,
    )

    terminations: list[tuple[Optional[int], Optional[str]]] = []
    if parent_row["status"] == "done":
        landing = kb._landing_status_after_parents(conn, parent_id)
        conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   completed_at = NULL,
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL
             WHERE id = ? AND status = 'done'
            """,
            (landing, parent_id),
        )
        kb._append_event(
            conn, parent_id, "changes_requested",
            {
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
                "review_task_id": task_id,
                "status": landing,
            },
        )
        kb._append_event(
            conn, parent_id, "status",
            {
                "status": landing, "reason": "review_child_changes_requested",
                "review_task_id": task_id, "previous_status": parent_row["status"],
            },
        )
        kb._insert_comment(
            conn, parent_id, reviewer or "reviewer",
            f"Reopened by review child {task_id}: {reason}", now,
        )
        result = kb.invalidate_descendants_for_parent_reopen(
            conn, parent_id, author=reviewer or "reviewer",
        )
        terminations = list(result.get("terminations") or [])
    return True, implementer, terminations
