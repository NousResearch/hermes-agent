"""Transaction-bound Triage release contracts for the Kanban database facade."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class TriageReleaseGuardResult:
    """Decision returned by a guarded Triage release precondition callback."""

    allowed: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class TriageReleaseReceipt:
    """Stable result of an attempted guarded Triage release."""

    released: bool
    outcome: str
    status: Optional[str]
    changed_fields: tuple[str, ...] = ()
    guard_reason: Optional[str] = None


class _GuardDenied(Exception):
    def __init__(self, receipt: TriageReleaseReceipt) -> None:
        super().__init__(receipt.guard_reason)
        self.receipt = receipt


def guarded_specify_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    guard: Callable[[sqlite3.Connection, str], TriageReleaseGuardResult],
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
) -> TriageReleaseReceipt:
    """Validate ``guard`` and release one Triage task in the same write transaction.

    The callback receives the transaction-bound connection for read-only precondition
    checks. Returning a denial rolls the transaction back before returning its receipt.
    Callback exceptions and malformed results propagate with the transaction rolled back.
    """
    from hermes_cli import kanban_db as kb

    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = kb._canonical_assignee(assignee)
    try:
        with kb.write_txn(conn):
            existing = conn.execute(
                "SELECT title, body, assignee FROM tasks WHERE id = ? AND status = 'triage'",
                (task_id,),
            ).fetchone()
            if existing is None:
                status = kb._task_status(conn, task_id)
                return TriageReleaseReceipt(False, "not_in_triage", status)

            changes_before_guard = conn.total_changes
            decision = guard(conn, task_id)
            if conn.total_changes != changes_before_guard:
                raise RuntimeError("Triage release guard must be read-only")
            if not isinstance(decision, TriageReleaseGuardResult):
                raise TypeError("guard must return TriageReleaseGuardResult")
            reason = (decision.reason or "").strip() or None
            if not decision.allowed:
                if reason is None:
                    raise ValueError("denied Triage release guard requires a reason")
                raise _GuardDenied(
                    TriageReleaseReceipt(
                        False, "precondition_failed", "triage", guard_reason=reason,
                    )
                )

            sets: list[str] = ["status = 'todo'"]
            params: list[object] = []
            changed_fields: list[str] = []
            if title is not None and title.strip() != (existing["title"] or ""):
                sets.append("title = ?")
                params.append(title.strip())
                changed_fields.append("title")
            if body is not None and (body or "") != (existing["body"] or ""):
                sets.append("body = ?")
                params.append(body)
                changed_fields.append("body")
            if assignee is not None and assignee != (existing["assignee"] or None):
                sets.append("assignee = ?")
                params.append(assignee)
                changed_fields.append("assignee")
            params.append(task_id)
            cur = conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE id = ? AND status = 'triage'",
                tuple(params),
            )
            if cur.rowcount != 1:
                return TriageReleaseReceipt(False, "not_in_triage", kb._task_status(conn, task_id))
            if changed_fields and author and author.strip():
                kb._insert_comment(
                    conn,
                    task_id,
                    author.strip(),
                    "Specified — updated " + ", ".join(changed_fields) + " and promoted to todo.",
                    int(time.time()),
                )
            kb._append_event(
                conn,
                task_id,
                "specified",
                {"changed_fields": changed_fields} if changed_fields else None,
            )
            final_status = "todo"
            if kb._parents_satisfied(conn, task_id):
                final_status = kb._resume_status_from_events(conn, task_id)
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'",
                    (final_status, task_id),
                )
                kb._append_event(
                    conn,
                    task_id,
                    "promoted",
                    {"status": final_status} if final_status != "ready" else None,
                )
            return TriageReleaseReceipt(
                True,
                "released",
                final_status,
                tuple(changed_fields),
            )
    except _GuardDenied as denied:
        return denied.receipt
