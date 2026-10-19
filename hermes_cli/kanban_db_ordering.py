"""Sort order for ``hermes kanban list --sort`` and the manual per-lane card order:
the canonical ``ORDER BY`` fragments, ``list_tasks`` (the query that applies them) and
``reorder_task`` (the write that moves one card within its lane).

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Optional

import hermes_cli.kanban_db as _kb
from hermes_cli.kanban_workflow import DEFAULT_STATUSES as VALID_STATUSES

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


# Canonical sort-order mappings for ``hermes kanban list --sort``.
# Each value is a raw SQL fragment appended after ``ORDER BY``.
VALID_SORT_ORDERS: dict[str, str] = {
    "created": "created_at ASC, id ASC",
    "created-desc": "created_at DESC, id DESC",
    "priority": "priority DESC, created_at ASC",
    "priority-desc": "priority ASC, created_at ASC",
    "status": "status ASC, created_at ASC",
    "assignee": "assignee ASC, created_at ASC",
    "title": "title ASC, id ASC",
    "updated": "started_at DESC NULLS LAST, created_at DESC",
    "completed-desc": "completed_at DESC NULLS LAST, id DESC",
    # Manual order written by ``reorder_task()``; ties (every never-reordered
    # row carries 0) keep the created_at ASC reading order. Display-only — the
    # dispatcher's claim order is unchanged.
    "manual": "sort_order ASC, created_at ASC, id ASC",
}


def list_tasks(
    conn: sqlite3.Connection, *, assignee: Optional[str] = None, status: Optional[str] = None,
    tenant: Optional[str] = None, session_id: Optional[str] = None, include_archived: bool = False,
    limit: Optional[int] = None, order_by: Optional[str] = None,
    workflow_template_id: Optional[str] = None, current_step_key: Optional[str] = None,
) -> list[Task]:
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    for col, val in (
        ("assignee", _kb._canonical_assignee(assignee)), ("status", status), ("tenant", tenant),
        ("session_id", session_id), ("workflow_template_id", workflow_template_id),
        ("current_step_key", current_step_key),
    ):
        if val is not None:
            query += f" AND {col} = ?"
            params.append(val)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    if order_by is not None:
        order_by = order_by.strip().lower()
        if order_by not in VALID_SORT_ORDERS:
            raise ValueError(f"order_by must be one of {sorted(VALID_SORT_ORDERS.keys())}")
        query += f" ORDER BY {VALID_SORT_ORDERS[order_by]}"
    else:
        query += " ORDER BY priority DESC, created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [_kb.Task.from_row(r) for r in rows]


# Lane renumbering step for ``reorder_task`` (see its docstring): positions are
# spaced by this much so an insert between two neighbours always keeps an
# integer midpoint free for the next move.
_MANUAL_ORDER_STRIDE = 2


def reorder_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    before_id: Optional[str] = None,
    after_id: Optional[str] = None,
    top: bool = False,
    bottom: bool = False,
) -> bool:
    """Move ``task_id`` within its lane for ``order_by="manual"``.

    Exactly one of ``before_id`` / ``after_id`` / ``top`` / ``bottom`` names the
    target slot; a ``before_id`` / ``after_id`` reference must be in the SAME
    status, because manual order only exists inside a lane. The moved task's
    ``sort_order`` becomes the midpoint of its new neighbours, so a single move
    never rewrites the lane; when the neighbours leave no integer midpoint (all
    rows start at 0, which is also what makes the default reading order
    ``created_at ASC``) the lane is renumbered around the new position first.

    Display-only: the dispatcher still claims by ``priority DESC, created_at ASC``.

    Returns False when ``task_id`` does not exist, and raises ValueError for an
    unusable selection (none or several of the four arguments, a self-reference,
    an unknown neighbour id, or a neighbour from another lane).
    """
    if (before_id is not None) + (after_id is not None) + int(bool(top)) + int(bool(bottom)) != 1:
        raise ValueError(
            "reorder_task needs exactly one of before_id / after_id / top / bottom"
        )
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return False
    lane = row["status"]
    anchor = before_id if before_id is not None else after_id
    if anchor is not None:
        if anchor == task_id:
            raise ValueError(f"cannot reorder {task_id} relative to itself")
        ref = conn.execute("SELECT status FROM tasks WHERE id = ?", (anchor,)).fetchone()
        if ref is None:
            raise ValueError(f"no such task: {anchor}")
        if ref["status"] != lane:
            raise ValueError(
                f"cannot reorder across lanes: {anchor} is in {ref['status']!r}, "
                f"{task_id} is in {lane!r}"
            )

    def _lane() -> list[tuple[str, int]]:
        """``(id, sort_order)`` for every task in the lane, in manual order."""
        return [
            (r["id"], int(r["sort_order"] or 0))
            for r in conn.execute(
                "SELECT id, sort_order FROM tasks WHERE status = ?"
                " ORDER BY sort_order ASC, created_at ASC, id ASC",
                (lane,),
            )
        ]

    def _slot(neighbours: list[tuple[str, int]]) -> Optional[int]:
        """``sort_order`` for the moved task, or None when no integer gap is left."""
        if not neighbours:
            return 0
        if top:
            return neighbours[0][1] - 1
        if bottom:
            return neighbours[-1][1] + 1
        idx = next(i for i, (tid, _) in enumerate(neighbours) if tid == anchor)
        if before_id is not None:
            if idx == 0:
                return neighbours[0][1] - 1
            left, right = neighbours[idx - 1][1], neighbours[idx][1]
        else:
            if idx + 1 == len(neighbours):
                return neighbours[-1][1] + 1
            left, right = neighbours[idx][1], neighbours[idx + 1][1]
        return (left + right) // 2 if right - left > 1 else None

    def _lane_after_move() -> list[str]:
        """Lane ids in the order they should read once ``task_id`` has moved."""
        order = [tid for tid, _ in _lane()]
        order.remove(task_id)
        if top:
            order.insert(0, task_id)
        elif bottom:
            order.append(task_id)
        else:
            idx = order.index(anchor)
            order.insert(idx if before_id is not None else idx + 1, task_id)
        return order

    with _kb.write_txn(conn):
        slot = _slot([r for r in _lane() if r[0] != task_id])
        if slot is None:
            for position, tid in enumerate(_lane_after_move()):
                conn.execute(
                    "UPDATE tasks SET sort_order = ? WHERE id = ?",
                    (position * _MANUAL_ORDER_STRIDE, tid),
                )
        else:
            conn.execute("UPDATE tasks SET sort_order = ? WHERE id = ?", (slot, task_id))
    return True
