"""Operator promotion of a parked card: ``todo``/``blocked`` -> ``ready``.

Split out of ``hermes_cli.kanban_db`` (facade at its line cap); the facade
re-exports :func:`promote_task`, and the origin-resident helpers it needs are
reached late-bound through ``_kb`` so monkeypatching ``kanban_db.<name>`` keeps
working.

The size gate runs BEFORE the status write in the same txn: an oversized card
parked behind a finished parent is routed to ``triage`` for decomposition
instead of being promoted into the dispatch lane, and ``dry_run`` reports that
verdict without writing anything.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from hermes_cli.kanban_size_gate import promote_gate_verdict


def promote_task(
    conn: sqlite3.Connection, task_id: str, *, actor: str, reason: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Operator promotion ``todo``/``blocked`` -> ``ready`` with an audit event.
    Refused while a parent is unfinished; ``dry_run`` only validates.
    Returns ``(ok, reason)``."""
    from hermes_cli.kanban_db import _append_event, _task_status, write_txn

    cur_status = _task_status(conn, task_id)
    if cur_status is None:
        return False, f"task {task_id} not found"

    if cur_status not in ("todo", "blocked"):
        return False, (
            f"task {task_id} is {cur_status!r}; promote only applies to "
            f"'todo' or 'blocked'"
        )

    # No override: claim_task demotes ready -> todo on an undone parent whichever
    # writer set 'ready', so a forced promotion would only report a success the
    # first claim silently reverts (#106195). The dependency itself is the knob.
    parents = conn.execute(
        "SELECT t.id, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ?", (task_id,),
    ).fetchall()
    unsatisfied = [p["id"] for p in parents if p["status"] not in ("done", "archived")]
    if unsatisfied:
        return False, (
            f"unsatisfied parent dependencies: {', '.join(unsatisfied)} "
            f"(the ready -> running claim re-checks parents, so promotion cannot "
            f"bypass them; complete the parents or drop the link with "
            f"`hermes kanban unlink <parent_id> {task_id}`)"
        )

    if dry_run:
        # The same decision path with record=False, so a dry run stamps nothing.
        refusal = promote_gate_verdict(conn, task_id, record=False)
        return (False, refusal) if refusal else (True, None)

    with write_txn(conn):
        refusal = promote_gate_verdict(conn, task_id)
        if refusal:
            return False, refusal
        upd = conn.execute(
            "UPDATE tasks SET status = 'ready' "
            "WHERE id = ? AND status IN ('todo', 'blocked')", (task_id,),
        )
        if upd.rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _append_event(conn, task_id, "promoted_manual", {"actor": actor, "reason": reason})

    return True, None
