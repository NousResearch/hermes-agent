"""Persist acceptance with the same ownership snapshot as the terminal write."""
from __future__ import annotations

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance


def _snapshot(conn, task_id):
    row = conn.execute("SELECT current_run_id, status, completion_contract, acceptance_revision FROM tasks WHERE id=?", (task_id,)).fetchone()
    return tuple(row) if row else None


def prepare_acceptance(conn, task_id, expected_run_id, metadata):
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract, revision = snapshot
    if not contract or contract == "local-only":
        return None
    if status not in {"running", "ready", "blocked", "review"} or (expected_run_id is not None and run_id != expected_run_id):
        return False
    published_pr = metadata.get("published_pr") if isinstance(metadata, dict) else None
    match = _PR.fullmatch(published_pr) if isinstance(published_pr, str) else None
    # Publication binds once. Retrying cannot replace the task's PR with a green sibling.
    if match and contract == match[1]:
        with write_txn(conn):
            if _snapshot(conn, task_id) != snapshot:
                return False
            conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (published_pr, task_id))
        snapshot = (run_id, status, published_pr, revision)
        contract = published_pr
    return snapshot, collect_acceptance(contract, published_pr)


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt = acceptance
    if _snapshot(conn, task_id) != snapshot:
        return False
    _append_event(conn, task_id, "pr_acceptance", receipt, run_id=snapshot[0])
    conn.execute(
        "UPDATE tasks SET acceptance_hold=?, acceptance_revision=acceptance_revision+1 WHERE id=?",
        (int(not receipt["ok"]), task_id),
    )
    # Diagnostics live in the receipt, never in the independent worker error.
    # Even filling an empty worker error with an infra receipt would create an
    # accidental auth guard that survives clearing this acceptance hold.
    return receipt["ok"]


def clear_acceptance_hold(conn, task_id):
    """Explicit operator recovery; never clear worker guards or claim a card.

    Advance the receipt generation so an in-flight network recheck cannot
    reinstall the hold after this operator decision.
    """
    from hermes_cli.kanban_db import _append_event

    with write_txn(conn):
        changed = conn.execute(
            "UPDATE tasks SET acceptance_hold=0, acceptance_revision=acceptance_revision+1 "
            "WHERE id=? AND acceptance_hold=1", (task_id,),
        ).rowcount
        if changed:
            _append_event(conn, task_id, "acceptance_hold_cleared", {"source": "operator"})
        return bool(changed)
