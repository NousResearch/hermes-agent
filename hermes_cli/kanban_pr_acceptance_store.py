"""Persist acceptance with the same ownership snapshot as the terminal write."""
from __future__ import annotations

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance


def _snapshot(conn, task_id):
    row = conn.execute("SELECT current_run_id, status, completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    return tuple(row) if row else None


def prepare_acceptance(conn, task_id, expected_run_id, metadata):
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract = snapshot
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
        snapshot = (run_id, status, published_pr)
        contract = published_pr
    # The assignee profile's gh login owns the repo: acceptance must not run as
    # the ambient login of whichever process completes the card (#122689).
    assignee = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()["assignee"]
    return snapshot, collect_acceptance(contract, published_pr, assignee=assignee)


def acceptance_failure_detail(receipt):
    """Human-facing stamp; classification authority remains the stored receipt."""
    return f"PR acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"


def pending_acceptance_failure(conn, task_id, failure_error):
    """Return the current typed failure, not an older superseded receipt.

    Match the producer's exact stamp only to establish ownership of the task's
    current failure. Never infer a classification from diagnostic wording.
    Clearing/replacing the stamp makes old acceptance events inapplicable.
    """
    from hermes_cli.kanban_db import _json_dict

    if not failure_error:
        return None
    event = conn.execute(
        "SELECT payload, created_at FROM task_events "
        "WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if event is None:
        return None
    receipt = _json_dict(event["payload"])
    if (receipt.get("ok") is not False
            or receipt.get("classification") not in ("auth", "retry", "policy", "infra")
            or "recovery" not in receipt
            or acceptance_failure_detail(receipt) != failure_error):
        return None
    return receipt["classification"], int(event["created_at"])


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt = acceptance
    if _snapshot(conn, task_id) != snapshot:
        return False
    _append_event(conn, task_id, "pr_acceptance", receipt, run_id=snapshot[0])
    if not receipt["ok"]:
        detail = acceptance_failure_detail(receipt)
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]
