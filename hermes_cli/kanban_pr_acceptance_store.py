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
    if contract == "artifact-review-v1":
        # A protected receipt must bind an actual review attempt.  Permitting a
        # parked review card with no active run would record run_id=NULL and
        # then synthesize a different terminal run, stranding dependants.
        if run_id is None:
            return False
        from hermes_cli.kanban_artifact_acceptance import collect_acceptance as collect_artifact_acceptance
        return snapshot, collect_artifact_acceptance(conn, task_id, snapshot, metadata)
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
    return snapshot, collect_acceptance(contract, published_pr)


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt = acceptance
    if _snapshot(conn, task_id) != snapshot:
        return False
    event_kind = (
        "artifact_review_acceptance"
        if snapshot[2] == "artifact-review-v1"
        else "pr_acceptance"
    )
    if event_kind == "artifact_review_acceptance":
        from hermes_cli.kanban_artifact_acceptance import revalidate_receipt_artifacts
        if receipt.get("ok") and not revalidate_receipt_artifacts(receipt):
            return False
    _append_event(conn, task_id, event_kind, receipt, run_id=snapshot[0])
    if not receipt["ok"]:
        label = "Artifact-review" if event_kind == "artifact_review_acceptance" else "PR"
        detail = f"{label} acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]
