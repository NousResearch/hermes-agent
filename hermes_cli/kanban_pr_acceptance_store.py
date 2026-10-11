"""Persist acceptance with the same ownership snapshot as the terminal write.

Two gates share this boundary: PR acceptance (``OWNER/REPO`` / PR URL
contracts, evidence read from GitHub) and proof acceptance (``proof:<command>``
contracts, evidence is the command's exit code in the card's workspace).
"""
from __future__ import annotations

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance
from hermes_cli.kanban_proof import collect_proof, failure_detail, is_proof_contract


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
    if is_proof_contract(contract):
        # The proof runs where the worker worked. The workspace is read now and
        # the snapshot is rechecked under the lock before the receipt lands.
        row = conn.execute("SELECT workspace_kind, workspace_path FROM tasks WHERE id=?", (task_id,)).fetchone()
        return snapshot, collect_proof(contract, task_id=task_id, workspace_kind=row["workspace_kind"],
                                       workspace_path=row["workspace_path"])
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


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt = acceptance
    if _snapshot(conn, task_id) != snapshot:
        return False
    is_proof = receipt.get("gate") == "proof"
    _append_event(conn, task_id, "proof_acceptance" if is_proof else "pr_acceptance", receipt,
                  run_id=snapshot[0])
    if not receipt["ok"]:
        if is_proof:
            detail = failure_detail(receipt)
        else:
            detail = f"PR acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]
