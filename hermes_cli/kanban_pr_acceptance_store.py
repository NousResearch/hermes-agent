"""Persist acceptance with the same ownership snapshot as the terminal write."""
from __future__ import annotations

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_delivery_acceptance import collect_delivery_acceptance, is_source_issue_intake
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance


def _snapshot(conn, task_id):
    row = conn.execute(
        "SELECT current_run_id, status, completion_contract, idempotency_key, assignee FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    return tuple(row) if row else None


def prepare_acceptance(conn, task_id, expected_run_id, metadata):
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract, source_key, assignee = snapshot
    if not isinstance(contract, str) or contract == "local-only":
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
        snapshot = (run_id, status, published_pr, source_key, assignee)
        contract = published_pr
    github_source = isinstance(source_key, str) and source_key.casefold().startswith("github:")
    valid_issue_source = is_source_issue_intake(source_key)
    candidate_gate = metadata.get("local_gate") if valid_issue_source and isinstance(metadata, dict) else None
    pr_receipt = collect_acceptance(
        contract, published_pr, local_gate=candidate_gate,
        coordinator=assignee if valid_issue_source else None,
    )
    delivery_receipt = None
    if github_source:
        delivery_receipt = collect_delivery_acceptance(
            source_key, published_pr, expected_head_sha=pr_receipt.get("head_sha"),
        )
    return snapshot, (pr_receipt, delivery_receipt)


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipts = acceptance
    pr_receipt, delivery_receipt = receipts
    if _snapshot(conn, task_id) != snapshot:
        return False
    _append_event(conn, task_id, "pr_acceptance", pr_receipt, run_id=snapshot[0])
    if delivery_receipt is not None:
        _append_event(conn, task_id, "delivery_acceptance", delivery_receipt, run_id=snapshot[0])
    failures = []
    if not pr_receipt["ok"]:
        failures.append(f"PR acceptance {pr_receipt['classification']}: {pr_receipt.get('detail', '')} {pr_receipt['recovery']}")
    if delivery_receipt is not None and not delivery_receipt["ok"]:
        failures.append(f"Delivery acceptance {delivery_receipt['classification']}: {delivery_receipt.get('detail', '')} {delivery_receipt['recovery']}")
    if failures:
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (" ".join(failures), task_id))
    return not failures
