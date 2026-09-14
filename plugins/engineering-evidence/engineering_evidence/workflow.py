"""Pure admission checks for the opt-in issue-to-PR workflow."""

from __future__ import annotations

import contextlib
from typing import Any, Iterable
from pathlib import Path

from .receipts import ReceiptValidationError, validate_exact_head


_EVIDENCE_CHILDREN = (
    ("exploration", "architecture-steward", "codebase_snapshot"),
    ("testing", "test-contract-steward", "test_discovery"),
    ("review", "review-verification-steward", "independent_review"),
    ("operations", "operations-steward", "deployment_correlation"),
)


def create_evidence_children(
    *,
    parent_id: str,
    repository: str,
    head_sha: str,
    db_path: Path | None = None,
    board: str | None = None,
) -> dict[str, Any]:
    """Create an explicit, idempotent evidence graph under an existing task.

    This is an opt-in command. It never changes ``kanban.auto_decompose`` and never
    grants publication, merge, deployment, or trading authority to a child.
    """

    if not repository.strip():
        raise ValueError("repository is required")
    if len(head_sha) != 40 or any(ch not in "0123456789abcdef" for ch in head_sha):
        raise ValueError("head_sha must be a 40-character hexadecimal commit")
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    if db_path is not None:
        kbc.init_db(db_path, board=board)
        conn = kbc.connect(db_path, board=board)
    else:
        kbc.init_db(board=board)
        conn = kbc.connect(board=board)
    try:
        if kb.get_task(conn, parent_id) is None:
            raise ValueError(f"unknown parent task {parent_id}")
        children: list[dict[str, Any]] = []
        for lane, assignee, receipt_type in _EVIDENCE_CHILDREN:
            child_id = kb.create_task(
                conn,
                title=f"Engineering evidence: {lane}",
                body=(
                    f"Parent task: {parent_id}\nRepository: {repository}\nExact HEAD: {head_sha}\n"
                    f"Expected receipt: {receipt_type}\nAuthority: diagnostic-only\n"
                    "Do not publish, merge, deploy, trade, or grant credentials from this task."
                ),
                assignee=assignee,
                created_by="engineering-evidence",
                parents=(parent_id,),
                initial_status="running",
                idempotency_key=f"engineering-evidence:{parent_id}:{head_sha}:{lane}",
                board=board,
            )
            task = kb.get_task(conn, child_id)
            children.append({"lane": lane, "task_id": child_id, "assignee": assignee, "status": task.status if task else None})
        return {
            "parent_task_id": parent_id,
            "repository": repository,
            "head_sha": head_sha,
            "children": children,
            "automatic_authority": [],
        }
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def evaluate_issue_to_pr_gate(
    receipts: Iterable[dict[str, Any]],
    *,
    current_head: str,
    independent_review: bool,
    operator_approved: bool,
    hosted_checks_complete: bool,
) -> dict[str, Any]:
    """Return a fail-closed decision; this function has no publication side effects."""

    reasons: list[str] = []
    validated: list[dict[str, Any]] = []
    for receipt in receipts:
        try:
            validated.append(validate_exact_head(receipt, current_head))
        except ReceiptValidationError as exc:
            reasons.append(str(exc))
    types = {receipt["receipt_type"] for receipt in validated}
    for required in ("codebase_snapshot", "test_discovery"):
        if required not in types:
            reasons.append(f"missing_receipt:{required}")
    if not independent_review:
        reasons.append("independent_review_required")
    if not hosted_checks_complete:
        reasons.append("hosted_checks_incomplete")
    if not operator_approved:
        reasons.append("operator_approval_required")
    return {
        "allowed": not reasons,
        "authority": "diagnostic-only",
        "publication_action": "none",
        "reasons": reasons,
        "receipt_types": sorted(types),
    }
