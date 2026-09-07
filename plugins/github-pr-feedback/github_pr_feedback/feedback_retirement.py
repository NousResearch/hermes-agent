"""Retire obsolete feedback without representing closure as a successful repair."""
from dataclasses import replace
from datetime import UTC, datetime

from .ledger import LedgerStateError


def retire_closed_feedback(policy, github, ledger, receipt):
    if receipt.feedback_kind == "pr_local_ci":
        raise ValueError("CI dispatches require their own governed lifecycle")
    current = github.get_pull_request(receipt.repository, receipt.pr_number)
    if (not policy.enabled or current.state not in {"CLOSED", "MERGED"}
            or current.head_sha != receipt.head_sha
            or current.number != receipt.pr_number or current.base_repository != receipt.repository
            or not policy.admit_pull_request(replace(current, state="OPEN")).admitted):
        raise ValueError("receipt is not bound to a closed canonical PR")
    # Re-read after admission: a reopened or changed PR must keep its pending gate.
    if github.get_pull_request(receipt.repository, receipt.pr_number) != current:
        raise ValueError("canonical PR changed during retirement")
    return _retire_dispatch(ledger, receipt, current, f"canonical PR {current.state}; repair superseded")


def retire_self_receipt(policy, github, ledger, receipt):
    from .controller import _is_self_resolution_receipt

    identity = policy.github_identity
    current = github.get_pull_request(receipt.repository, receipt.pr_number)
    if (identity is None or not policy.enabled or current.head_sha != receipt.head_sha
            or current.number != receipt.pr_number or current.base_repository != receipt.repository
            or not policy.admit_pull_request(current).admitted):
        raise ValueError("receipt is not bound to an admitted current PR")

    def matching_feedback():
        return next((item for item in github.list_feedback(receipt.repository, receipt.pr_number)
                     if item.kind == receipt.feedback_kind and item.feedback_id == receipt.feedback_id), None)

    feedback = matching_feedback()
    if feedback is None or not _is_self_resolution_receipt(feedback, owner_login=identity.expected_login):
        raise ValueError("feedback is not a verified self-maintenance receipt")
    if github.get_pull_request(receipt.repository, receipt.pr_number) != current or matching_feedback() != feedback:
        raise ValueError("canonical PR or feedback changed during retirement")
    return _retire_dispatch(ledger, receipt, current, "non-actionable self-maintenance receipt; no repair claimed")


def _retire_dispatch(ledger, receipt, current, reason):
    with ledger._transaction():
        row = ledger._connection.execute(
            "SELECT task_id, status, action_status FROM feedback_receipts "
            "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
            "AND feedback_id = ? AND head_sha = ?", receipt.key,
        ).fetchone()
        if not row or not row[0] or row[1] != "completed" or row[2] not in {"pending", "superseded"}:
            raise LedgerStateError("receipt is not an exact pending dispatch")
        if row[2] == "pending":
            ledger._connection.execute(
                "UPDATE feedback_receipts SET action_status = 'superseded', actioned_at = ?, "
                "last_error = ? WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
                "AND feedback_id = ? AND head_sha = ? AND action_status = 'pending'",
                (datetime.now(UTC).isoformat(), reason, *receipt.key),
            )
    return {"status": "retired", "task_id": row[0], "repository": receipt.repository,
            "pr_number": receipt.pr_number, "head_sha": receipt.head_sha, "pr_state": current.state}


def run_retirement(ctx, args):
    import json
    from .cli import _github_client, _load_policy_from_context
    from .github_client import GitHubClientError
    from .ledger import FeedbackLedger
    from .policy import FeedbackReceipt

    ledger = None
    try:
        policy = _load_policy_from_context(ctx)
        receipt = FeedbackReceipt(args.repository, args.pr_number, args.feedback_kind,
                                  args.feedback_id, args.receipt_head_sha)
        github = _github_client(policy)
        ledger = FeedbackLedger.for_current_profile()
        retire = retire_self_receipt if getattr(args, "self_receipt", False) else retire_closed_feedback
        payload = retire(policy, github, ledger, receipt)
    except (GitHubClientError, ValueError, LedgerStateError) as error:
        print(json.dumps({"status": "retirement_unavailable", "reason": str(error)}))
        return 1
    finally:
        if ledger is not None:
            ledger.close()
    print(json.dumps(payload, sort_keys=True))
    return 0
