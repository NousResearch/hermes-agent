"""Retire obsolete feedback without representing closure as a successful repair."""
from dataclasses import replace
from datetime import UTC, datetime

from .ledger import LedgerStateError


def retire_closed_feedback(policy, github, ledger, receipt):
    # pr_local_ci normally completes through audit-pr's own typed-receipt flow,
    # not this one -- but audit-pr rejects a non-OPEN PR identity outright, so
    # a card whose PR closes mid-audit has no other path to clear its pending
    # ledger row. Retire it here too rather than leaving it stuck forever.
    current = github.get_pull_request(receipt.repository, receipt.pr_number)
    if (not policy.enabled or current.state not in {"CLOSED", "MERGED"}
            or current.head_sha != receipt.head_sha
            or current.number != receipt.pr_number or current.base_repository != receipt.repository
            or not policy.admit_pull_request(replace(current, state="OPEN")).admitted):
        raise ValueError("receipt is not bound to a closed canonical PR")
    # Re-read after admission: a reopened or changed PR must keep its pending gate.
    if github.get_pull_request(receipt.repository, receipt.pr_number) != current:
        raise ValueError("canonical PR changed during retirement")
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
                (datetime.now(UTC).isoformat(), f"canonical PR {current.state}; repair superseded", *receipt.key),
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
        payload = retire_closed_feedback(policy, github, ledger, receipt)
    except (GitHubClientError, ValueError, LedgerStateError) as error:
        print(json.dumps({"status": "retirement_unavailable", "reason": str(error)}))
        return 1
    finally:
        if ledger is not None:
            ledger.close()
    print(json.dumps(payload, sort_keys=True))
    return 0
