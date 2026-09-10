from datetime import UTC, datetime, timedelta

import pytest

from github_pr_feedback.ci_runner import CIAuditIdentity
from github_pr_feedback.cli_audit_task import owns_current_audit_task
from github_pr_feedback.controller import _local_ci_feedback_id
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import FeedbackReceipt


@pytest.mark.parametrize("kind,task,head,base,expected", [
    ("pr_local_ci", "audit", "a", "b", True),
    ("review_comment", "audit", "a", "b", False),
    ("pr_local_ci", "repair", "a", "b", False),
    ("pr_local_ci", "audit", "c", "b", False),
    ("pr_local_ci", "audit", "a", "c", False),
])
def test_only_exact_audit_dispatch_owns_worker_lifecycle(tmp_path, monkeypatch, kind, task, head, base, expected):
    identity = CIAuditIdentity("acme/widgets", 17, "a" * 40, "b" * 40)
    ledger = FeedbackLedger(tmp_path / "feedback.db")
    now = datetime.now(UTC)
    dispatched = CIAuditIdentity("acme/widgets", 17, head * 40, base * 40)
    receipt = FeedbackReceipt("acme/widgets", 17, kind, _local_ci_feedback_id(dispatched), dispatched.head_sha)
    try:
        lease = ledger.claim(receipt, owner="test", claimed_at=now, stale_before=now - timedelta(minutes=5))
        ledger.finalize(receipt, "audit", lease)
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)
        assert owns_current_audit_task(ledger, identity) is expected
    finally:
        ledger.close()
