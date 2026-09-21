from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from github_pr_feedback.feedback_retirement import retire_closed_feedback
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import FeedbackReceipt, PullRequest, load_policy


@pytest.fixture
def dispatched(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    policy = load_policy({
        "enabled": True, "repositories": [{"base_repository": "acme/widgets",
        "head_repository": "acme/widgets", "local_path": str(tmp_path),
        "owner_login": "owner", "branch_prefixes": ["codex/"]}],
        "reviewer_logins": ["reviewer"], "reviewer_associations": [], "not_before": "2026-08-24T00:00:00Z",
        "assignee": "repair-agent", "board": "repairs",
    })
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt("acme/widgets", 17, "review", "42", "a" * 40)
    now = datetime.now(UTC)
    lease = ledger.claim(receipt, owner="test", claimed_at=now, stale_before=now-timedelta(minutes=5))
    ledger.finalize(receipt, "task-1", lease)
    pull = PullRequest(17, "CLOSED", "acme/widgets", "acme/widgets", "owner",
                       "codex/fix", "a" * 40, base_branch="main", base_sha="b" * 40)
    yield policy, ledger, receipt, pull
    ledger.close()


@pytest.mark.parametrize("state", ["CLOSED", "MERGED"])
def test_closed_retirement_clears_pending_gate_without_claiming_repair_success(dispatched, state):
    policy, ledger, receipt, pull = dispatched
    pull = replace(pull, state=state)
    github = SimpleNamespace(get_pull_request=lambda *_: pull)
    assert ledger.exact_pending_task_binding(receipt) is not None
    for _ in range(2):
        assert retire_closed_feedback(policy, github, ledger, receipt)["task_id"] == "task-1"
    assert ledger.exact_pending_task_binding(receipt) is None
    assert ledger.pending_task_bindings_for_head(receipt) == ()
    assert not ledger.was_actioned_on_any_head(receipt)
    # New feedback after reopening on this same head must still be dispatchable.
    later = replace(receipt, feedback_id="43")
    now = datetime.now(UTC)
    assert ledger.claim(later, owner="reopened", claimed_at=now, stale_before=now-timedelta(minutes=5))


@pytest.mark.parametrize("change", ["open", "raced_open", "head", "repository", "number"])
def test_unverified_closure_leaves_pending_receipt_intact(dispatched, change):
    policy, ledger, receipt, pull = dispatched
    updated = {"open": replace(pull, state="OPEN"), "raced_open": replace(pull, state="OPEN"),
               "head": replace(pull, head_sha="c" * 40),
               "repository": replace(pull, base_repository="elsewhere/widgets"),
               "number": replace(pull, number=18)}[change]
    pulls = iter([pull, updated] if change == "raced_open" else [updated, updated])
    github = SimpleNamespace(get_pull_request=lambda *_: next(pulls))
    with pytest.raises(ValueError):
        retire_closed_feedback(policy, github, ledger, receipt)
    assert ledger.exact_pending_task_binding(receipt) is not None
