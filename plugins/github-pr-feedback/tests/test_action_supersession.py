from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from github_pr_feedback.ci_admission import local_ci_admission_blocker
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.ledger_action_supersession import reconcile_inactive_actioned_duplicates
from github_pr_feedback.policy import FeedbackReceipt


def dispatched(ledger, identity, head, task):
    item = FeedbackReceipt("acme/repo", 1, "review_comment", identity, head)
    now = datetime.now(UTC)
    lease = ledger.claim(item, owner="test", claimed_at=now, stale_before=now - timedelta(minutes=15))
    assert lease is not None
    ledger.finalize(item, task, lease)
    return item


@pytest.mark.parametrize("task_status", ["done", "archived"])
def test_acknowledged_duplicate_no_longer_deadlocks_exact_head_ci(tmp_path, task_status):
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    old = dispatched(ledger, "comment-1", "a" * 40, "old-task")
    current_item = dispatched(ledger, "comment-1", "b" * 40, "new-task")
    ledger.mark_feedback_actioned(current_item, resolved_head_sha="b" * 40, actioned_at=datetime.now(UTC))
    current = SimpleNamespace(base_repository="acme/repo", number=1, head_sha="b" * 40, base_sha="c" * 40)
    state = SimpleNamespace(repository="acme/repo", number=1, head_sha=current.head_sha,
                            base_sha=current.base_sha, state="OPEN", merged=False,
                            mergeable=True, merge_state_status="CLEAN")
    github = SimpleNamespace(get_merge_state=lambda *_: state)
    kanban = SimpleNamespace(task_status=lambda *_: task_status)
    try:
        assert ledger.was_actioned_on_any_head(old)
        assert local_ci_admission_blocker(github, ledger, current) == "mutation_pending"
        assert reconcile_inactive_actioned_duplicates(ledger, kanban, github, current, board="test") == 1
        assert local_ci_admission_blocker(github, ledger, current) is None
        assert ledger.latest_ci_receipt_for_head("acme/repo", 1, current.head_sha) is None
        assert reconcile_inactive_actioned_duplicates(ledger, kanban, github, current, board="test") == 0
    finally:
        ledger.close()


@pytest.mark.parametrize("task_status,acknowledged,head_race", [
    ("running", True, False), ("ready", True, False), (None, True, False),
    ("blocked", True, False), ("done", False, False), ("done", True, True),
])
def test_unfinished_unacknowledged_or_changed_work_stays_pending(tmp_path, task_status, acknowledged, head_race):
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    dispatched(ledger, "comment-1", "a" * 40, "old-task")
    item = dispatched(ledger, "comment-1", "b" * 40, "new-task")
    if acknowledged:
        ledger.mark_feedback_actioned(item, resolved_head_sha="b" * 40, actioned_at=datetime.now(UTC))
    current = SimpleNamespace(base_repository="acme/repo", number=1, head_sha="b" * 40, base_sha="c" * 40)
    state = SimpleNamespace(repository="acme/repo", number=1, head_sha="d" * 40 if head_race else current.head_sha,
                            base_sha=current.base_sha, state="OPEN", merged=False)
    try:
        assert reconcile_inactive_actioned_duplicates(
            ledger, SimpleNamespace(task_status=lambda *_: task_status),
            SimpleNamespace(get_merge_state=lambda *_: state), current, board="test") == 0
        assert ledger.has_pending_mutation("acme/repo", 1)
    finally:
        ledger.close()
