from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.ledger_conflict_supersession import reconcile_inactive_conflicts
from github_pr_feedback.policy import FeedbackReceipt


@pytest.mark.parametrize("card_status,raced,conflict,expected", [
    ("done", False, False, 1),
    ("archived", False, False, 1),
    ("running", False, False, 0),
    ("done", True, False, 0),
    ("done", False, True, 0),
])
def test_only_inactive_obsolete_conflicts_are_superseded(tmp_path, card_status, raced, conflict, expected):
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    now = datetime.now(UTC)
    old = FeedbackReceipt("acme/repo", 1, "pr_repair", "repair:merge_conflict:target-base:" + "b" * 40, "a" * 40)
    lease = ledger.claim(old, owner="test", claimed_at=now, stale_before=now - timedelta(minutes=15))
    ledger.finalize(old, "old-card", lease)
    pull = SimpleNamespace(repository="acme/repo", number=1, head_sha="c" * 40, base_sha="b" * 40,
                           state="OPEN", merged=False, mergeable=not conflict, merge_state_status="DIRTY" if conflict else "CLEAN")
    current = SimpleNamespace(**vars(pull))
    if raced:
        current.head_sha = "d" * 40
    kanban = SimpleNamespace(task_status=lambda board, task: card_status)
    github = SimpleNamespace(get_merge_state=lambda repo, number: current)
    try:
        assert reconcile_inactive_conflicts(ledger, kanban, github, pull, board="test") == expected
        assert ledger.has_pending_mutation("acme/repo", 1) is (expected == 0)
        assert reconcile_inactive_conflicts(ledger, kanban, github, pull, board="test") == 0
    finally:
        ledger.close()


def test_current_head_conflicts_and_review_findings_remain_pending(tmp_path):
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    now = datetime.now(UTC)
    for kind, identity, head in (("pr_repair", "repair:merge_conflict:target-base:" + "b" * 40, "c" * 40),
                                 ("review_comment", "17", "a" * 40)):
        receipt = FeedbackReceipt("acme/repo", 1, kind, identity, head)
        lease = ledger.claim(receipt, owner="test", claimed_at=now, stale_before=now - timedelta(minutes=15))
        ledger.finalize(receipt, identity, lease)
    pull = SimpleNamespace(repository="acme/repo", number=1, head_sha="c" * 40, base_sha="b" * 40,
                           state="OPEN", merged=False, mergeable=True, merge_state_status="CLEAN")
    try:
        assert reconcile_inactive_conflicts(ledger, SimpleNamespace(task_status=lambda *_: "done"),
                                           SimpleNamespace(get_merge_state=lambda *_: pull), pull, board="test") == 0
        assert ledger.has_pending_mutation("acme/repo", 1)
    finally:
        ledger.close()
