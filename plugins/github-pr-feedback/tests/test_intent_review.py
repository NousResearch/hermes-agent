from __future__ import annotations

from datetime import UTC, datetime, timedelta

from github_pr_feedback.github_client import Feedback
from github_pr_feedback.intent_review import classify_feedback, pending_intent_review
from github_pr_feedback.policy import Reviewer


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def feedback(body: str, *, login: str = "codex", offset: int = 0, bot: bool = True) -> Feedback:
    return Feedback("issue_comment", str(offset + 1), Reviewer(login), body, NOW + timedelta(seconds=offset), bot)


def test_normal_bot_fix_comment_remains_automatic() -> None:
    assert classify_feedback(feedback("Please fix the timeout and push the tested change.")) is None


def test_explicit_disagreement_requires_per_pr_intent_decision() -> None:
    item = feedback("I disagree with this fix; use the bounded retry approach instead.")
    assert classify_feedback(item) is not None
    assert pending_intent_review((item,), owner_login="operator") is True


def test_operator_decision_clears_only_that_intent_review() -> None:
    items = (
        feedback("Do not apply the proposed patch; use the typed receipt instead."),
        feedback("use alternative: typed receipt", login="operator", offset=1, bot=False),
    )
    assert pending_intent_review(items, owner_login="operator") is False


def test_bot_cannot_clear_operator_intent_review() -> None:
    items = (
        feedback("Rather use the local implementation instead."),
        feedback("approve original", login="codex", offset=1),
    )
    assert pending_intent_review(items, owner_login="operator") is True


def test_multiple_intent_reviews_require_an_explicit_comment_id() -> None:
    items = (
        feedback("Do not apply this; use the typed receipt instead.", offset=0),
        feedback("I disagree; rather use the bounded retry instead.", offset=1),
        feedback("dismiss", login="operator", offset=2, bot=False),
    )
    assert pending_intent_review(items, owner_login="operator") is True
