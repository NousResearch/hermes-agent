from __future__ import annotations

import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins" / "github-pr-feedback"
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from datetime import UTC, datetime, timedelta

from github_pr_feedback.github_client import Feedback
from github_pr_feedback.intent_review import classify_feedback, pending_intent_comment_ids
from github_pr_feedback.policy import Reviewer


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def feedback(body: str, *, login: str = "codex", offset: int = 0, bot: bool = True) -> Feedback:
    return Feedback(
        "issue_comment",
        str(offset + 1),
        Reviewer(login),
        body,
        NOW + timedelta(seconds=offset),
        bot,
    )


def test_operator_decision_clears_only_one_explicit_intent_review() -> None:
    items = (
        feedback("Do not apply the proposed patch; use the typed receipt instead."),
        feedback("I disagree; use the bounded retry instead.", offset=1),
        feedback("intent-review: 1 approve original", login="operator", offset=2, bot=False),
    )

    assert classify_feedback(items[0]) is not None
    assert pending_intent_comment_ids(items, owner_login="operator") == frozenset({"2"})


def test_owner_named_bot_cannot_clear_a_pending_review() -> None:
    items = (
        feedback("Rather use the local implementation instead."),
        feedback("dismiss", login="operator", offset=2, bot=True),
    )

    assert pending_intent_comment_ids(items, owner_login="operator") == frozenset({"1"})
