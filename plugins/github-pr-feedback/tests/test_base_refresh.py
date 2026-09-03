from __future__ import annotations

from github_pr_feedback.base_refresh import BaseRefreshIdentity, _receipt_comment
from github_pr_feedback.policy import CODEX_REVIEW_TRIGGER


def test_receipt_comment_mentions_codex_review_since_the_head_moved_past_its_last_review() -> (
    None
):
    """A base-refresh push moves the PR onto a new head Codex has not seen.

    Codex's GitHub App never re-reviews on an ordinary push, only on this
    exact mention -- without it the merge maintainer's codex_review_pending
    gate would wait forever for a review nothing ever asks for.
    """

    identity = BaseRefreshIdentity(
        repository="acme/widgets",
        pr_number=17,
        observed_base_sha="b" * 40,
        target_base_sha="c" * 40,
        base_branch="stable",
        head_repository="acme/widgets",
        head_branch="codex/fix",
        head_sha="a" * 40,
    )

    body = _receipt_comment(identity, "d" * 40, "e" * 64)

    assert CODEX_REVIEW_TRIGGER in body
    assert "<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair head=" + "d" * 40 in body
