from __future__ import annotations

from github_pr_feedback.base_refresh import BaseRefreshIdentity, _receipt_comment


def test_receipt_comment_does_not_request_duplicate_codex_review() -> (
    None
):
    """Repository review automation owns Codex scheduling after a base refresh."""

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

    assert "<!-- pr-maintenance-receipt:v1 status=completed kind=pr_repair head=" + "d" * 40 in body
