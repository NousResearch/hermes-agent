"""Thin ``gh`` CLI wrapper for submitting PR reviews.

Isolated from the Slack wiring so the posting decisions (event → flag mapping,
own-PR fallback, error surfacing) are pure and unit-testable with an injected
``run`` callable. The real ``run`` is :func:`subprocess.run`.

Auth is whatever ``gh`` is already logged in as (keyring, account
``jtomek-strike48``) — the same identity the cron review sweep uses.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)

Number = Union[int, str]

# event → gh flag. "comment" leaves a non-blocking review; the others gate.
_EVENT_FLAG = {
    "approve": "--approve",
    "request-changes": "--request-changes",
    "comment": "--comment",
}

# gh/GraphQL error fragments that mean "you can't formally review your own PR".
# We fall back to a plain issue comment so the operator's click still lands.
_OWN_PR_MARKERS = (
    "can not approve your own",
    "cannot approve your own",
    "can not request changes on your own",
    "review your own pull request",
)


@dataclass
class ReviewResult:
    ok: bool
    detail: str = ""
    # True when the post was refused because the PR is authored by the acting
    # gh user. The operator does not want Hermes to approve/comment on its own
    # PRs, so this is a deliberate skip, not a failure to retry.
    own_pr: bool = False


def current_head_sha(
    repo: str,
    number: Number,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Optional[str]:
    """Return the PR's current head SHA, or ``None`` if it can't be fetched.

    Used for the stale-guard: if the head moved since the review was staged,
    the click handler refuses to post a review of code the operator didn't see.
    A ``None`` return now fails the guard *closed* (see ``actions.is_stale``),
    so we log the underlying ``gh`` error — otherwise a persistent auth failure
    looks identical to the head genuinely moving.
    """
    proc = run(
        ["gh", "pr", "view", str(number), "-R", repo, "--json", "headRefOid"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(
            "[pr_review_buttons] gh pr view %s#%s failed (rc=%s): %s",
            repo, number, proc.returncode, (proc.stderr or "").strip()[:200],
        )
        return None
    try:
        return json.loads(proc.stdout).get("headRefOid") or None
    except (ValueError, TypeError, AttributeError):
        logger.warning(
            "[pr_review_buttons] could not parse headRefOid for %s#%s", repo, number,
        )
        return None


def _looks_like_own_pr(stderr: str) -> bool:
    low = (stderr or "").lower()
    return any(m in low for m in _OWN_PR_MARKERS)


def _review_failed(proc: subprocess.CompletedProcess) -> bool:
    """True when ``gh pr review`` did not actually create the review.

    ``gh pr review`` is unreliable about its exit code: on a rejected review
    (notably "can not approve your own pull request") it prints the error to
    stderr but STILL EXITS 0. So a clean exit is not sufficient proof of
    success — we must also treat a "failed to create review" / GraphQL error on
    stderr as a failure, or we'd report "posted" while nothing landed.
    """
    if proc.returncode != 0:
        return True
    low = (proc.stderr or "").lower()
    return "failed to create review" in low or "graphql:" in low


def submit_review(
    repo: str,
    number: Number,
    event: str,
    body: str,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> ReviewResult:
    """Submit a PR review via ``gh pr review``.

    ``event`` is one of ``approve`` / ``request-changes`` / ``comment``. If the
    PR is authored by the acting gh user, the post is REFUSED (``own_pr=True``,
    ``ok=False``) rather than posted — the operator does not want Hermes acting
    on its own PRs, and nothing is written to GitHub. Own-PRs are normally
    filtered out at scan time; this is the defensive second layer. Any other
    failure surfaces its stderr in ``ReviewResult.detail``.

    Note: ``gh pr review`` can exit 0 even when it failed to create the review,
    so success is judged by :func:`_review_failed` (rc AND stderr), not rc alone.
    """
    flag = _EVENT_FLAG.get(event)
    if flag is None:
        return ReviewResult(ok=False, detail=f"unknown event: {event!r}")

    proc = run(
        ["gh", "pr", "review", str(number), "-R", repo, flag, "--body", body],
        capture_output=True,
        text=True,
    )
    if not _review_failed(proc):
        return ReviewResult(ok=True, detail="posted")

    # Own-PR rejection (checked on stderr regardless of exit code): refuse to
    # post anything. No comment fallback — the operator does not want Hermes to
    # act on its own PRs.
    if _looks_like_own_pr(proc.stderr):
        return ReviewResult(
            ok=False,
            own_pr=True,
            detail="skipped: cannot review your own PR",
        )

    return ReviewResult(ok=False, detail=(proc.stderr or "gh pr review failed").strip())
