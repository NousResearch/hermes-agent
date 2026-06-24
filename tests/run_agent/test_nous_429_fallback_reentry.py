"""Regression guard: a genuine Nous 429 must re-enter the retry loop so the
top-of-loop Nous rate-limit guard can activate the fallback chain.

Bug (found in the #44061 audit): the genuine-rate-limit branch in
``agent/conversation_loop.py`` set ``retry_count = max_retries`` then
``continue``-d, intending the top-of-loop guard to "handle fallback or bail
cleanly".  But the loop condition is ``while retry_count < max_retries`` —
setting retry_count equal to max_retries makes the condition False
immediately, so the guard NEVER runs.  No fallback activation, no clean
rate-limit message: the turn dies with the generic retry-exhaustion error.

The fix sets ``retry_count = max(0, max_retries - 1)`` so the loop body runs
exactly once more: the guard sees the breaker state recorded by
``record_nous_rate_limit()`` moments earlier and either activates a fallback
provider (resetting retry_count) or returns the explicit rate-limit failure.
"""
from __future__ import annotations

import inspect
import re


def _loop_reenters(retry_count: int, max_retries: int) -> bool:
    """Mirror of the ``while retry_count < max_retries`` loop condition."""
    return retry_count < max_retries


class TestGenuineNous429ReentersLoop:
    """The assignment used by the genuine-429 branch must leave the loop
    condition True so the top-of-loop guard gets a chance to run."""

    def test_fixed_assignment_reenters_for_typical_max_retries(self):
        for max_retries in (1, 2, 3, 5, 10):
            retry_count = max(0, max_retries - 1)
            assert _loop_reenters(retry_count, max_retries), (
                f"max_retries={max_retries}: guard would never run"
            )

    def test_buggy_assignment_never_reenters(self):
        """Documents the bug shape: retry_count = max_retries exits the
        loop immediately, skipping the fallback guard."""
        for max_retries in (1, 2, 3, 5, 10):
            retry_count = max_retries
            assert not _loop_reenters(retry_count, max_retries)


class TestSourceUsesReentrantAssignment:
    """Belt-and-suspenders: the production source must use the re-entrant
    form in the genuine-Nous-429 branch.  Protects against an accidental
    revert (e.g. a stale-branch merge resolving in favor of the old code)."""

    def test_genuine_branch_does_not_skip_to_max_retries(self):
        from agent import conversation_loop

        src = inspect.getsource(conversation_loop)
        # Locate the genuine-rate-limit branch.
        match = re.search(
            r"if _genuine_nous_rate_limit:\n(?:.*\n)*?\s*continue\n",
            src,
        )
        # There are two `if _genuine_nous_rate_limit` sites (record + branch);
        # the regex above finds the first block ending in `continue`, which is
        # the retry-count branch.
        assert match is not None, (
            "genuine-Nous-429 branch not found in conversation_loop — "
            "update this test if the branch was refactored"
        )
        block = match.group(0)
        assert "retry_count = max(0, max_retries - 1)" in block, (
            "genuine-Nous-429 branch must re-enter the retry loop "
            "(retry_count = max(0, max_retries - 1)); "
            "`retry_count = max_retries` makes the while condition False "
            "and the fallback guard never runs."
        )
        assert "retry_count = max_retries\n" not in block
