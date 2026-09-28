"""Checkout ancestry for the update-restart obligation (``update_cmd_fleet`` sibling).

The obligation records the SHA a pull landed on; the checkout may later sit past it by a carried
local commit (a cherry-picked hotfix). Whether that recorded SHA is still *contained* in HEAD is
the question these readers ask, so the live fleet can be held to the code on disk (#119367).
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def checkout_contains(sha: str) -> bool:
    """True when ``sha`` is an ancestor of (or equal to) the checkout HEAD; False on any probe failure.

    Fail-closed on purpose: an unknown ancestry is not evidence that the fleet serves the update.
    """
    from hermes_cli.update_cmd import _m
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            cwd=_m().PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("Checkout ancestry probe for %s failed: %s", sha[:10], exc)
        return False
