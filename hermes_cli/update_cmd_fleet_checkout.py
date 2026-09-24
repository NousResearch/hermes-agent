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
    """True when ``sha`` is an ancestor of (or equal to) the code this checkout runs; False on any
    probe failure.

    Fail-closed on purpose: an unknown ancestry is not evidence that the fleet serves the update.

    A Docker/Cloud image has no ``.git``; its identity is the baked build stamp
    (``build_info.get_code_identity`` → ``source == "build-file"``). There is no history to walk, so
    "contained" collapses to "equal to the stamp" — without this the probe was always False on an
    image and the pending-restart catch-up printed "every gateway serves the checkout" and "still
    off the checkout code" in the same breath.
    """
    from hermes_cli.build_info import get_code_identity
    from hermes_cli.update_cmd import _m
    identity = get_code_identity() or {}
    if identity.get("source") == "build-file":
        stamped = str(identity.get("sha") or "")
        return bool(stamped) and (stamped == sha or stamped.startswith(sha) or sha.startswith(stamped))
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            cwd=_m().PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("Checkout ancestry probe for %s failed: %s", sha[:10], exc)
        return False
