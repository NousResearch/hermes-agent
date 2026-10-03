"""Checkout ancestry for the update-restart obligation (``update_cmd_fleet`` sibling).

The obligation records the SHA a pull landed on; the checkout may later sit past it by a carried
local commit (a cherry-picked hotfix). Whether that recorded SHA is still *contained* in HEAD is
the question these readers ask, so the live fleet can be held to the code on disk (#119367).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Stamp-baked provenance per version_info: on a tree with NO ``.git`` any of these means
# there is no history to walk, so "contained" collapses to "equal to the stamp". "git" is
# live git; "local" is a stamp written FROM a git checkout (scripts/write_install_stamp
# resolves the commit from git), so it only collapses the same way once ``.git`` is gone —
# with ``.git`` present the ancestry walk below is strictly more accurate and covers HEAD
# sitting past the stamp by a carried hotfix (#119367). "unknown" carries no sha at all
# and stays fail-closed below.
_STAMPED_SOURCES = frozenset(
    {"build", "commit-build", "ci", "docker", "fallback", "local", "nix"}
)


def _has_git_history() -> bool:
    """True when the checkout root carries a ``.git`` — history a merge-base can walk."""
    from hermes_cli.update_cmd import _m
    return Path(_m().PROJECT_ROOT, ".git").exists()


def checkout_contains(sha: str) -> bool:
    """True when ``sha`` is an ancestor of (or equal to) the code this checkout runs; False
    on any probe failure.

    Fail-closed on purpose: an unknown ancestry is not evidence that the fleet serves the
    update — and that includes the identity reader itself raising (an import-stripped
    tree, a malformed environment): the read sits inside the same fail-closed guard as the
    git probe, honouring "False on any probe failure" for the whole function, not just the
    subprocess leg.

    A Docker/Cloud/Nix image has no ``.git``; its identity is the baked install stamp
    (``version_info.get_code_identity`` → a stamped ``source``). There is no history to
    walk, so "contained" collapses to "equal to the stamp" — without this the probe was
    always False on an image and the pending-restart catch-up printed "every gateway
    serves the checkout" and "still off the checkout code" in the same breath. A stamped
    tree that DOES carry ``.git`` (a "local"/"git" stamp on a live checkout) takes the
    ancestry walk instead: stamp equality answers only equality-or-prefix and would miss
    HEAD sitting past the recorded sha by a cherry-picked hotfix (#119367).
    """
    from hermes_cli.version_info import get_code_identity
    try:
        if not _has_git_history():
            identity = get_code_identity() or {}
            if identity.get("source") in _STAMPED_SOURCES:
                stamped = str(identity.get("sha") or "")
                return bool(stamped) and (stamped == sha or stamped.startswith(sha) or sha.startswith(stamped))
        from hermes_cli.update_cmd import _m
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            cwd=_m().PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("Checkout ancestry probe for %s failed: %s", sha[:10], exc)
        return False
