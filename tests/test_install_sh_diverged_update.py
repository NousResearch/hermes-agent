"""Regression tests for install.sh updates from a diverged managed clone.

Issue #53257 showed ``scripts/install.sh`` aborting during updates when the
managed checkout had both local-only and remote-only commits. ``git pull
--ff-only`` cannot resolve that state, so the installer must detect the
divergence after ``git fetch`` and hard-reset the managed clone to
``origin/$BRANCH`` instead of exiting with Git's fast-forward error.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_clone_repo_update_block() -> str:
    text = INSTALL_SH.read_text()
    match = re.search(
        r"(?P<block>git remote set-branches origin \"\$BRANCH\".*?git pull --ff-only origin \"\$BRANCH\".*?fi)",
        text,
        re.DOTALL,
    )
    assert match is not None, (
        "Could not locate the managed-install update block in scripts/install.sh"
    )
    return match["block"]


def test_install_script_detects_diverged_history_before_pull() -> None:
    block = _extract_clone_repo_update_block()

    assert 'git rev-list --left-right --count HEAD...origin/$BRANCH' in block
    assert 'if [ "$local_ahead" -gt 0 ] && [ "$local_behind" -gt 0 ]; then' in block
    assert 'git reset --hard "origin/$BRANCH"' in block


def test_install_script_resets_diverged_clone_instead_of_aborting() -> None:
    block = _extract_clone_repo_update_block()

    rev_idx = block.find('git rev-list --left-right --count HEAD...origin/$BRANCH')
    reset_idx = block.find('git reset --hard "origin/$BRANCH"')
    pull_idx = block.find('git pull --ff-only origin "$BRANCH"')

    assert rev_idx != -1, "expected divergence probe in clone_repo()"
    assert reset_idx != -1, "expected reset fallback in clone_repo()"
    assert pull_idx != -1, "expected ff-only pull path in clone_repo()"
    assert rev_idx < reset_idx, "divergence probe must run before the reset fallback"
    assert rev_idx < pull_idx, "divergence probe must run before the ff-only pull"
