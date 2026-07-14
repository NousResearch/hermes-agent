"""Installer/bootstrap must refuse to destroy local-only commits.

Bootstrap paths retain a reset fallback for a branch proven to carry no local
commits. A diverged managed checkout with carried commits must instead name
those commits, print a literal ``fix:`` command, restore its autostash, and
abort before reset is reachable.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _extract_install_sh_update_block() -> str:
    text = INSTALL_SH.read_text()
    match = re.search(
        r"(?P<block>git fetch origin \"\$BRANCH\".*?fi\n\n            if \[ -n \"\$autostash_ref\" \])",
        text,
        re.DOTALL,
    )
    assert match is not None, "managed-install update block not found in install.sh"
    return match["block"]


def _extract_install_ps1_branch_update_block() -> str:
    text = INSTALL_PS1.read_text()
    start = text.find("# Check the target branch even when another branch is")
    end = text.find("# Default to restoring so work is never silently dropped.", start)
    assert start != -1 and end != -1, "branch update block not found in install.ps1"
    return text[start:end]


def test_install_sh_aborts_with_fix_before_reset_for_local_commits() -> None:
    block = _extract_install_sh_update_block()

    assert 'git rev-list --reverse "origin/$BRANCH..refs/heads/$BRANCH"' in block
    assert 'git show -s --format=' in block
    assert "fix: git -C $INSTALL_DIR rebase origin/$BRANCH" in block
    assert "exit 4" in block
    assert 'git stash apply "$autostash_ref"' in block
    assert 'git merge --ff-only "origin/$BRANCH"' in block
    assert 'git reset --hard "origin/$BRANCH"' in block

    guard_idx = block.find('git rev-list --reverse "origin/$BRANCH..refs/heads/$BRANCH"')
    abort_idx = block.find("exit 4")
    reset_idx = block.find('git reset --hard "origin/$BRANCH"')
    assert guard_idx != -1 and abort_idx != -1 and reset_idx != -1
    assert guard_idx < abort_idx < reset_idx


def test_install_ps1_aborts_with_fix_before_reset_for_local_commits() -> None:
    block = _extract_install_ps1_branch_update_block()

    assert 'rev-list --reverse "origin/$Branch..refs/heads/$Branch"' in block
    assert 'show -s --format="  • %h %s"' in block
    assert "fix: git -C" in block
    assert "rebase origin/$Branch" in block
    assert "installer update refused to discard local-only commits" in block
    assert "stash apply $autostashRef" in block
    assert 'merge --ff-only "origin/$Branch"' in block
    assert 'reset --hard "origin/$Branch"' in block

    guard_idx = block.find('rev-list --reverse "origin/$Branch..refs/heads/$Branch"')
    abort_idx = block.find("installer update refused to discard local-only commits")
    reset_idx = block.find('reset --hard "origin/$Branch"')
    assert guard_idx != -1 and abort_idx != -1 and reset_idx != -1
    assert guard_idx < abort_idx < reset_idx
