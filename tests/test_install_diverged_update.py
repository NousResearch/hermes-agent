"""Regression: installer/bootstrap must recover from diverged managed clones.

When ``~/.hermes/hermes-agent`` has local-only commits (or diverged history),
``git pull --ff-only`` fails with exit 128 and bootstrap aborts at the
repository stage. ``hermes update`` already resets to ``origin/$BRANCH`` in
that case; both installer scripts must do the same.

Fixes the bootstrap failure seen in #53257 and desktop update paths that run
``install.ps1`` / ``install.sh`` non-interactively.
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
        r"(?P<block>git checkout \"\$BRANCH\".*?fi\n\n            if \[ -n \"\$autostash_ref\" \])",
        text,
        re.DOTALL,
    )
    assert match is not None, "managed-install update block not found in install.sh"
    return match["block"]


def _extract_install_ps1_branch_update_block() -> str:
    text = INSTALL_PS1.read_text()
    match = re.search(
        r"(?P<block>git -c windows\.appendAtomically=false checkout \$Branch.*?elseif \(\$Tag\))",
        text,
        re.DOTALL,
    )
    assert match is not None, "branch update block not found in install.ps1"
    return match["block"]


def test_install_sh_resets_when_ff_only_pull_fails() -> None:
    block = _extract_install_sh_update_block()

    assert 'git pull --ff-only origin "$BRANCH"' in block
    assert 'git reset --hard "origin/$BRANCH"' in block
    assert "Fast-forward not possible" in block

    pull_idx = block.find('git pull --ff-only origin "$BRANCH"')
    reset_idx = block.find('git reset --hard "origin/$BRANCH"')
    assert pull_idx != -1 and reset_idx != -1
    assert pull_idx < reset_idx, "ff-only pull must be attempted before reset fallback"


def test_install_ps1_resets_when_ff_only_pull_fails() -> None:
    block = _extract_install_ps1_branch_update_block()

    assert "pull --ff-only origin $Branch" in block
    assert 'reset --hard "origin/$Branch"' in block
    assert "Fast-forward not possible" in block

    pull_idx = block.find("pull --ff-only origin $Branch")
    reset_idx = block.find('reset --hard "origin/$Branch"')
    assert pull_idx != -1 and reset_idx != -1
    assert pull_idx < reset_idx, "ff-only pull must be attempted before reset fallback"
