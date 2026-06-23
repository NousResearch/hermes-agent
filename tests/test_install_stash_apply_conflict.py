"""Regression: installer must not leave conflict markers after a stash apply.

The installer's update path stashes local changes, pulls, then ``git stash
apply``s the user's work back on top. When that apply *conflicts*, git writes
``<<<<<<< Updated upstream`` / ``>>>>>>> Stashed changes`` markers into the
tracked source files. If the installer leaves those markers behind, Hermes
becomes completely unrunnable -- a poisoned ``toolsets.py`` raises a
``SyntaxError`` on import and the gateway never starts (#46791, real report:
``File ".../toolsets.py", line 1  <<<<<<< Updated upstream``).

The ``hermes update`` Python path (``_restore_stashed_changes``) already
guards this: on a conflicting apply it resets the working tree to the
freshly-pulled HEAD and keeps the stash for manual recovery. Both installer
scripts must do the same -- never leave a bricked checkout behind.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _extract_restore_block() -> str:
    """Pull the ``restore_now`` if/else block out of install.sh's update path."""
    text = INSTALL_SH.read_text()
    m = re.search(
        r'if \[ "\$restore_now" = "yes" \]; then.*?\n                fi\n',
        text,
        re.DOTALL,
    )
    assert m is not None, "restore_now block not found in install.sh"
    return m.group(0)


def _run_restore_block(repo: Path) -> subprocess.CompletedProcess:
    block = _extract_restore_block()
    script = (
        "set -e\n"
        'log_info() { echo "INFO: $*"; }\n'
        'log_warn() { echo "WARN: $*"; }\n'
        'log_error() { echo "ERROR: $*"; }\n'
        'restore_now="yes"\n'
        'autostash_ref="stash@{0}"\n'
        "run() {\n"
        f"{block}"
        "}\n"
        "run\n"
        "echo BLOCK_OK\n"
    )
    return subprocess.run(
        ["bash", "-c", script], cwd=repo, capture_output=True, text=True
    )


@pytest.mark.live_system_guard_bypass  # runs against a dedicated throwaway repo
def test_install_sh_conflicting_apply_resets_tree_keeps_stash(tmp_path: Path) -> None:
    """A conflicting ``git stash apply`` must leave a clean, runnable tree
    (no conflict markers) while preserving the user's changes in the stash."""
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _git(repo, "init")
    (repo / "toolsets.py").write_text("x = 1\n")
    _git(repo, "add", "toolsets.py")
    _git(repo, "commit", "-m", "base")

    # User has a local edit (the autostash) ...
    (repo / "toolsets.py").write_text("x = 2  # local\n")
    _git(repo, "stash", "push", "--include-untracked", "-m", "hermes-install-autostash")

    # ... and the pull advanced HEAD with a *conflicting* change on the same line.
    (repo / "toolsets.py").write_text("x = 99  # upstream\n")
    _git(repo, "add", "toolsets.py")
    _git(repo, "commit", "-m", "upstream change")

    # Sanity: applying the stash now genuinely conflicts and writes markers.
    pre = _git(repo, "stash", "apply", "stash@{0}", check=False)
    assert pre.returncode != 0 or "<<<<<<<" in (repo / "toolsets.py").read_text(), (
        "test setup failed to produce a conflicting apply"
    )
    _git(repo, "reset", "--hard", "HEAD")  # undo the sanity apply; stash stays

    res = _run_restore_block(repo)

    assert res.returncode == 0, res.stderr  # must NOT abort the install
    assert "BLOCK_OK" in res.stdout
    assert "Working tree reset to clean state" in res.stdout

    content = (repo / "toolsets.py").read_text()
    assert "<<<<<<<" not in content and ">>>>>>>" not in content, (
        "conflict markers must not be left in source files"
    )
    assert content == "x = 99  # upstream\n", "tree must match the pulled HEAD"
    assert _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.strip() == ""
    # The user's work is never lost — it stays recoverable in the stash.
    assert _git(repo, "stash", "list").stdout.strip(), (
        "stash must be preserved for manual recovery on a conflicting apply"
    )


@pytest.mark.live_system_guard_bypass  # runs against a dedicated throwaway repo
def test_install_sh_clean_apply_restores_and_drops_stash(tmp_path: Path) -> None:
    """The happy path is unchanged: a non-conflicting apply restores the user's
    work and drops the stash."""
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _git(repo, "init")
    (repo / "toolsets.py").write_text("x = 1\n")
    (repo / "other.py").write_text("y = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    # Local edit to other.py only ...
    (repo / "other.py").write_text("y = 2  # local\n")
    _git(repo, "stash", "push", "--include-untracked", "-m", "hermes-install-autostash")

    # ... pull advanced a *different* file, so the apply does not conflict.
    (repo / "toolsets.py").write_text("x = 99  # upstream\n")
    _git(repo, "add", "toolsets.py")
    _git(repo, "commit", "-m", "upstream change")

    res = _run_restore_block(repo)

    assert res.returncode == 0, res.stderr
    assert "restored on top of the updated codebase" in res.stdout
    assert (repo / "other.py").read_text() == "y = 2  # local\n"
    assert (repo / "toolsets.py").read_text() == "x = 99  # upstream\n"
    assert _git(repo, "stash", "list").stdout.strip() == "", (
        "a clean apply must drop the stash"
    )


def test_install_ps1_resets_tree_on_conflicting_apply_source_order() -> None:
    """install.ps1's restore path must reset the tree and keep the stash on a
    conflicting apply (pwsh isn't always available in CI, so assert the source
    contract instead of executing it)."""
    text = INSTALL_PS1.read_text()
    apply_idx = text.index("stash apply $autostashRef")
    # Conflicts are detected via unmerged paths as well as a non-zero exit ...
    diff_idx = text.index("diff --name-only --diff-filter=U", apply_idx)
    # ... and on conflict the working tree is reset to a clean, runnable state.
    reset_idx = text.index("reset --hard HEAD", apply_idx)
    assert apply_idx < diff_idx < reset_idx, (
        "install.ps1 must check for unmerged paths and reset --hard after apply"
    )
    # The destructive `throw` that left a bricked tree must be gone.
    assert "git stash apply failed after update" not in text, (
        "install.ps1 must not throw and leave conflict markers behind"
    )
