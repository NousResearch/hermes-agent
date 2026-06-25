"""Regression: installer autostash restore must never leave conflict markers.

A ``git stash apply`` during the installer's update path can conflict with the
freshly checked-out commit. The old code aborted on that conflict but left
``<<<<<<< Updated upstream`` / ``>>>>>>> Stashed changes`` markers in tracked
source, so the next backend boot crashed at import time with
``SyntaxError: invalid syntax`` (e.g. ``utils.py`` line 1). A previously
corrupted tree was also re-stashed and re-applied, perpetuating the markers
across runs -- which is why the install "kept failing every time".

Both installers must, on any conflict, revert the working tree to the updated
commit (so the checkout is always bootable) while PRESERVING the user's changes
in the stash for manual, conflict-aware recovery.
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
    """Pull the autostash *restore* if-block from install.sh's update_repo()."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    m = re.search(
        r'if \[ -n "\$autostash_ref" \]; then.*?\n            fi\n',
        text,
        re.DOTALL,
    )
    assert m is not None, "autostash restore block not found in install.sh"
    return m.group(0)


def _make_conflicting_stash_repo(repo: Path) -> str:
    """Leave ``repo`` with a stash@{0} that conflicts with HEAD on utils.py.

    Mirrors the installer state right before the restore step: local edits were
    stashed, then the checkout advanced the same line, so ``git stash apply``
    will conflict. Returns the HEAD ("upstream") content a correct recovery
    must restore.
    """
    _git(repo, "init")
    (repo / "utils.py").write_text("v1\n")
    _git(repo, "add", "utils.py")
    _git(repo, "commit", "-m", "base")

    # Local edit, stashed (the installer's pre-update autostash).
    (repo / "utils.py").write_text("local change\n")
    _git(
        repo,
        "stash",
        "push",
        "--include-untracked",
        "-m",
        "hermes-install-autostash-test",
    )

    # "Upstream" advances the same line -> stash apply will conflict.
    (repo / "utils.py").write_text("upstream change\n")
    _git(repo, "add", "utils.py")
    _git(repo, "commit", "-m", "upstream")
    return "upstream change\n"


@pytest.mark.live_system_guard_bypass  # runs against a dedicated throwaway repo
def test_install_sh_autostash_conflict_reverts_and_preserves_stash(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    upstream = _make_conflicting_stash_repo(repo)

    block = _extract_restore_block()
    script = (
        "set -e\n"
        'log_info() { echo "INFO: $*"; }\n'
        'log_warn() { echo "WARN: $*"; }\n'
        'log_error() { echo "ERROR: $*"; }\n'
        "run() {\n"
        '  local autostash_ref="stash@{0}"\n'
        f"{block}"
        "}\n"
        "run\n"
        "echo BLOCK_OK\n"
    )
    res = subprocess.run(
        ["bash", "-c", script], cwd=repo, capture_output=True, text=True
    )

    # A restore conflict must NOT abort the install (the old code `exit 1`-ed).
    assert res.returncode == 0, res.stderr
    assert "BLOCK_OK" in res.stdout
    assert "Reverting the working tree" in res.stdout

    # The working tree is reverted to the bootable upstream source, with no
    # conflict markers left behind.
    content = (repo / "utils.py").read_text()
    assert content == upstream, content
    assert "<<<<<<<" not in content and ">>>>>>>" not in content

    # The user's local changes survive in the stash for manual recovery -- the
    # conflict path must never drop the stash.
    assert _git(repo, "stash", "list").stdout.strip(), "stash must be preserved"


@pytest.mark.live_system_guard_bypass
def test_install_sh_autostash_clean_apply_still_restores(tmp_path: Path) -> None:
    """A non-conflicting restore must still apply and drop the stash (the fix
    must not regress the happy path)."""
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _git(repo, "init")
    (repo / "utils.py").write_text("v1\n")
    (repo / "other.py").write_text("base\n")
    _git(repo, "add", "utils.py", "other.py")
    _git(repo, "commit", "-m", "base")

    # Local edit to other.py, stashed; upstream advances a DIFFERENT file, so
    # the apply is clean.
    (repo / "other.py").write_text("local edit\n")
    _git(repo, "stash", "push", "-m", "hermes-install-autostash-test")
    (repo / "utils.py").write_text("upstream change\n")
    _git(repo, "add", "utils.py")
    _git(repo, "commit", "-m", "upstream")

    block = _extract_restore_block()
    script = (
        "set -e\n"
        'log_info() { echo "INFO: $*"; }\n'
        'log_warn() { echo "WARN: $*"; }\n'
        'log_error() { echo "ERROR: $*"; }\n'
        "run() {\n"
        '  local autostash_ref="stash@{0}"\n'
        f"{block}"
        "}\n"
        "run\n"
        "echo BLOCK_OK\n"
    )
    res = subprocess.run(
        ["bash", "-c", script], cwd=repo, capture_output=True, text=True
    )

    assert res.returncode == 0, res.stderr
    assert "BLOCK_OK" in res.stdout
    # Local edit was restored on top of the update ...
    assert (repo / "other.py").read_text() == "local edit\n"
    assert (repo / "utils.py").read_text() == "upstream change\n"
    # ... and the stash was dropped (clean apply consumes it).
    assert _git(repo, "stash", "list").stdout.strip() == "", (
        "a clean apply must drop the stash"
    )


def test_install_ps1_autostash_conflict_reverts_not_throws() -> None:
    """install.ps1's restore path must revert to a bootable tree on conflict and
    keep the stash, instead of throwing and leaving conflict markers on disk.

    Asserted on source (CI runs on Linux; install.ps1 can't execute there).
    """
    text = INSTALL_PS1.read_text(encoding="utf-8")
    idx = text.index('Write-Info "Restoring local changes..."')
    region = text[idx : idx + 2500]

    # On conflict it reverts the working tree to the checked-out commit.
    assert "reset --hard HEAD" in region, (
        "the conflict path must revert the working tree to a bootable state"
    )
    # The old unconditional abort is gone -- a restore conflict must not fail
    # the whole install and leave markers on disk.
    assert "git stash apply failed after update" not in text, (
        "the old throw-on-conflict must be replaced by a revert"
    )
    # Conflict detection is git's own exit code, NOT string-sniffing the working
    # tree for `<<<<<<<` markers (that band-aid was removed as a code smell).
    assert "grep" not in region.lower(), (
        "conflict detection must rely on git's exit code, not marker grepping"
    )


def test_install_sh_autostash_conflict_no_unconditional_exit() -> None:
    """Source contract: install.sh's restore block must not `exit 1` on a
    restore conflict (it must revert + preserve the stash instead), and must
    detect the conflict via git's exit code rather than marker-grepping."""
    block = _extract_restore_block()
    assert "git reset --hard HEAD" in block
    assert "exit 1" not in block, (
        "a restore conflict must not abort the install"
    )
    assert "grep" not in block.lower(), (
        "conflict detection must rely on git's exit code, not marker grepping"
    )
