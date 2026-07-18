"""Regression: installer autostash restore conflicts must not abort the run.

An interrupted/repeated managed install can leave local tracked edits in the
checkout. If upstream then changes the same lines, ``git stash apply`` conflicts
during the repository-update stage. Both installers must leave the stash intact,
reset the worktree clean, and complete the real repository stage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = next(
    (candidate for candidate in ("pwsh", "powershell") if shutil.which(candidate)),
    None,
)


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _make_conflicted_managed_checkout(tmp_path: Path) -> Path:
    """Create a managed checkout whose autostash conflicts with its origin."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init")
    (seed / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(seed, "add", "tracked.txt")
    _git(seed, "commit", "-m", "base")
    _git(seed, "branch", "-M", "main")

    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")

    managed = tmp_path / "hermes-agent"
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(managed))

    (managed / "tracked.txt").write_text("local edit\n", encoding="utf-8")

    upstream = tmp_path / "upstream"
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(upstream))
    (upstream / "tracked.txt").write_text("upstream edit\n", encoding="utf-8")
    _git(upstream, "commit", "-am", "upstream")
    _git(upstream, "push", "origin", "main")

    return managed


def _assert_conflict_was_recovered(repo: Path, output: str) -> None:
    assert "restoring local changes hit conflicts" in output
    assert "Conflicted files:" in output
    assert "tracked.txt" in output
    assert "Working tree reset to clean state." in output
    assert "Restore your changes later with: git stash apply stash@{0}" in output
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""
    assert _git(repo, "stash", "list").stdout.strip(), "stash must be preserved"
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "upstream edit\n"


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)
def test_install_sh_repository_stage_recovers_from_autostash_conflict(
    tmp_path: Path,
) -> None:
    managed = _make_conflicted_managed_checkout(tmp_path)
    env = os.environ | {
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_INSTALL_DIR": str(managed),
    }

    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--stage", "repository", "--non-interactive"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    _assert_conflict_was_recovered(managed, result.stdout)


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or POWERSHELL is None,
    reason="needs git and PowerShell",
)
def test_install_ps1_repository_stage_recovers_from_autostash_conflict(
    tmp_path: Path,
) -> None:
    managed = _make_conflicted_managed_checkout(tmp_path)
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-File",
            str(INSTALL_PS1),
            "-Stage",
            "repository",
            "-NonInteractive",
            "-InstallDir",
            str(managed),
            "-HermesHome",
            str(tmp_path / "hermes-home"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    _assert_conflict_was_recovered(managed, result.stdout)
