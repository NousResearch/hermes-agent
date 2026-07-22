"""Installer/bootstrap must refuse to destroy local-only commits.

Bootstrap paths retain a reset fallback for a branch proven to carry no local
commits. A managed checkout whose target branch carries commits that are not
on the remote must instead name those commits, print a literal ``fix:``
command, restore its autostash, and abort (exit 4) before any reset can run.

These tests execute the real installers' repository stage against isolated
temporary bare-remote/clone repositories (no source-text assertions).
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


def _make_managed_checkout(tmp_path: Path) -> Path:
    """Create a managed checkout tracking a local bare remote named origin."""
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
    return managed


def _advance_remote(tmp_path: Path, message: str = "upstream advance") -> str:
    """Push a new commit to the bare remote from a scratch clone; return SHA."""
    upstream = tmp_path / f"upstream-{message.replace(' ', '-')}"
    _git(tmp_path, "clone", "--branch", "main", str(tmp_path / "origin.git"), str(upstream))
    (upstream / "remote.txt").write_text(f"{message}\n", encoding="utf-8")
    _git(upstream, "add", "remote.txt")
    _git(upstream, "commit", "-m", message)
    _git(upstream, "push", "origin", "main")
    return _git(upstream, "rev-parse", "HEAD").stdout.strip()


def _commit_local(managed: Path, message: str = "carried local patch") -> str:
    (managed / "local.txt").write_text("local\n", encoding="utf-8")
    _git(managed, "add", "local.txt")
    _git(managed, "commit", "-m", message)
    return _git(managed, "rev-parse", "HEAD").stdout.strip()


def _run_install_sh_repository_stage(
    tmp_path: Path, managed: Path
) -> subprocess.CompletedProcess:
    env = os.environ | {
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_INSTALL_DIR": str(managed),
    }
    return subprocess.run(
        ["bash", str(INSTALL_SH), "--stage", "repository", "--non-interactive"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def _run_install_ps1_repository_stage(
    tmp_path: Path, managed: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
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


def _assert_local_commits_preserved_and_abort(
    managed: Path,
    output: str,
    pre_update_head: str,
    local_message: str,
) -> None:
    # The abort must name the endangered commits and carry a fix: line.
    assert "local commit(s)" in output
    assert local_message in output
    assert "fix:" in output
    assert "rebase origin/main" in output
    # Nothing was reset: HEAD and history are untouched.
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == pre_update_head
    assert local_message in _git(managed, "log", "--format=%s").stdout.splitlines()


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)
def test_install_sh_aborts_before_reset_when_branch_carries_local_commits(
    tmp_path: Path,
) -> None:
    managed = _make_managed_checkout(tmp_path)
    local_message = "carried local patch"
    pre_update_head = _commit_local(managed, local_message)
    _advance_remote(tmp_path)

    result = _run_install_sh_repository_stage(tmp_path, managed)

    assert result.returncode == 4, (result.stdout, result.stderr)
    _assert_local_commits_preserved_and_abort(
        managed, result.stdout + result.stderr, pre_update_head, local_message
    )


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)
def test_install_sh_abort_restores_autostash(tmp_path: Path) -> None:
    """Dirty worktree + carried commit: the abort must restore the autostash."""
    managed = _make_managed_checkout(tmp_path)
    local_message = "carried local patch"
    pre_update_head = _commit_local(managed, local_message)
    _advance_remote(tmp_path)
    (managed / "tracked.txt").write_text("dirty local edit\n", encoding="utf-8")
    (managed / "untracked.txt").write_text("untracked survives\n", encoding="utf-8")

    result = _run_install_sh_repository_stage(tmp_path, managed)

    assert result.returncode == 4, (result.stdout, result.stderr)
    _assert_local_commits_preserved_and_abort(
        managed, result.stdout + result.stderr, pre_update_head, local_message
    )
    # The autostash was applied back and dropped: dirty + untracked state
    # survives and no stash entry is left behind.
    assert (managed / "tracked.txt").read_text(encoding="utf-8") == "dirty local edit\n"
    assert (managed / "untracked.txt").read_text(encoding="utf-8") == "untracked survives\n"
    assert _git(managed, "stash", "list").stdout.strip() == ""


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)
def test_install_sh_fast_forwards_when_no_local_commits(tmp_path: Path) -> None:
    """No carried commits: the update stage fast-forwards and succeeds."""
    managed = _make_managed_checkout(tmp_path)
    remote_sha = _advance_remote(tmp_path)

    result = _run_install_sh_repository_stage(tmp_path, managed)

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == remote_sha


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)
def test_install_sh_guards_target_branch_from_other_checkout(tmp_path: Path) -> None:
    """A different checked-out branch must not hide main's carried commits."""
    managed = _make_managed_checkout(tmp_path)
    local_message = "main-only permanent patch"
    _commit_local(managed, local_message)
    main_head = _git(managed, "rev-parse", "main").stdout.strip()
    _git(managed, "checkout", "-b", "side-branch")
    _advance_remote(tmp_path)

    result = _run_install_sh_repository_stage(tmp_path, managed)

    assert result.returncode == 4, (result.stdout, result.stderr)
    output = result.stdout + result.stderr
    assert local_message in output
    assert "fix:" in output
    # main still carries the local commit; nothing was reset.
    assert _git(managed, "rev-parse", "main").stdout.strip() == main_head


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or POWERSHELL is None,
    reason="needs git and PowerShell",
)
def test_install_ps1_aborts_before_reset_when_branch_carries_local_commits(
    tmp_path: Path,
) -> None:
    managed = _make_managed_checkout(tmp_path)
    local_message = "carried local patch"
    pre_update_head = _commit_local(managed, local_message)
    _advance_remote(tmp_path)

    result = _run_install_ps1_repository_stage(tmp_path, managed)

    assert result.returncode != 0, (result.stdout, result.stderr)
    _assert_local_commits_preserved_and_abort(
        managed, result.stdout + result.stderr, pre_update_head, local_message
    )


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or POWERSHELL is None,
    reason="needs git and PowerShell",
)
def test_install_ps1_abort_restores_autostash(tmp_path: Path) -> None:
    managed = _make_managed_checkout(tmp_path)
    local_message = "carried local patch"
    pre_update_head = _commit_local(managed, local_message)
    _advance_remote(tmp_path)
    (managed / "tracked.txt").write_text("dirty local edit\n", encoding="utf-8")
    (managed / "untracked.txt").write_text("untracked survives\n", encoding="utf-8")

    result = _run_install_ps1_repository_stage(tmp_path, managed)

    assert result.returncode != 0, (result.stdout, result.stderr)
    _assert_local_commits_preserved_and_abort(
        managed, result.stdout + result.stderr, pre_update_head, local_message
    )
    assert (managed / "tracked.txt").read_text(encoding="utf-8") == "dirty local edit\n"
    assert (managed / "untracked.txt").read_text(encoding="utf-8") == "untracked survives\n"
    assert _git(managed, "stash", "list").stdout.strip() == ""


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("git") is None or POWERSHELL is None,
    reason="needs git and PowerShell",
)
def test_install_ps1_fast_forwards_when_no_local_commits(tmp_path: Path) -> None:
    managed = _make_managed_checkout(tmp_path)
    remote_sha = _advance_remote(tmp_path)

    result = _run_install_ps1_repository_stage(tmp_path, managed)

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == remote_sha
