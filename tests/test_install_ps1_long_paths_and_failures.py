"""Windows installer checkout and caller-failure behavior."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = next(
    (candidate for candidate in ("pwsh", "powershell") if shutil.which(candidate)),
    None,
)

pytestmark = [
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(
        sys.platform != "win32" or shutil.which("git") is None or POWERSHELL is None,
        reason="needs Windows, Git for Windows, and PowerShell",
    ),
]


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        input=input,
        check=check,
        capture_output=True,
        text=True,
    )


def _git(
    cwd: Path,
    *args: str,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> str:
    result = _run(
        [
            "git",
            "-c",
            "user.name=Hermes Installer Test",
            "-c",
            "user.email=test@example.com",
            *args,
        ],
        cwd=cwd,
        env=env,
        input=input,
    )
    return result.stdout.strip()


def _powershell_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _make_long_path_remote(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Create two commits without materializing the long path in the seed repo."""
    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(remote))

    long_path = "/".join(
        [f"segment-{letter * 48}" for letter in "abcd"] + ["tracked.txt"]
    )
    index_path = tmp_path / "fixture.index"
    index_env = os.environ | {"GIT_INDEX_FILE": str(index_path)}

    pinned_blob = _git(
        tmp_path,
        "--git-dir",
        str(remote),
        "hash-object",
        "-w",
        "--stdin",
        input="pinned revision\n",
    )
    _git(tmp_path, "--git-dir", str(remote), "read-tree", "--empty", env=index_env)
    _git(
        tmp_path,
        "--git-dir",
        str(remote),
        "update-index",
        "--add",
        "--cacheinfo",
        "100644",
        pinned_blob,
        long_path,
        env=index_env,
    )
    pinned_tree = _git(tmp_path, "--git-dir", str(remote), "write-tree", env=index_env)
    pinned_commit = _git(
        tmp_path,
        "--git-dir",
        str(remote),
        "commit-tree",
        pinned_tree,
        "-m",
        "pinned revision",
    )

    tip_blob = _git(
        tmp_path,
        "--git-dir",
        str(remote),
        "hash-object",
        "-w",
        "--stdin",
        input="branch tip\n",
    )
    _git(
        tmp_path,
        "--git-dir",
        str(remote),
        "update-index",
        "--cacheinfo",
        "100644",
        tip_blob,
        long_path,
        env=index_env,
    )
    tip_tree = _git(tmp_path, "--git-dir", str(remote), "write-tree", env=index_env)
    tip_commit = _git(
        tmp_path,
        "--git-dir",
        str(remote),
        "commit-tree",
        tip_tree,
        "-p",
        pinned_commit,
        "-m",
        "branch tip",
    )
    _git(
        tmp_path, "--git-dir", str(remote), "update-ref", "refs/heads/main", tip_commit
    )
    _git(tmp_path, "--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main")
    return remote, long_path, pinned_commit, pinned_blob


def test_repository_stage_clones_and_pins_over_max_path_fixture(tmp_path: Path) -> None:
    remote, long_path, pinned_commit, pinned_blob = _make_long_path_remote(tmp_path)
    install_dir = tmp_path / "nested-install-root" / "hermes-agent"
    checkout_path = install_dir.joinpath(*long_path.split("/"))
    assert len(str(checkout_path)) > 260

    isolated_global_config = tmp_path / "global.gitconfig"
    discovered_git = Path(shutil.which("git")).resolve()
    git_exe = next(
        (
            candidate
            for parent in discovered_git.parents
            if (candidate := parent / "cmd" / "git.exe").is_file()
        ),
        discovered_git,
    )
    git_dir = str(git_exe.parent)
    git_root = git_exe.parent.parent
    child_path = os.pathsep.join(
        str(path)
        for path in (
            git_exe.parent,
            git_root / "mingw64" / "bin",
            git_root / "usr" / "bin",
        )
    )
    env = os.environ | {
        "COMSPEC": str(Path(os.environ["SYSTEMROOT"]) / "System32" / "cmd.exe"),
        "GIT_CONFIG_GLOBAL": str(isolated_global_config),
        "GIT_EXEC_PATH": _git(tmp_path, "--exec-path"),
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "PATH": child_path + os.pathsep + os.environ["PATH"],
    }
    command = "\n".join([
        f"$env:Path = {_powershell_literal(git_dir)} + ';' + $env:Path",
        f"Set-Alias -Name git -Value {_powershell_literal(git_exe)}",
        f". {_powershell_literal(INSTALL_PS1)}",
        f"$InstallDir = {_powershell_literal(install_dir)}",
        f"$HermesHome = {_powershell_literal(tmp_path / 'hermes-home')}",
        f"$RepoUrlSsh = {_powershell_literal(remote.as_uri())}",
        f"$RepoUrlHttps = {_powershell_literal(remote.as_uri())}",
        "$Branch = 'main'",
        f"$Commit = {_powershell_literal(pinned_commit)}",
        "$Tag = ''",
        "$ForceCommit = $true",
        "Install-Repository",
    ])
    result = _run(
        [POWERSHELL, "-NoProfile", "-Command", command],
        cwd=tmp_path,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(install_dir, "rev-parse", "HEAD") == pinned_commit
    assert _git(install_dir, "config", "--get", "core.longpaths") == "true"
    assert _git(install_dir, "hash-object", "--", long_path) == pinned_blob
    assert _git(install_dir, "status", "--porcelain") == ""


def _make_unusable_hermes_home(tmp_path: Path) -> Path:
    path = tmp_path / "hermes-home-is-a-file"
    path.write_text("not a directory", encoding="ascii")
    return path


def test_noninteractive_file_invocation_reports_failure_exit_code(
    tmp_path: Path,
) -> None:
    unusable_home = _make_unusable_hermes_home(tmp_path)
    result = _run(
        [
            POWERSHELL,
            "-NoProfile",
            "-File",
            str(INSTALL_PS1),
            "-NonInteractive",
            "-HermesHome",
            str(unusable_home),
        ],
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Installation failed:" in result.stdout


def test_invoke_expression_failure_keeps_caller_session_alive(tmp_path: Path) -> None:
    unusable_home = _make_unusable_hermes_home(tmp_path)
    command = "\n".join([
        f"$env:HERMES_HOME = {_powershell_literal(unusable_home)}",
        f"Get-Content -LiteralPath {_powershell_literal(INSTALL_PS1)} -Raw | Invoke-Expression",
        "Write-Output 'CALLER_SESSION_ALIVE'",
    ])
    result = _run(
        [POWERSHELL, "-NoProfile", "-Command", command],
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Installation failed:" in result.stdout
    assert "CALLER_SESSION_ALIVE" in result.stdout
