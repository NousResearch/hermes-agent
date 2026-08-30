"""Behavioral installer regressions for divergent managed clones."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = ROOT / "scripts" / "install.sh"
INSTALL_PS1 = ROOT / "scripts" / "install.ps1"
GIT = ["git", "-c", "user.name=Hermes Test", "-c", "user.email=test@example.invalid"]


@pytest.fixture(autouse=True)
def _require_explicit_git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match CI hosts where Git cannot synthesize a committer identity."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.useConfigOnly")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    home = repo.parent / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        GIT + list(args),
        cwd=repo,
        env={**os.environ, "HOME": str(home), "HERMES_HOME": str(home / ".hermes")},
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _commit(repo: Path, path: str, text: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _managed_clone(tmp_path: Path) -> tuple[Path, Path, str]:
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    install = tmp_path / "install"
    _git(tmp_path, "init", "--bare", str(origin))
    seed.mkdir()
    _git(seed, "init", "-b", "main")
    base = _commit(seed, "tracked.txt", "base\n", "base")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-u", "origin", "main")
    _git(tmp_path, "clone", "--branch", "main", str(origin), str(install))
    return seed, install, base


def _rewrite_remote(seed: Path, base: str, *, conflict: bool = False) -> str:
    _git(seed, "reset", "--hard", base)
    remote = _commit(
        seed,
        "tracked.txt" if conflict else "replacement.txt",
        "remote\n",
        "rewritten upstream",
    )
    _git(seed, "push", "--force", "origin", "main")
    return remote


def _run_sh(install: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(INSTALL_SH),
            "--stage",
            "repository",
            "--branch",
            "main",
            "--dir",
            str(install),
            "--hermes-home",
            str(install.parent / "home"),
            "--non-interactive",
        ],
        cwd=install.parent,
        env={**os.environ, "HOME": str(install.parent / "home")},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _powershell() -> str | None:
    return next(
        (shutil.which(name) for name in ("pwsh", "powershell") if shutil.which(name)),
        None,
    )


def _run_ps(install: Path) -> subprocess.CompletedProcess[str]:
    host = _powershell()
    assert host is not None
    return subprocess.run(
        [
            host,
            "-NoProfile",
            "-File",
            str(INSTALL_PS1),
            "-Stage",
            "repository",
            "-Branch",
            "main",
            "-InstallDir",
            str(install),
            "-HermesHome",
            str(install.parent / "home"),
            "-NonInteractive",
        ],
        cwd=install.parent,
        env={**os.environ, "HOME": str(install.parent / "home")},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture(params=["sh", "ps"])
def run_installer(request) -> Callable[[Path], subprocess.CompletedProcess[str]]:
    if request.param == "ps":
        if _powershell() is None:
            pytest.skip("PowerShell is not available")
        return _run_ps
    return _run_sh


def _assert_clean_at(repo: Path, sha: str) -> None:
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == sha
    assert not _git(repo, "status", "--porcelain").stdout
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").stdout.strip())
    assert not (git_dir / "rebase-merge").exists()
    assert not (git_dir / "rebase-apply").exists()


def test_installer_preserves_only_genuine_local_commit(
    run_installer: Callable[[Path], subprocess.CompletedProcess[str]], tmp_path: Path
) -> None:
    seed, install, base = _managed_clone(tmp_path)
    discarded = _commit(seed, "obsolete.txt", "obsolete\n", "discarded upstream")
    _git(seed, "push", "origin", "main")
    _git(install, "pull", "--ff-only")
    _commit(install, "local.txt", "local\n", "genuine local commit")
    remote = _rewrite_remote(seed, base)

    result = run_installer(install)

    assert result.returncode == 0, result.stderr or result.stdout
    assert _git(install, "merge-base", "--is-ancestor", remote, "HEAD").returncode == 0
    assert _git(
        install, "log", "--format=%s", f"{remote}..HEAD"
    ).stdout.splitlines() == ["genuine local commit"]
    assert _git(install, "log", "--format=%an <%ae>", f"{remote}..HEAD").stdout.strip() == (
        "Hermes Test <test@example.invalid>"
    )
    assert not (install / "obsolete.txt").exists()
    assert discarded != remote


def test_installer_conflict_restores_original_head(
    run_installer: Callable[[Path], subprocess.CompletedProcess[str]], tmp_path: Path
) -> None:
    seed, install, base = _managed_clone(tmp_path)
    original = _commit(install, "tracked.txt", "local\n", "local conflict")
    _rewrite_remote(seed, base, conflict=True)

    result = run_installer(install)

    assert result.returncode != 0
    _assert_clean_at(install, original)
    assert (install / "tracked.txt").read_text(encoding="utf-8") == "local\n"


def test_installer_without_local_commits_tracks_rewritten_remote(
    run_installer: Callable[[Path], subprocess.CompletedProcess[str]], tmp_path: Path
) -> None:
    seed, install, base = _managed_clone(tmp_path)
    remote = _rewrite_remote(seed, base)

    result = run_installer(install)

    assert result.returncode == 0, result.stderr or result.stdout
    _assert_clean_at(install, remote)


def test_installer_fails_closed_without_reflog_evidence(
    run_installer: Callable[[Path], subprocess.CompletedProcess[str]], tmp_path: Path
) -> None:
    seed, install, base = _managed_clone(tmp_path)
    original = _commit(install, "local.txt", "local\n", "local")
    _rewrite_remote(seed, base)
    _git(install, "config", "core.logAllRefUpdates", "false")
    remote_logs = install / ".git" / "logs" / "refs" / "remotes"
    if remote_logs.exists():
        shutil.rmtree(remote_logs)

    result = run_installer(install)

    assert result.returncode != 0
    _assert_clean_at(install, original)
