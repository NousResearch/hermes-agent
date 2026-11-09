"""Tests for plugins_cmd._resolve_git_executable: which git plugin git operations run."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def resolve_git():
    from hermes_cli import plugins_cmd as pc

    pc._resolve_git_executable.cache_clear()
    yield pc._resolve_git_executable
    pc._resolve_git_executable.cache_clear()


def _stub_executable(path: Path, body: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _pm_reports_git(monkeypatch, binary: Path | None) -> None:
    import pm
    from types import SimpleNamespace

    installed = None if binary is None else SimpleNamespace(path=binary.parent.parent, version="pinned", binary=binary)
    monkeypatch.setattr(pm, "installed_package", lambda name, **_kw: installed if name == "git" else None)


@pytest.mark.parametrize("pm_has_git", [True, False], ids=["pm-git", "no-pm-git"])
def test_plugin_git_prefers_pm_pinned_git_over_path(resolve_git, tmp_path, monkeypatch, pm_has_git):
    """Plugin git operations run the git PM pinned for this install, not one the user put on PATH;
    with no PM git, the PATH git still answers."""
    suffix = ".exe" if os.name == "nt" else ""
    path_git = _stub_executable(tmp_path / "user-bin" / f"git{suffix}")
    pinned = _stub_executable(tmp_path / "tools" / "git-2.53.0+3-win32-x64" / "cmd" / f"git{suffix}")
    monkeypatch.setenv("PATH", str(path_git.parent))
    _pm_reports_git(monkeypatch, pinned if pm_has_git else None)
    assert Path(shutil.which("git") or "") == path_git

    assert Path(resolve_git() or "") == (pinned if pm_has_git else path_git)


@pytest.mark.platforms("posix")
def test_git_url_plugin_clone_runs_pm_pinned_git(resolve_git, tmp_path, monkeypatch):
    """A git-url install clones with PM's git even when PATH leads with a different git."""
    import subprocess as sp

    from hermes_cli.plugins_cmd import _clone_plugin_repo

    real_git = shutil.which("git")
    assert real_git, "the clone test needs a real git"
    origin = tmp_path / "origin"
    origin.mkdir()
    for args in (["init", "-q", "-b", "main"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty", "-qm", "init"]):
        sp.run([real_git, *args], cwd=origin, check=True, capture_output=True)
    head = sp.run([real_git, "rev-parse", "HEAD"], cwd=origin, check=True, capture_output=True,
                  text=True).stdout.strip()
    decoy = _stub_executable(tmp_path / "user-bin" / "git", "#!/bin/sh\necho 'the PATH git ran' >&2\nexit 1\n")
    monkeypatch.setenv("PATH", f"{decoy.parent}{os.pathsep}{os.environ['PATH']}")
    _pm_reports_git(monkeypatch, Path(real_git))

    assert _clone_plugin_repo(tmp_path / "clone", origin.as_uri(), None) == head
