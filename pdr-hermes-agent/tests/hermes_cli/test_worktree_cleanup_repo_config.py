"""Exit-time worktree cleanup deletes the session branch without running repository hooks.

``cli._cleanup_worktree`` removes a clean, pushed worktree and then runs ``git branch -D``, which
runs the repository's ``reference-transaction`` hook under a plain environment. Real temporary
repository with real ``git``; ``core.hooksPath`` points at hooks that touch marker files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

_HOOKS = ("reference-transaction", "post-checkout", "post-index-change", "pre-commit", "post-commit")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def session_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("# repo\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    worktree = repo / ".worktrees" / "session"
    _git(repo, "worktree", "add", "-q", "-b", "hermes/session", str(worktree), "HEAD")
    markers = tmp_path / "markers"
    markers.mkdir()
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for name in _HOOKS:
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\ntouch '{markers / name}'\nexit 0\n")
        hook.chmod(0o755)
    _git(repo, "config", "core.hooksPath", str(hooks))
    return repo, worktree, markers


def _cleanup(monkeypatch, repo: Path, worktree: Path) -> str:
    from hermes_agent import cli

    printed: list[str] = []
    monkeypatch.setattr(cli, "release_lsp_clients", lambda path: None)
    monkeypatch.setattr(cli, "_cprint", printed.append)
    cli._cleanup_worktree({"path": str(worktree), "branch": "hermes/session", "repo_root": str(repo)})
    return "".join(printed)


def test_cleanup_deletes_the_session_branch_without_running_repository_hooks(
        session_worktree, monkeypatch, tmp_path):
    repo, worktree, markers = session_worktree
    monkeypatch.setenv("HOME", str(tmp_path))

    out = _cleanup(monkeypatch, repo, worktree)

    assert sorted(p.name for p in markers.iterdir()) == []
    assert not worktree.exists()
    assert _git(repo, "branch", "--list", "hermes/session").strip() == ""
    assert "cleaned up" in out


def test_worktree_removal_steps_still_run_under_the_inherited_environment(session_worktree, monkeypatch, tmp_path):
    repo, worktree, _markers = session_worktree
    monkeypatch.setenv("HOME", str(tmp_path))
    _git(repo, "worktree", "lock", str(worktree))
    seen: list[tuple[list[str], object]] = []
    real_run = subprocess.run

    def recording_run(argv, *args, **kwargs):
        seen.append((list(argv), kwargs.get("env")))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)

    _cleanup(monkeypatch, repo, worktree)

    removal = [env for argv, env in seen if argv[:3] in (["git", "worktree", "unlock"], ["git", "worktree", "remove"])]
    assert len(removal) == 2 and all(env is None for env in removal)
    assert not worktree.exists()
    assert os.path.isdir(repo)
