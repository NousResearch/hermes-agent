"""Kanban linked worktrees of an npm-workspaces repo get their own ``node_modules``.

Without it, Node resolution from ``<repo>/.worktrees/<id>`` walks up into the main
checkout's ``node_modules``, whose workspace links point at the main checkout's
packages, so the worker silently runs code that is not on its branch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db_workspace as kbw
from hermes_cli.kanban_db import Task

_NPM = shutil.which("npm")
pytestmark = pytest.mark.skipif(_NPM is None or shutil.which("node") is None, reason="needs node + npm")


@pytest.fixture(autouse=True)
def _host_npm(monkeypatch):
    """Use the host npm in place of Hermes' PM-managed one, which the test env lacks."""
    monkeypatch.setattr(kbw, "find_node_executable", lambda name: _NPM)
    monkeypatch.setattr(kbw, "with_hermes_node_path", lambda: dict(os.environ))


def _run(cwd: Path, *args: str) -> str:
    return subprocess.run(
        list(args), cwd=cwd, check=True, capture_output=True, text=True, timeout=120,
    ).stdout.strip()


def _npm_workspace_repo(tmp_path: Path) -> Path:
    """A committed npm-workspaces repo whose main checkout is already installed."""
    repo = tmp_path / "repo"
    for name, extra in (("a", {}), ("b", {"dependencies": {"@fixture/a": "*"}})):
        pkg = repo / "packages" / name
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text(
            json.dumps({"name": f"@fixture/{name}", "version": "1.0.0", **extra}), encoding="utf-8"
        )
        (pkg / "index.js").write_text(f"module.exports = {name!r};\n", encoding="utf-8")
    (repo / "package.json").write_text(
        json.dumps({"name": "fixture-root", "private": True, "workspaces": ["packages/*"]}),
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text("node_modules/\n.worktrees/\n", encoding="utf-8")
    _run(repo, str(_NPM), "install", "--ignore-scripts", "--no-audit", "--no-fund", "--offline")
    _run(repo, "git", "init", "-q", "-b", "main")
    _run(repo, "git", "add", ".")
    _run(
        repo, "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init",
    )
    return repo


def _resolves_inside(worktree: Path) -> bool:
    resolved = _run(worktree, "node", "-p", "require.resolve('@fixture/a')")
    return Path(resolved).resolve().is_relative_to(worktree.resolve())


def _task(task_id: str, workspace: Path) -> Task:
    return Task(
        id=task_id, title="t", body=None, assignee=None, status="running", priority=0,
        created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="worktree", workspace_path=str(workspace), claim_lock=None,
        claim_expires=None, tenant=None, branch_name=f"wt/{task_id}",
    )


def test_new_worktree_resolves_workspace_packages_from_its_own_checkout(tmp_path):
    repo = _npm_workspace_repo(tmp_path)
    target = repo / ".worktrees" / "t_new"

    kbw._ensure_git_worktree(repo, target, "wt/t_new")

    assert _resolves_inside(target)


def test_reused_same_branch_worktree_is_provisioned(tmp_path):
    repo = _npm_workspace_repo(tmp_path)
    target = repo / ".worktrees" / "t_reused"
    target.parent.mkdir()
    _run(repo, "git", "worktree", "add", "-q", "-b", "wt/t_reused", str(target), "HEAD")

    workspace, branch = kbw._resolve_worktree_workspace(_task("t_reused", target))

    assert (workspace, branch) == (target.resolve(), "wt/t_reused")
    assert _resolves_inside(target)


@pytest.mark.platforms("posix")
def test_failed_install_does_not_block_dispatch_or_leave_partial_tree(tmp_path, monkeypatch, caplog):
    repo = _npm_workspace_repo(tmp_path)
    fake_npm = tmp_path / "npm"
    fake_npm.write_text("#!/bin/sh\nmkdir node_modules\necho 'registry down' >&2\nexit 1\n")
    fake_npm.chmod(0o755)
    monkeypatch.setattr(kbw, "find_node_executable", lambda name: str(fake_npm))
    target = repo / ".worktrees" / "t_failed"

    with caplog.at_level("WARNING", logger=kbw.__name__):
        workspace, branch = kbw._resolve_worktree_workspace(_task("t_failed", target))

    assert (workspace, branch) == (target, "wt/t_failed")
    assert (target / "package.json").is_file()
    assert not (target / "node_modules").exists()
    assert "registry down" in caplog.text
