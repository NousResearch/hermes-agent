"""Tests for workspace + project-root resolution."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.lsp.workspace import (
    clear_cache,
    find_git_worktree,
    is_inside_workspace,
    nearest_root,
    normalize_path,
    resolve_workspace_for_file,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()




def test_find_git_worktree_finds_dotgit(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert find_git_worktree(str(sub)) == str(repo)


def test_shared_temp_root_is_not_a_worktree(tmp_path: Path, monkeypatch):
    """A stray `.git` in the shared temp dir must not make every path under it
    look like a workspace.

    `/tmp` is world-writable, so any process can leave a `.git` behind. Without
    this guard the gate opens for paths that have no project at all, and Hermes
    spawns language servers for them — the failure mode the workspace gate
    exists to prevent.
    """
    fake_tmp = tmp_path / "shared-tmp"
    (fake_tmp / ".git").mkdir(parents=True)
    (fake_tmp / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    work = fake_tmp / "scratch"
    work.mkdir()
    # Patched on utils, not this module: find_git_worktree delegates the
    # walk (and the temp-root guard) to utils.find_git_root.
    monkeypatch.setattr("utils.tempfile.gettempdir", lambda: str(fake_tmp))
    # Asserts the temp root is not *claimed* rather than asserting ``None``:
    # ``tmp_path`` itself lives under the machine's real temp dir, and a host
    # with its own stray ``/tmp/.git`` would resolve to that instead — making an
    # ``is None`` assertion pass or fail on ambient state rather than on the
    # behavior under test. (That ambient sensitivity is the very bug this guard
    # fixes; the test must not inherit it.)
    assert find_git_worktree(str(work)) != str(fake_tmp)


def test_real_repo_under_the_temp_root_still_resolves(tmp_path: Path, monkeypatch):
    """Only the temp root itself is skipped, not everything beneath it —
    otherwise every `tmp_path`-based workspace test would go blind."""
    fake_tmp = tmp_path / "shared-tmp"
    repo = fake_tmp / "checkout"
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    # Patched on utils, not this module: find_git_worktree delegates the
    # walk (and the temp-root guard) to utils.find_git_root.
    monkeypatch.setattr("utils.tempfile.gettempdir", lambda: str(fake_tmp))
    assert find_git_worktree(str(repo)) == str(repo)








def test_nearest_root_finds_first_marker(tmp_path: Path):
    root = tmp_path / "p"
    deep = root / "src" / "pkg"
    deep.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    found = nearest_root(str(deep / "mod.py"), ["pyproject.toml"])
    assert found == str(root)






def test_resolve_workspace_for_file_uses_cwd_first(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    file_path = repo / "x.py"
    file_path.write_text("")
    # cwd is inside the repo
    monkeypatch.chdir(str(repo))
    root, gated = resolve_workspace_for_file(str(file_path))
    assert root == str(repo)
    assert gated is True






def test_normalize_path_expands_tilde(monkeypatch):
    monkeypatch.setenv("HOME", "/home/user")
    p = normalize_path("~/x.py")
    assert p == os.path.abspath("/home/user/x.py")
