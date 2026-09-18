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
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert find_git_worktree(str(sub)) == str(repo)








def test_nearest_root_finds_first_marker(tmp_path: Path):
    root = tmp_path / "p"
    deep = root / "src" / "pkg"
    deep.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    found = nearest_root(str(deep / "mod.py"), ["pyproject.toml"])
    assert found == str(root)


def test_nearest_root_skips_package_dirs(tmp_path: Path):
    # hermes_cli/setup.py is a module inside a package, not a project
    # marker; treating it as one spawned a second pyright per worktree.
    root = tmp_path / "p"
    pkg = root / "hermes_cli"
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "setup.py").write_text("")
    found = nearest_root(str(pkg / "main.py"), ["pyproject.toml", "setup.py"])
    assert found == str(root)






def test_resolve_workspace_for_file_uses_cwd_first(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    file_path = repo / "x.py"
    file_path.write_text("")
    # cwd is inside the repo
    monkeypatch.chdir(str(repo))
    root, gated = resolve_workspace_for_file(str(file_path))
    assert root == str(repo)
    assert gated is True


def test_resolve_workspace_for_file_survives_deleted_cwd(tmp_path: Path, monkeypatch):
    """A removed process cwd must read as "no anchor", not raise — the LSP
    workspace resolver runs inside a write tool and must never break a write
    that already landed on disk."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    file_path = repo / "x.py"
    file_path.write_text("")

    def _deleted_cwd():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr("agent.lsp.workspace.os.getcwd", _deleted_cwd)
    # cwd argument absent → the process cwd is consulted and raises
    root, gated = resolve_workspace_for_file(str(file_path))
    # falls through to the file's own worktree
    assert root == str(repo)
    assert gated is True


def test_current_dir_none_when_cwd_unreadable(monkeypatch):
    from agent.lsp.workspace import _current_dir

    def _deleted_cwd():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr("agent.lsp.workspace.os.getcwd", _deleted_cwd)
    assert _current_dir() is None
    monkeypatch.setattr("agent.lsp.workspace.os.getcwd", lambda: "/somewhere")
    assert _current_dir() == "/somewhere"






def test_normalize_path_expands_tilde(monkeypatch):
    monkeypatch.setenv("HOME", "/home/user")
    p = normalize_path("~/x.py")
    assert p == os.path.abspath("/home/user/x.py")
