"""Tests for workspace + project-root resolution."""
from __future__ import annotations

import os
import sys
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
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert find_git_worktree(str(sub)) == str(repo)


def test_find_git_worktree_ignores_empty_dotgit_dir(tmp_path: Path):
    # A stray empty ~/.git made every file under $HOME "a repo", rooting servers at $HOME;
    # intelephense then crawled the whole home directory until V8 hit its heap limit.
    home = tmp_path / "home"
    (home / ".git").mkdir(parents=True)
    work = home / "scratch"
    work.mkdir()
    assert find_git_worktree(str(work)) is None
    assert resolve_workspace_for_file(str(work / "x.php"), cwd=str(home)) == (None, False)


def test_find_git_worktree_accepts_gitfile(tmp_path: Path):
    # Linked worktrees and submodules have a ``.git`` file pointing at the real gitdir.
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    assert find_git_worktree(str(wt)) == str(wt)


def test_nearest_root_finds_first_marker(tmp_path: Path):
    root = tmp_path / "p"
    deep = root / "src" / "pkg"
    deep.mkdir(parents=True)
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    found = nearest_root(str(deep / "mod.py"), ["pyproject.toml"])
    assert found == str(root)


def test_nearest_root_skips_package_dirs(tmp_path: Path):
    # hermes_cli/setup.py is a module inside a package, not a project
    # marker; treating it as one spawned a second pyright per worktree.
    root = tmp_path / "p"
    pkg = root / "hermes_cli"
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "setup.py").write_text("", encoding="utf-8")
    found = nearest_root(str(pkg / "main.py"), ["pyproject.toml", "setup.py"])
    assert found == str(root)


def test_resolve_workspace_for_file_uses_cwd_first(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    file_path = repo / "x.py"
    file_path.write_text("", encoding="utf-8")
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
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    file_path = repo / "x.py"
    file_path.write_text("")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    scratch.rmdir()
    with pytest.raises(OSError):
        os.getcwd()

    root, gated = resolve_workspace_for_file(str(file_path))

    assert root == str(repo)
    assert gated is True
    # The diagnostics path logs through eventlog; its cwd-relative shortener must
    # not raise either, or the write succeeds with diagnostics silently dropped.
    from agent.lsp.eventlog import _short_path

    assert _short_path(str(file_path)) == str(file_path)


@pytest.mark.platforms("linux")
def test_normalize_path_expands_tilde(monkeypatch):
    # expanduser keys off USERPROFILE on native Windows, HOME elsewhere —
    # set the var the running host actually consults (host-native rule:
    # never fake the platform).
    if sys.platform == "win32":
        monkeypatch.setenv("USERPROFILE", r"C:\Users\fakeuser")
        expected_base = r"C:\Users\fakeuser"
    else:
        monkeypatch.setenv("HOME", "/home/user")
        expected_base = "/home/user"
    p = normalize_path("~/x.py")
    assert p == os.path.abspath(os.path.join(expected_base, "x.py"))


def test_find_git_worktree_cache_is_capped(tmp_path: Path, monkeypatch):
    """The start-dir cache resets past _WORKSPACE_CACHE_CAP instead of growing per distinct dir touched."""
    import agent.lsp.workspace as ws

    monkeypatch.setattr(ws, "_WORKSPACE_CACHE_CAP", 4)
    for i in range(6):
        d = tmp_path / f"d{i}"
        d.mkdir()
        assert find_git_worktree(str(d)) is None
    assert len(ws._workspace_cache) <= 4
