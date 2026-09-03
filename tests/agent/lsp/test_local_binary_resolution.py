"""Tests for marker-gated repo-local LSP binary resolution.

Ported/adapted from oh-my-openagent PR #7424: a project's own pinned
language server (``node_modules/.bin``, ``.venv/bin``, ...) must win over
a globally installed one, but only when a sibling ecosystem marker file
authorizes the bin directory — an unmarked stray directory must never
inject an executable into the resolution order.
"""
import os
import stat
import sys

import pytest

from agent.lsp import servers as srv


def _make_exe(directory: str, name: str) -> str:
    os.makedirs(directory, exist_ok=True)
    if sys.platform == "win32":
        path = os.path.join(directory, name + ".cmd")
        with open(path, "w") as f:
            f.write("@echo off\n")
    else:
        path = os.path.join(directory, name)
        with open(path, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture(autouse=True)
def _clear_cache():
    srv._LOCAL_RESOLVE_CACHE.clear()
    yield
    srv._LOCAL_RESOLVE_CACHE.clear()


def test_node_local_bin_wins_when_marked(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "package.json").write_text("{}")
    local = _make_exe(str(root / "node_modules" / ".bin"), "typescript-language-server")
    monkeypatch.setattr(srv.shutil, "which", lambda n, path=None: (
        local if path and os.path.dirname(local) == path else "/usr/bin/" + n))
    assert srv._which("typescript-language-server", root=str(root)) == local


def test_unmarked_node_modules_is_ignored(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    # bin exists but NO package.json / lockfile marker → must not resolve
    _make_exe(str(root / "node_modules" / ".bin"), "evil-langserver")
    assert srv._resolve_local_binary("evil-langserver", str(root)) is None


def test_python_venv_bin_resolves_with_marker(tmp_path):
    root = tmp_path / "pyproj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    sub = "Scripts" if sys.platform == "win32" else "bin"
    local = _make_exe(str(root / ".venv" / sub), "pyright-langserver")
    assert srv._resolve_local_binary("pyright-langserver", str(root)) == local


def test_walk_up_stops_at_git_boundary(tmp_path):
    # repo/.git + repo/package.json + repo/node_modules/.bin/tool
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "package.json").write_text("{}")
    local = _make_exe(str(repo / "node_modules" / ".bin"), "some-ls")
    nested = repo / "packages" / "app"
    nested.mkdir(parents=True)
    # Nested dir with no markers walks up and finds the repo-level bin.
    assert srv._resolve_local_binary("some-ls", str(nested)) == local

    # Tooling ABOVE the repo root must not leak in: outer dir has a marked
    # bin, but the walk from inside the repo ends at repo/.git.
    (tmp_path / "package.json").write_text("{}")
    _make_exe(str(tmp_path / "node_modules" / ".bin"), "outer-ls")
    assert srv._resolve_local_binary("outer-ls", str(nested)) is None


def test_path_fallback_when_no_local(tmp_path, monkeypatch):
    root = tmp_path / "plain"
    root.mkdir()
    monkeypatch.setattr(srv.shutil, "which", lambda n, path=None: (
        None if path else "/usr/local/bin/" + n))
    assert srv._which("gopls", root=str(root)) == "/usr/local/bin/gopls"


def test_which_without_root_unchanged(monkeypatch):
    monkeypatch.setattr(srv.shutil, "which", lambda n, path=None: (
        "/usr/bin/rust-analyzer" if n == "rust-analyzer" else None))
    assert srv._which("rust-analyzer") == "/usr/bin/rust-analyzer"
    assert srv._which("nope-ls") is None


def test_resolution_is_cached(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "package.json").write_text("{}")
    local = _make_exe(str(root / "node_modules" / ".bin"), "cached-ls")
    calls = {"n": 0}
    real_which = srv.shutil.which

    def counting(n, path=None):
        calls["n"] += 1
        return real_which(n, path=path)

    monkeypatch.setattr(srv.shutil, "which", counting)
    first = srv._resolve_local_binary("cached-ls", str(root))
    n_after_first = calls["n"]
    second = srv._resolve_local_binary("cached-ls", str(root))
    assert first == second == local
    assert calls["n"] == n_after_first  # no new probes on the cached call


def test_spawn_typescript_uses_project_local_server(tmp_path, monkeypatch):
    """E2E through a real spawn builder: SpawnSpec.command[0] is the local bin."""
    root = tmp_path / "webapp"
    root.mkdir()
    (root / "package.json").write_text("{}")
    local = _make_exe(str(root / "node_modules" / ".bin"), "typescript-language-server")
    # Global PATH also has one — the local pin must win.
    monkeypatch.setattr(
        srv.shutil, "which",
        lambda n, path=None: (local if path == os.path.dirname(local)
                              else "/usr/bin/typescript-language-server"))
    ctx = srv.ServerContext(workspace_root=str(root), install_strategy="off")
    spec = srv._spawn_typescript(str(root), ctx)
    assert spec is not None
    assert spec.command[0] == local
