"""Tests for agent/runtime_cwd.py — the single source of truth for the agent working directory."""

import os
from pathlib import Path

import pytest

import agent.runtime_cwd as rt
from agent.runtime_cwd import (
    clear_session_cwd,
    resolve_agent_cwd,
    resolve_context_cwd,
    set_session_cwd,
)


def _raise_oserror(*args, **kwargs):
    raise OSError("cwd gone")


class TestResolveAgentCwd:
    def test_prefers_terminal_cwd_over_getcwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        monkeypatch.chdir(os.path.expanduser("~"))
        assert resolve_agent_cwd() == tmp_path




    def test_propagates_oserror_from_getcwd(self, monkeypatch):
        # The fallback arm calls os.getcwd(), which can raise OSError (deleted cwd).
        # The resolver must NOT swallow it — build_environment_hints owns the
        # try/except OSError guard at the call site (prompt_builder.py:805).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(rt.os, "getcwd", _raise_oserror)
        with pytest.raises(OSError):
            resolve_agent_cwd()


class TestResolveContextCwd:
    def test_returns_dir_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert resolve_context_cwd() == tmp_path




    def test_expands_leading_tilde(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "~")
        assert resolve_context_cwd() == Path(os.path.expanduser("~"))



class TestSessionCwdOverride:
    """The #29531 per-session arm: a contextvar cwd wins over TERMINAL_CWD so a
    multi-session gateway can pin each session to its own folder."""

    def test_session_cwd_overrides_terminal_cwd(self, monkeypatch, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd(str(other))
        try:
            assert resolve_agent_cwd() == other
            assert resolve_context_cwd() == other
        finally:
            rt._SESSION_CWD.reset(token)


    def test_clear_session_cwd_restores_terminal_cwd(self, monkeypatch, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd(str(other))
        try:
            clear_session_cwd()
            assert resolve_agent_cwd() == tmp_path
        finally:
            rt._SESSION_CWD.reset(token)


def _declare_project(db_path, *, name, slug, folders):
    """Create a real project row in a temp projects DB (schema auto-inits)."""
    import hermes_cli.projects_db as pdb

    conn = pdb.connect(db_path=db_path)
    try:
        return pdb.create_project(conn, name=name, slug=slug, folders=folders)
    finally:
        conn.close()


def _point_projects_db(tmp_path, monkeypatch, *, seed=True):
    """Point the projects DAO at a temp DB (optionally seeded); returns its path."""
    import hermes_cli.projects_db as pdb

    path = tmp_path / "projects.db"
    if seed:
        _declare_project(path, name="Seed", slug="seed", folders=[str(tmp_path / "seed-folder")])
    monkeypatch.setattr(pdb, "projects_db_path", lambda: path)
    return path


class TestResolveWorkspaceIdentity:
    """Declared project → git root → cwd basename → "" cascade (the memory-provider
    ``agent_workspace`` identity; mirrors the Desktop project tree's resolution)."""

    def test_declared_project_slug_wins(self, tmp_path, monkeypatch):
        proj = tmp_path / "work" / "myproj"
        proj.mkdir(parents=True)
        db_path = _point_projects_db(tmp_path, monkeypatch)
        _declare_project(db_path, name="My Project", slug="myproject", folders=[str(proj)])
        assert rt.resolve_workspace_identity(str(proj)) == "myproject"
        # Nested folders resolve to the deepest owning project.
        assert rt.resolve_workspace_identity(str(proj / "sub" / "dir")) == "myproject"

    def test_deepest_owning_project_wins(self, tmp_path, monkeypatch):
        outer = tmp_path / "outer"
        inner = outer / "inner"
        leaf = inner / "apps"
        leaf.mkdir(parents=True)
        db_path = _point_projects_db(tmp_path, monkeypatch)
        _declare_project(db_path, name="Outer", slug="outer", folders=[str(outer)])
        _declare_project(db_path, name="Inner", slug="inner", folders=[str(inner)])
        assert rt.resolve_workspace_identity(str(leaf)) == "inner"

    def test_declared_project_beats_git_root(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo-name"
        repo.mkdir()
        (repo / ".git").mkdir()
        db_path = _point_projects_db(tmp_path, monkeypatch)
        _declare_project(db_path, name="Display", slug="declared-slug", folders=[str(repo)])
        assert rt.resolve_workspace_identity(str(repo)) == "declared-slug"

    def test_git_repo_resolves_from_subdirectory(self, tmp_path, monkeypatch):
        repo = tmp_path / "somerepo"
        sub = repo / "apps" / "web"
        sub.mkdir(parents=True)
        (repo / ".git").mkdir()
        _point_projects_db(tmp_path, monkeypatch)
        assert rt.resolve_workspace_identity(str(sub)) == "somerepo"

    def test_plain_directory_uses_basename(self, tmp_path, monkeypatch):
        plain = tmp_path / "plaindir"
        plain.mkdir()
        _point_projects_db(tmp_path, monkeypatch)
        assert rt.resolve_workspace_identity(str(plain)) == "plaindir"

    def test_repo_root_argument_wins_over_walkup(self, tmp_path, monkeypatch):
        # Worktree-style: the checkout itself is not a repo root candidate; the
        # session-stamped repo_root is.
        cwd = tmp_path / "checkouts" / "wt-abc"
        cwd.mkdir(parents=True)
        root = tmp_path / "mainrepo"
        root.mkdir()
        _point_projects_db(tmp_path, monkeypatch)
        assert rt.resolve_workspace_identity(str(cwd), repo_root=str(root)) == "mainrepo"

    def test_non_workspace_paths_return_empty(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "hermes-home"
        (fake_home / "internal").mkdir(parents=True)
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: fake_home)
        _point_projects_db(tmp_path, monkeypatch)
        assert rt.resolve_workspace_identity(os.sep) == ""
        assert rt.resolve_workspace_identity(os.path.expanduser("~")) == ""
        assert rt.resolve_workspace_identity("/home") == ""
        assert rt.resolve_workspace_identity("/Users") == ""
        assert rt.resolve_workspace_identity(str(fake_home)) == ""
        assert rt.resolve_workspace_identity(str(fake_home / "internal")) == ""

    def test_projects_db_missing_falls_back_to_basename(self, tmp_path, monkeypatch):
        import hermes_cli.projects_db as pdb

        monkeypatch.setattr(pdb, "projects_db_path", lambda: tmp_path / "absent" / "projects.db")
        d = tmp_path / "nodb"
        d.mkdir()
        assert rt.resolve_workspace_identity(str(d)) == "nodb"

    def test_projects_db_corrupt_falls_back_without_raising(self, tmp_path, monkeypatch):
        import hermes_cli.projects_db as pdb

        corrupt = tmp_path / "projects.db"
        corrupt.write_bytes(b"this is not a sqlite database")
        monkeypatch.setattr(pdb, "projects_db_path", lambda: corrupt)
        d = tmp_path / "notadb"
        d.mkdir()
        assert rt.resolve_workspace_identity(str(d)) == "notadb"

    def test_empty_input_returns_empty(self):
        assert rt.resolve_workspace_identity("") == ""
        assert rt.resolve_workspace_identity("   ") == ""
