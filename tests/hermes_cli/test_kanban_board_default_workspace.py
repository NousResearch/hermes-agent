"""Tests for board-declared default workspace kind (#123288).

Regression for #123288: a board whose ``default_workdir`` is a Git repo root
used to produce ``dir`` tasks the governed backend's workspace-binding guard
always rejects (the guard demands a *strict* descendant AND the exact Git
root — only a worktree satisfies both). Boards can now declare
``default_workspace_kind`` in ``board.json``; ``create_task`` falls back to
it when no explicit workspace is given.

Behavior contracts covered:

* ``write_board_metadata`` validates the kind and clears on "".
* A ``worktree`` default requires a ``default_workdir`` anchor.
* ``create_task`` inherits the declared kind for omitted workspaces, and an
  explicit ``scratch`` still overrides it (the #106342 opt-out contract).
* CLI: ``boards create --default-workspace`` and
  ``boards set-default-workspace`` round-trip through ``board.json``.
* Board export strips the machine-local declared kind alongside workdir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_transfer


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


@pytest.fixture
def repo(tmp_path):
    """A real git repo to anchor board default_workdir / worktrees."""
    import subprocess
    r = tmp_path / "repo"
    r.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path)}
    def git(*args):
        subprocess.run(["git", *args], cwd=r, check=True, capture_output=True, env=env)
    git("init", "-q")
    git("commit", "--allow-empty", "-q", "-m", "init")
    return r


# ---------------------------------------------------------------------------
# Metadata layer
# ---------------------------------------------------------------------------

class TestBoardMetadataKind:
    def test_write_valid_kind_round_trips(self, fresh_home):
        kb.create_board("demo", default_workspace_kind="dir")
        meta = kb.read_board_metadata("demo")
        assert meta["default_workspace_kind"] == "dir"
        on_disk = json.loads(
            (fresh_home / "kanban" / "boards" / "demo" / "board.json").read_text())
        assert on_disk["default_workspace_kind"] == "dir"

    def test_write_invalid_kind_rejected(self, fresh_home):
        with pytest.raises(ValueError, match="default_workspace_kind"):
            kb.create_board("demo", default_workspace_kind="blob")

    def test_empty_string_clears_kind(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        meta = kb.write_board_metadata("demo", default_workspace_kind="")
        assert meta["default_workspace_kind"] is None
        assert kb.read_board_metadata("demo")["default_workspace_kind"] is None

    def test_unset_leaves_kind_unchanged(self, fresh_home):
        kb.create_board("demo", default_workspace_kind="dir")
        meta = kb.write_board_metadata("demo", name="Renamed")
        assert meta["default_workspace_kind"] == "dir"

    def test_worktree_kind_requires_default_workdir(self, fresh_home):
        with pytest.raises(ValueError, match="default_workdir"):
            kb.write_board_metadata("demo", default_workspace_kind="worktree")

    def test_worktree_kind_with_workdir_ok(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        meta = kb.read_board_metadata("demo")
        assert meta["default_workspace_kind"] == "worktree"
        assert meta["default_workdir"] == str(repo)

    def test_clearing_workdir_then_kind_guard(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        # Clearing the workdir while kind=worktree would strand every default.
        with pytest.raises(ValueError, match="default_workdir"):
            kb.write_board_metadata("demo", default_workdir="")


# ---------------------------------------------------------------------------
# create_task default resolution
# ---------------------------------------------------------------------------

class TestCreateTaskDefaultKind:
    def test_task_inherits_declared_worktree_kind(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        with kbc.connect_closing(board="demo") as conn:
            tid = kb.create_task(conn, title="t", workspace_path=str(repo), board="demo")
            task = kb.get_task(conn, tid)
        assert task.workspace_kind == "worktree"

    def test_explicit_scratch_overrides_board_default(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        with kbc.connect_closing(board="demo") as conn:
            tid = kb.create_task(conn, title="t", workspace_kind="scratch", board="demo")
            task = kb.get_task(conn, tid)
        assert task.workspace_kind == "scratch"
        # Opt-out contract (#106342): scratch must not inherit the board workdir.
        assert task.workspace_path is None

    def test_no_declared_kind_defaults_to_scratch(self, fresh_home):
        kb.create_board("demo")
        with kbc.connect_closing(board="demo") as conn:
            tid = kb.create_task(conn, title="t", board="demo")
            task = kb.get_task(conn, tid)
        assert task.workspace_kind == "scratch"

    def test_dir_kind_with_workdir(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="dir")
        with kbc.connect_closing(board="demo") as conn:
            tid = kb.create_task(conn, title="t", board="demo")
            task = kb.get_task(conn, tid)
        assert task.workspace_kind == "dir"
        assert task.workspace_path == str(repo)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def _run_cli(*argv, home):
    """Parse real CLI argv and dispatch the boards handler."""
    import argparse
    from hermes_cli import kanban_boards
    import hermes_cli.kanban as kc
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="top")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", "boards", *argv])
    return kanban_boards._dispatch_boards(args)


class TestCli:
    def test_create_with_default_workspace(self, fresh_home, capsys):
        rc = _run_cli("create", "demo", "--default-workspace", "dir",
                      "--default-workdir", "/tmp/whatever", home=fresh_home)
        assert rc == 0
        assert kb.read_board_metadata("demo")["default_workspace_kind"] == "dir"

    def test_set_default_workspace_round_trip(self, fresh_home, capsys):
        kb.create_board("demo")
        rc = _run_cli("set-default-workspace", "demo", "worktree", home=fresh_home)
        assert rc == 2  # guard: no default_workdir yet
        kb.write_board_metadata("demo", default_workdir="/tmp/whatever")
        rc = _run_cli("set-default-workspace", "demo", "worktree", home=fresh_home)
        assert rc == 0
        assert kb.read_board_metadata("demo")["default_workspace_kind"] == "worktree"
        rc = _run_cli("set-default-workspace", "demo", home=fresh_home)
        assert rc == 0
        assert kb.read_board_metadata("demo")["default_workspace_kind"] is None

    def test_set_default_workspace_rejects_unknown(self, fresh_home):
        kb.create_board("demo")
        rc = _run_cli("set-default-workspace", "demo", "blob", home=fresh_home)
        assert rc == 2


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_strips_declared_kind(self, fresh_home, tmp_path, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        with kbc.connect_closing(board="demo") as conn:
            kb.create_task(conn, title="t", workspace_path=str(repo), board="demo")
        out = tmp_path / "demo.tar.gz"
        kanban_transfer.export_board("demo", str(out))
        import tarfile
        with tarfile.open(out) as tf:
            meta = json.load(tf.extractfile("demo/board.json"))
        assert meta["default_workspace_kind"] is None
        assert meta["default_workdir"] is None
