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
    def test_write_valid_kind_round_trips(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="dir")
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

    def test_unset_leaves_kind_unchanged(self, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="dir")
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

    def test_dir_kind_requires_default_workdir(self, fresh_home):
        # dir dies identically at dispatch ("dir but no workspace_path"), so
        # the write-side guard covers both persistent kinds.
        with pytest.raises(ValueError, match="default_workdir"):
            kb.write_board_metadata("demo", default_workspace_kind="dir")

    def test_setting_dir_kind_on_workdirless_board_rejected(self, fresh_home):
        kb.create_board("demo")
        with pytest.raises(ValueError, match="dir"):
            kb.write_board_metadata("demo", default_workspace_kind="dir")

    def test_hand_edited_garbage_kind_refused_at_create_task(self, fresh_home):
        # board.json is hand-editable and validation is write-side only; a
        # truthy garbage value used to brick EVERY card creation with an
        # opaque "workspace_kind must be one of ..." while falsy garbage
        # (False/0) silently fell through to scratch. Refuse loudly with
        # the fix command instead.
        kb.create_board("demo")
        meta_path = fresh_home / "kanban" / "boards" / "demo" / "board.json"
        meta = json.loads(meta_path.read_text())
        meta["default_workspace_kind"] = '"off"'
        meta_path.write_text(json.dumps(meta))
        with kbc.connect_closing(board="demo") as conn:
            with pytest.raises(ValueError, match="invalid default_workspace_kind"):
                kb.create_task(conn, title="t", board="demo")

    def test_hand_edited_falsy_garbage_kind_refused_at_create_task(self, fresh_home):
        kb.create_board("demo")
        meta_path = fresh_home / "kanban" / "boards" / "demo" / "board.json"
        meta = json.loads(meta_path.read_text())
        meta["default_workspace_kind"] = False
        meta_path.write_text(json.dumps(meta))
        with kbc.connect_closing(board="demo") as conn:
            with pytest.raises(ValueError, match="invalid default_workspace_kind"):
                kb.create_task(conn, title="t", board="demo")


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

    def test_declared_scratch_pins_kind_on_project_board(self, fresh_home, repo, tmp_path):
        # The declared kind is read BEFORE the project-inheritance check:
        # a board declaring scratch on a project-scoped board must NOT have
        # it upgraded to worktree by _resolve_project_link — the same
        # #106342 no-project opt-out an explicit scratch gets.
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            pid = pdb.create_project(
                pconn, name="Proj", slug="proj", primary_path=str(repo))
        kb.create_board("demo", project_id="proj", default_workspace_kind="scratch")
        with kbc.connect_closing(board="demo") as conn:
            tid = kb.create_task(conn, title="t", board="demo")
            task = kb.get_task(conn, tid)
        assert task is not None
        assert task.workspace_kind == "scratch"
        assert task.workspace_path is None
        assert task.project_id is None


# ---------------------------------------------------------------------------
# Dashboard API: declared vs recommended kind (rb/123543, rb/121149)
# ---------------------------------------------------------------------------

class TestDashboardDeclaredVsRecommended:
    """GET /boards must return the declared kind in ``default_workspace_kind``
    and the derived hint under ``recommended_workspace_kind`` — otherwise one
    read-modify-write (the settings dialog's PATCH) mints a real declaration
    for every board whose workdir merely looks like a repo."""

    @pytest.fixture
    def client(self, fresh_home, repo):
        import importlib.util
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        # Load the plugin exactly the way tests/plugins/test_kanban_dashboard_plugin.py
        # does (the dashboard ships as a file, not an importable package).
        plugin_file = _WORKTREE / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
        spec = importlib.util.spec_from_file_location(
            "hermes_dashboard_plugin_kanban_dws_test", plugin_file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/kanban")
        return TestClient(app)

    def test_get_boards_does_not_mint_a_declaration(self, client, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo))  # a repo workdir, nothing declared
        r = client.get("/api/plugins/kanban/boards")
        assert r.status_code == 200, r.text
        board = next(b for b in r.json()["boards"] if b["slug"] == "demo")
        assert board["default_workspace_kind"] is None       # NOT minted as "worktree"
        assert board["recommended_workspace_kind"] == "worktree"

    def test_read_modify_write_preserves_undeclared(self, client, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo))
        # The dangerous round trip: read the board, PATCH the ENTIRE payload
        # back unchanged. With the derived hint overwriting the declared
        # field this minted default_workspace_kind=worktree on disk.
        r = client.get("/api/plugins/kanban/boards")
        board = next(b for b in r.json()["boards"] if b["slug"] == "demo")
        r2 = client.patch(
            "/api/plugins/kanban/boards/demo",
            json={"name": board.get("name"), "default_workspace_kind": board["default_workspace_kind"]},
        )
        assert r2.status_code == 200, r2.text
        assert kb.read_board_metadata("demo")["default_workspace_kind"] is None

    def test_declared_kind_round_trips_through_get(self, client, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        r = client.get("/api/plugins/kanban/boards")
        board = next(b for b in r.json()["boards"] if b["slug"] == "demo")
        assert board["default_workspace_kind"] == "worktree"
        assert board["recommended_workspace_kind"] == "worktree"

    def test_patch_clears_declaration(self, client, fresh_home, repo):
        kb.create_board("demo", default_workdir=str(repo), default_workspace_kind="worktree")
        r = client.patch(
            "/api/plugins/kanban/boards/demo",
            json={"default_workspace_kind": ""},
        )
        assert r.status_code == 200, r.text
        assert kb.read_board_metadata("demo")["default_workspace_kind"] is None

    def test_post_boards_rejects_dir_without_workdir(self, client, fresh_home):
        r = client.post("/api/plugins/kanban/boards", json={
            "slug": "demo2", "default_workspace_kind": "dir"})
        assert r.status_code == 400, r.text
        assert "default_workdir" in r.json()["detail"]


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
