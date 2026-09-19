"""Board→project scoping in kanban_db.

A kanban board can be scoped to a first-class Hermes project so every task on
it anchors to that project (deterministic worktree + branch). Covers the
metadata round-trip and the create-time inheritance.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import projects_db as pdb


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def test_board_metadata_project_id_roundtrip(fresh_home):
    assert kb.read_board_metadata("default").get("project_id") is None

    kb.write_board_metadata("default", project_id="p_abc123")
    assert kb.read_board_metadata("default")["project_id"] == "p_abc123"

    # None leaves unchanged; "" clears.
    kb.write_board_metadata("default", name="Still Here")
    assert kb.read_board_metadata("default")["project_id"] == "p_abc123"
    kb.write_board_metadata("default", project_id="")
    assert kb.read_board_metadata("default")["project_id"] is None


def test_create_board_accepts_project_id(fresh_home):
    meta = kb.create_board("proj-board", name="Proj Board", project_id="p_xyz")
    assert meta["project_id"] == "p_xyz"
    assert kb.read_board_metadata("proj-board")["project_id"] == "p_xyz"


def test_create_task_inherits_board_project(fresh_home, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Widget", primary_path=str(repo))

    kb.create_board("scoped", name="Scoped", project_id=proj_id)
    conn = kbc.connect(board="scoped")
    try:
        tid = kb.create_task(conn, title="inherit me", board="scoped")
        assert kb.get_task(conn, tid).project_id == proj_id
    finally:
        conn.close()


def test_create_task_explicit_project_beats_board(fresh_home, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pdb.connect_closing() as pconn:
        board_proj = pdb.create_project(pconn, name="BoardProj", primary_path=str(tmp_path / "a"))
        task_proj = pdb.create_project(pconn, name="TaskProj", primary_path=str(tmp_path / "b"))

    kb.create_board("scoped2", name="Scoped2", project_id=board_proj)
    conn = kbc.connect(board="scoped2")
    try:
        tid = kb.create_task(conn, title="explicit", board="scoped2", project_id=task_proj)
        assert kb.get_task(conn, tid).project_id == task_proj
    finally:
        conn.close()


def test_create_task_explicit_scratch_beats_board(fresh_home, tmp_path):
    """#106342: every surface (CLI --workspace scratch, dashboard workspace_kind,
    tool) funnels here; an explicit scratch on a project-scoped board must stay
    scratch, while an omitted kind still inherits the project worktree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Widget", primary_path=str(repo))

    kb.create_board("scoped3", name="Scoped3", project_id=proj_id)
    conn = kbc.connect(board="scoped3")
    try:
        scratch = kb.get_task(conn, kb.create_task(
            conn, title="explicit scratch", board="scoped3", workspace_kind="scratch"))
        assert (scratch.workspace_kind, scratch.project_id) == ("scratch", None)
        default = kb.get_task(conn, kb.create_task(conn, title="default", board="scoped3"))
        assert (default.workspace_kind, default.project_id) == ("worktree", proj_id)
    finally:
        conn.close()


def test_create_task_uses_the_store_board_not_the_session_board(fresh_home, tmp_path, monkeypatch):
    """A pinned store must not leak the *session's* board scope into the row.

    ``HERMES_KANBAN_DB`` is injected into every dispatcher-spawned worker and wins
    over the board slug (``_board_path``), so the store a create lands in can differ
    from the board the session sits on. The board-derived columns have to come from
    the store that actually received the task, not from ``get_current_board()``.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Widget", primary_path=str(repo))

    kb.create_board("scoped", name="Scoped", project_id=proj_id, default_workdir=str(elsewhere))
    kb.set_current_board("scoped")  # the session sits on the scoped board ...
    # ... but the store in play is the default board's, as it is for a spawned worker.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_home() / "kanban.db"))

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="pinned store")
        task = kb.get_task(conn, tid)
        assert (task.project_id, task.workspace_kind, task.branch_name) == (None, "scratch", None)

        # The row really landed in the pinned store (default board), not the scoped one.
        raw = sqlite3.connect(str(kb.kanban_db_path()))
        try:
            landed = raw.execute("SELECT count(*) FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        finally:
            raw.close()
        assert landed == 1

        # A persistent kind must not inherit the other board's default_workdir either.
        dir_task = kb.get_task(
            conn, kb.create_task(conn, title="pinned store dir", workspace_kind="dir"))
        assert dir_task.workspace_path is None
    finally:
        conn.close()


def test_create_task_inherits_from_the_store_board(fresh_home, tmp_path):
    """The inverse direction: with no ``board=`` argument the store still decides.

    A session whose active board is ``default`` must not strip the scope from a
    create against the scoped board's store — the explicit-argument path above is
    the only case where the caller gets to name the board.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Widget", primary_path=str(repo))

    kb.create_board("scoped2", name="Scoped2", project_id=proj_id)
    kb.set_current_board("default")
    conn = kbc.connect(board="scoped2")
    try:
        task = kb.get_task(conn, kb.create_task(conn, title="store decides"))
        assert (task.workspace_kind, task.project_id) == ("worktree", proj_id)
        assert task.branch_name
    finally:
        conn.close()
