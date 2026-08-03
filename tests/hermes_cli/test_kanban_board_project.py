"""Board→project scoping in kanban_db.

A kanban board can be scoped to a first-class Hermes project so every task on
it anchors to that project (deterministic worktree + branch). Covers the
metadata round-trip and the create-time inheritance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
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
    conn = kb.connect(board="scoped")
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
    conn = kb.connect(board="scoped2")
    try:
        tid = kb.create_task(conn, title="explicit", board="scoped2", project_id=task_proj)
        assert kb.get_task(conn, tid).project_id == task_proj
    finally:
        conn.close()
