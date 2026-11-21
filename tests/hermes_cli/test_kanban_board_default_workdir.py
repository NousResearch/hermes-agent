"""A board's ``default_workdir`` inherited by a new card.

Only persistent kinds (``dir``/``worktree``) inherit it — a ``scratch`` card
inheriting it would point cleanup at the user's source tree — and a
``worktree`` card inherits it only when dispatch can actually materialize a
worktree there. The board default used to be copied *after*
``resolve_project_link`` had returned, so a board whose default workdir was
not inside a git repo accepted a ``worktree`` card and then killed it at spawn
(``board '<slug>' default_workdir ... is not inside a git repo``). Creation now
applies the same rule dispatch does, and still accepts every shape dispatch can
resolve.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import projects_db as pdb


@pytest.fixture
def fresh_kanban_home(tmp_path, monkeypatch):
    """An empty kanban home per test: board metadata and DBs are per-home."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    return home


@pytest.fixture
def conn():
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


def _git_init(repo):
    """A real git work tree — ``git init`` alone is what a worktree anchor needs.

    No commit: the predicate is "is this path inside a work tree", which an
    unborn branch satisfies, and no worktree is materialized at creation.
    """
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _non_repo(path):
    """A folder that exists but is not inside a git work tree."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _board_with_default(slug, default_workdir):
    kb.create_board(slug, name=slug.replace("-", " ").title(),
                    default_workdir=str(default_workdir))
    assert kb.read_board_metadata(slug)["default_workdir"] == str(default_workdir)
    return slug


def _task_rows(c):
    return c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]


def test_board_default_workdir_outside_a_repo_with_worktree_raises_at_creation(
    fresh_kanban_home, conn, tmp_path,
):
    """A ``worktree`` card inheriting a default outside any repo cannot be
    dispatched: refuse it at creation, naming the board and the path, instead of
    creating a card that dies at spawn."""
    default = _non_repo(tmp_path / "not-a-repo")
    board = _board_with_default("bad-default", default)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(conn, title="stranded", board=board, workspace_kind="worktree")

    message = str(excinfo.value)
    assert board in message
    assert str(default) in message
    assert "not inside a git repo" in message
    # Refused before the insert: no half-created card is left behind.
    assert _task_rows(conn) == 0


def test_board_default_workdir_inside_a_repo_is_inherited(fresh_kanban_home, conn, tmp_path):
    """Legitimate case: the default can host a worktree, so the card inherits it."""
    repo = _git_init(tmp_path / "mono")
    board = _board_with_default("good-default", repo)

    tid = kb.create_task(conn, title="inherits", board=board, workspace_kind="worktree")
    task = kb.get_task(conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(repo)
    assert task.project_id is None


def test_board_default_workdir_outside_a_repo_is_inherited_by_dir_workspaces(
    fresh_kanban_home, conn, tmp_path,
):
    """Only ``worktree`` needs a git anchor: a ``dir`` card keeps inheriting a
    plain-directory default (dispatch resolves a ``dir`` workspace anywhere)."""
    default = _non_repo(tmp_path / "shared" / "work")
    board = _board_with_default("plain-dir", default)

    tid = kb.create_task(conn, title="persistent dir", board=board, workspace_kind="dir")
    task = kb.get_task(conn, tid)

    assert task.workspace_kind == "dir"
    assert task.workspace_path == str(default)


def test_scratch_does_not_inherit_board_default_workdir(fresh_kanban_home, conn, tmp_path):
    """A ``scratch`` card inheriting the board default would point cleanup at the
    user's source tree — it must stay pathless even when the default is bogus."""
    default = _non_repo(tmp_path / "not-a-repo")
    board = _board_with_default("scratchy", default)

    tid = kb.create_task(conn, title="ad hoc", board=board)
    task = kb.get_task(conn, tid)

    assert task.workspace_kind == "scratch"
    assert task.workspace_path is None


def test_explicit_workspace_path_beats_board_default_workdir(fresh_kanban_home, conn, tmp_path):
    """An explicit path overrides the default at dispatch, so it wins here too —
    and a usable explicit path is accepted despite a broken board default."""
    repo = _git_init(tmp_path / "mono")
    board = _board_with_default("explicit-wins", _non_repo(tmp_path / "not-a-repo"))
    target = repo / ".worktrees" / "chosen"

    tid = kb.create_task(
        conn, title="explicit", board=board, workspace_kind="worktree",
        workspace_path=str(target),
    )
    task = kb.get_task(conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(target)


def test_project_anchor_beats_board_default_workdir(fresh_kanban_home, conn, tmp_path):
    """A project's repo owns the worktree path, so a broken board default is
    never consulted: the deterministic ``<repo>/.worktrees/<id>`` still wins."""
    repo = _git_init(tmp_path / "webapp")
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Web App", folders=[str(repo)])
        proj = pdb.get_project(pconn, proj_id)
    board = _board_with_default("project-default", _non_repo(tmp_path / "not-a-repo"))

    tid = kb.create_task(conn, title="anchored", board=board, project_id=proj.id)
    task = kb.get_task(conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == os.path.join(str(repo), ".worktrees", tid)
    assert task.branch_name == f"{proj.slug}/{tid}-anchored"
