"""Default Kanban scratch roots live beside a dot-prefixed Hermes home.

Web-framework file senders reject an absolute path with a dot-directory
ancestor, so when the kanban home is below one (``~/.hermes``) scratch
workspaces default to ``<kanban-home-parent>/hermes-workspaces/<slug>``.
Any other home (a container's ``/opt/data``) keeps the legacy
``kanban/.../workspaces`` roots, which also stay managed for older tasks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    kb.init_db()
    return home


def _complete_scratch_task(conn, board=None) -> Path:
    t = kb.create_task(conn, title="scratch")
    task = kb.get_task(conn, t)
    assert task is not None
    ws = kbw.resolve_workspace(task, board=board)
    kbw.set_workspace_path(conn, t, ws)
    assert kb._is_managed_scratch_path(ws)
    assert kb.complete_task(conn, t, result="done")
    return ws


def test_default_scratch_root_has_no_dot_dir_ancestor(kanban_home, tmp_path):
    kb.create_board("ops")
    for board in (kb.DEFAULT_BOARD, "ops"):
        root = kb.workspaces_root(board=board)
        assert root == kbw.default_workspaces_root(board)
        assert root.is_relative_to(tmp_path)
        assert not any(part.startswith(".") for part in root.relative_to(tmp_path).parts)


@pytest.mark.parametrize(
    ("home", "default_root", "ops_root"),
    [
        ("/srv/u/.hermes", "/srv/u/hermes-workspaces/default", "/srv/u/hermes-workspaces/ops"),
        ("/opt/data", "/opt/data/kanban/workspaces", "/opt/data/kanban/boards/ops/workspaces"),
    ],
)
def test_scratch_root_moves_only_for_a_dot_dir_home(monkeypatch, home, default_root, ops_root):
    """Only a home below a dot-directory moves scratch beside it; any other home
    keeps the legacy roots, since its parent (root-owned ``/opt`` in the Docker
    image) may be unwritable or outside the data volume."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", home)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    assert kb.workspaces_root(board=kb.DEFAULT_BOARD) == Path(default_root)
    assert kb.workspaces_root(board="ops") == Path(ops_root)


def test_archived_boards_dir_does_not_break_scratch_cleanup(kanban_home):
    """Archiving a board creates ``boards/_archived/``, which is not a valid
    board slug. The managed-root walk must skip it rather than raise, or every
    completion that checks scratch storage fails and scratch dirs leak."""
    kb.create_board("old")
    kb.remove_board("old", archive=True)
    assert (kb.boards_root() / "_archived").is_dir()

    with kbc.connect() as conn:
        ws = _complete_scratch_task(conn)
    assert not ws.exists()


def test_legacy_scratch_root_stays_managed(kanban_home):
    """Tasks created before the move still have their workspace cleaned up."""
    legacy = kanban_home / "kanban" / "workspaces" / "t_legacy"
    legacy.mkdir(parents=True)
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="scratch")
        kbw.set_workspace_path(conn, t, legacy)
        assert kb.complete_task(conn, t, result="done")
    assert not legacy.exists()


def test_deleting_a_board_removes_its_scratch_root(kanban_home):
    """The scratch root lives outside the board dir, so a hard delete must
    remove it too instead of orphaning it; other boards' roots are kept."""
    kb.create_board("gone")
    root = kb.workspaces_root(board="gone")
    (root / "t_1").mkdir(parents=True)
    keep = kb.workspaces_root(board=kb.DEFAULT_BOARD)
    keep.mkdir(parents=True, exist_ok=True)

    kb.remove_board("gone", archive=False)

    assert not root.exists()
    assert keep.is_dir()
