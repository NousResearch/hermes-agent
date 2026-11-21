"""Default Kanban scratch roots live beside a dot-prefixed Hermes home.

Web-framework file senders reject an absolute path with a dot-directory
ancestor, so when the kanban home is itself a dot-directory (``~/.hermes``) scratch
workspaces default to ``<kanban-home-parent>/hermes-workspaces/<home-name>/<slug>``.
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
        ("/srv/u/.hermes", "/srv/u/hermes-workspaces/hermes/default", "/srv/u/hermes-workspaces/hermes/ops"),
        ("/srv/u/.hermes-work", "/srv/u/hermes-workspaces/hermes-work/default", "/srv/u/hermes-workspaces/hermes-work/ops"),
        # A dot-directory ABOVE the home does not move scratch: the new root would
        # still sit below it.
        # A home directly under the filesystem root keeps the legacy roots.
        ("/.hermes", "/.hermes/kanban/workspaces", "/.hermes/kanban/boards/ops/workspaces"),
        ("/srv/.cfg/hermes", "/srv/.cfg/hermes/kanban/workspaces", "/srv/.cfg/hermes/kanban/boards/ops/workspaces"),
        ("/opt/data", "/opt/data/kanban/workspaces", "/opt/data/kanban/boards/ops/workspaces"),
    ],
)
def test_scratch_root_moves_only_for_a_dot_dir_named_home(monkeypatch, home, default_root, ops_root):
    """Only a home that is itself a dot-directory (with no dot-directory above it)
    moves scratch beside it, under its own name; any other home
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


def test_hard_delete_leaves_a_sibling_installs_scratch_alone(tmp_path, monkeypatch):
    """Two homes under one parent (``.hermes``, ``.hermes-work``) get separate
    scratch roots, so hard-deleting a board in one leaves the other's in-flight
    scratch of the same slug untouched."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    roots = {}
    for name in (".hermes", ".hermes-work"):
        home = tmp_path / name
        home.mkdir()
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
        kb.init_db()
        kb.create_board("ops")
        roots[name] = kb.workspaces_root(board="ops")
        (roots[name] / "t_live").mkdir(parents=True)
    assert roots[".hermes"] != roots[".hermes-work"]

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / ".hermes"))
    kb.remove_board("ops", archive=False)

    assert not roots[".hermes"].exists()
    assert (roots[".hermes-work"] / "t_live").is_dir()


def test_hard_delete_leaves_a_pinned_scratch_root_alone(kanban_home, tmp_path, monkeypatch):
    """``HERMES_KANBAN_WORKSPACES_ROOT`` names a root the operator owns; a board
    delete must remove only the default root, never the pinned one."""
    pinned = tmp_path / "pinned"
    (pinned / "t_1").mkdir(parents=True)
    kb.create_board("gone")
    default_root = kbw.default_workspaces_root("gone")
    (default_root / "t_2").mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(pinned))

    kb.remove_board("gone", archive=False)

    assert (pinned / "t_1").is_dir()
    assert not default_root.exists()


def test_non_dot_home_scratch_is_managed_and_cleaned_up(tmp_path, monkeypatch):
    """A Docker-style home (``/opt/data``) keeps the legacy roots, and those are
    recognised as managed scratch so completion cleans them up."""
    home = tmp_path / "data"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    kb.init_db()
    with kbc.connect() as conn:
        ws = _complete_scratch_task(conn)
    assert ws.is_relative_to(home / "kanban" / "workspaces")
    assert not ws.exists()
