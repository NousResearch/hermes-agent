"""Behavioral tests for delegation context and Kanban write fencing."""
import os
import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    _fenced_kanban_root,
    kanban_path_is_fenced,
    scrub_kanban_env,
)
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def kanban_env(tmp_path, monkeypatch):
    """Isolated shared home with initialized boards."""
    home = tmp_path / "kanban_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    for key in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_TASK",
    ):
        monkeypatch.delenv(key, raising=False)
    kb.create_board("board-a")
    kb.create_board("board-b")
    return home


def test_worker_on_board_a_fenced_on_board_a_not_board_b(
    kanban_env, monkeypatch
):
    """Worker on board A is fenced on board A's DB, but not on board B's DB."""
    db_a = kb.kanban_db_path(board="board-a")
    db_b = kb.kanban_db_path(board="board-b")
    db_default = kb.kanban_db_path(board="default")

    conn_a = connect(board="board-a")
    task_a = kb.create_task(conn_a, title="task on a")
    conn_a.close()

    worker_env = {
        "HERMES_KANBAN_BOARD": "board-a",
        "HERMES_KANBAN_DB": str(db_a),
        "HERMES_KANBAN_TASK": task_a,
    }

    child_env = scrub_kanban_env(worker_env)
    expected_root = str(kb.board_dir("board-a").resolve())
    assert child_env[DELEGATED_CHILD_ENV_MARKER] == expected_root
    assert "HERMES_KANBAN_TASK" not in child_env

    # Run in descendant process context
    monkeypatch.setenv(
        DELEGATED_CHILD_ENV_MARKER, child_env[DELEGATED_CHILD_ENV_MARKER]
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)

    assert kanban_path_is_fenced(db_a)
    assert not kanban_path_is_fenced(db_b)
    assert not kanban_path_is_fenced(db_default)

    # Board metadata under board directory is fenced only for board A
    assert kanban_path_is_fenced(kb.board_metadata_path(board="board-a"))
    assert not kanban_path_is_fenced(kb.board_metadata_path(board="board-b"))

    # Real SQLite mutation: board B write succeeds, board A write is denied
    conn_b = connect(board="board-b")
    task_b = kb.create_task(conn_b, title="written by descendant on b")
    assert kb.get_task(conn_b, task_b).title == "written by descendant on b"
    conn_b.close()

    conn_a_ro = connect(board="board-a")
    with pytest.raises(PermissionError):
        kb.create_task(conn_a_ro, title="should fail on a")
    conn_a_ro.close()


def test_fenced_kanban_root_derives_from_pinned_db(kanban_env, monkeypatch):
    """When only HERMES_KANBAN_DB is set, the lineage root derives from it."""
    db_a = kb.kanban_db_path(board="board-a")
    db_b = kb.kanban_db_path(board="board-b")

    env = {"HERMES_KANBAN_DB": str(db_a)}
    root = _fenced_kanban_root(env)
    assert root == str(kb.board_dir("board-a").resolve())

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, root)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    assert kanban_path_is_fenced(db_a)
    assert not kanban_path_is_fenced(db_b)


def test_fenced_kanban_root_derives_from_task_membership(
    kanban_env, monkeypatch
):
    """When only HERMES_KANBAN_TASK is set, search boards for its root."""
    db_a = kb.kanban_db_path(board="board-a")
    db_b = kb.kanban_db_path(board="board-b")

    conn_a = connect(board="board-a")
    task_a = kb.create_task(conn_a, title="task on a")
    conn_a.close()

    env = {"HERMES_KANBAN_TASK": task_a}
    root = _fenced_kanban_root(env)
    assert root == str(kb.board_dir("board-a").resolve())

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, root)
    assert kanban_path_is_fenced(db_a)
    assert not kanban_path_is_fenced(db_b)


def test_worker_on_default_board_not_fenced_on_named_board(
    kanban_env, monkeypatch
):
    """Worker on default board is not fenced on named board B."""
    db_default = kb.kanban_db_path(board="default")
    db_b = kb.kanban_db_path(board="board-b")

    env = {
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_DB": str(db_default),
    }
    root = _fenced_kanban_root(env)
    assert root == str(kb.kanban_home().resolve())

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, root)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    assert kanban_path_is_fenced(db_default)
    assert not kanban_path_is_fenced(db_b)


def test_inherited_marker_preserved(kanban_env):
    """An existing valid path marker in env is kept without re-deriving."""
    custom_marker = str(kanban_env / "custom-root")
    env = {
        DELEGATED_CHILD_ENV_MARKER: custom_marker,
        "HERMES_KANBAN_BOARD": "board-a",
    }
    cleaned = scrub_kanban_env(env)
    assert cleaned[DELEGATED_CHILD_ENV_MARKER] == custom_marker
