"""The cross-board workspace-sharing scan must not fail closed on a stray
board file that holds no tasks (a 0-byte ``kanban.db`` left by an ad-hoc
``sqlite3`` open), but must still fail closed on a real board it cannot read."""
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="probe")
    return tmp_path


def _in_use(path) -> object:
    with kbc.connect_closing() as conn:
        return kbw._workspace_in_use_by_other(conn, "t_probe", path)


def _stray_board_file(slug: str) -> Path:
    db = kb.boards_root() / slug / "kanban.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return db


def test_empty_file_under_boards_default_is_not_a_board(board, tmp_path):
    _stray_board_file("default").touch()
    assert _in_use(tmp_path / "ws") is None


def test_named_board_file_without_tasks_table_is_not_a_board(board, tmp_path):
    _stray_board_file("empty").touch()
    other = sqlite3.connect(_stray_board_file("other"))
    other.execute("CREATE TABLE unrelated (x)")
    other.commit()
    other.close()
    assert _in_use(tmp_path / "ws") is None
    assert (kb.boards_root() / "empty" / "kanban.db").stat().st_size == 0


def test_corrupt_board_db_still_fails_closed(board, tmp_path):
    _stray_board_file("broken").write_bytes(b"not a sqlite database" * 64)
    assert _in_use(tmp_path / "ws") == "unknown"


def test_locked_board_db_still_fails_closed(board, tmp_path):
    db = _stray_board_file("busy")
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("PRAGMA journal_mode=DELETE")
    holder.execute("CREATE TABLE tasks (id TEXT, status TEXT, workspace_path TEXT)")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        assert _in_use(tmp_path / "ws") == "unknown"
    finally:
        holder.execute("ROLLBACK")
        holder.close()
