"""Ownership scans during workspace cleanup follow board identity.

Regression for #160064. ``boards/_archived/`` and a database-only stub are
not boards, so an empty ``kanban.db`` there must not defer scratch cleanup.
A real board whose database cannot be read still blocks deletion.
"""

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    return tmp_path


def _empty_db(path: Path) -> None:
    """A zero-byte SQLite file: read-only ``SELECT`` raises ``no such table: tasks``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _archived_scratch(conn) -> tuple[str, Path]:
    tid = kb.create_task(conn, title="archived scratch")
    ws = kb.workspaces_root() / tid
    ws.mkdir(parents=True)
    (ws / "note.txt").write_text("done", encoding="utf-8")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='archived', workspace_kind='scratch', "
            "workspace_path=? WHERE id=?",
            (str(ws), tid),
        )
    return tid, ws


def _deferred_reasons(conn, task_id: str) -> list:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id",
        (task_id, "workspace_cleanup_deferred_shared"),
    ).fetchall()
    return [json.loads(r["payload"])["reason"] for r in rows]


def test_cleanup_ignores_empty_db_in_archived_dir_and_stub(board):
    """An empty db under ``_archived`` or a stub dir is not another board."""
    _empty_db(kb.boards_root() / "_archived" / "kanban.db")
    _empty_db(kb.board_dir("ghost") / "kanban.db")
    assert not (kb.board_dir("ghost") / "board.json").exists()

    with kbc.connect_closing() as conn:
        tid, ws = _archived_scratch(conn)
        kbw._cleanup_workspace(conn, tid)
        assert _deferred_reasons(conn, tid) == []
    assert not (ws / "note.txt").exists()


def test_cleanup_defers_when_active_board_db_is_unreadable(board):
    """A canonical board (``board.json`` present) with an unreadable db still defers."""
    kb.create_board("live")
    db_path = kb.board_dir("live") / "kanban.db"
    assert (db_path.parent / "board.json").is_file()
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    _empty_db(db_path)

    with kbc.connect_closing() as conn:
        tid, ws = _archived_scratch(conn)
        kbw._cleanup_workspace(conn, tid)
        assert _deferred_reasons(conn, tid) == ["unknown"]
    assert (ws / "note.txt").read_text(encoding="utf-8") == "done"
