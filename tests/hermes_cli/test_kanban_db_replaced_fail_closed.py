"""A deleted or replaced board DB fails closed — it is never silently re-initialized.

Regression: a board's DB file was deleted while the gateway was running.
``connect()`` had the path in its process-local
``_INITIALIZED_PATHS`` cache, the open recreated an empty SQLite file, and the stale
cache entry was dropped so the full init path re-ran ``SCHEMA_SQL``. The board came
back EMPTY and every card on it was gone, with no error raised anywhere: the loss was
only visible as "my tasks disappeared". Recreating an empty board is never the right
recovery for a replaced DB — the operator restores a backup, or asks for a fresh
board explicitly. These tests pin that contract at each entry point:

* the cached fast path (``connect()`` after this process initialized the path),
* a fresh process (nothing cached — the durable per-board marker decides),
* the explicit admin path (``init_db(..., allow_recreate=True)``), which must still work.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture(autouse=True)
def _clean_module_caches():
    """Module-level caches are process-global; the runner isolates per FILE only."""
    kbc._INITIALIZED_PATHS.clear()
    kbc._REPLACED_BOARDS.clear()
    yield
    kbc._INITIALIZED_PATHS.clear()
    kbc._REPLACED_BOARDS.clear()


@pytest.fixture
def board_db(tmp_path, monkeypatch) -> Path:
    """A default-board path under a temp HERMES_HOME (nothing created yet)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kbc._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return db_path


def _initialize_with_one_task(db_path: Path) -> str:
    """Init the board through the normal path and leave one card on it."""
    with kbc.connect_closing(db_path) as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at) "
            "VALUES ('t-1', 'keep me', 'ready', 1000)"
        )
        conn.commit()
    assert str(db_path.resolve()) in kbc._INITIALIZED_PATHS
    return "t-1"


def _delete_db_files(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)


def _tables(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def test_replaced_db_records_the_event(board_db):
    """The refusal names what it saw, so a post-mortem does not have to guess."""
    _initialize_with_one_task(board_db)
    _delete_db_files(board_db)

    with pytest.raises(kbc.KanbanDbReplacedError):
        with kbc.connect_closing(board_db):
            pass

    events = kbc.replaced_board_events()
    assert len(events) == 1
    event = events[0]
    assert event.path == board_db.resolve()
    assert event.file_exists is False
    assert event.file_size is None
    assert event.page_count is None
    assert event.cached is True
    assert event.marker_present is True
    assert event.detected_at > 0
    assert event.detected_iso.endswith("+00:00")


def test_replaced_board_fails_closed_in_a_fresh_process(board_db):
    """Nothing cached (a new CLI/gateway process) must not create a fresh empty board."""
    _initialize_with_one_task(board_db)
    _delete_db_files(board_db)
    kbc._INITIALIZED_PATHS.discard(str(board_db.resolve()))  # fresh process

    with pytest.raises(kbc.KanbanDbReplacedError):
        with kbc.connect_closing(board_db):
            pass

    assert not board_db.exists()


def test_unrelated_sqlite_file_is_not_grafted_into_a_board(board_db):
    """A valid SQLite file that is not a kanban board is refused, not adopted.

    SCHEMA_SQL is additive, so running it on a foreign DB "works" and leaves the
    caller holding a board-shaped view of somebody else's file. Nothing about
    that is a recovery, so it is refused too.
    """
    _initialize_with_one_task(board_db)
    _delete_db_files(board_db)
    with sqlite3.connect(str(board_db)) as conn:
        conn.execute("CREATE TABLE something_else (id INTEGER)")

    # A fresh process: nothing cached, only the durable marker says a board lived
    # here, so this is the arm that has to inspect the bytes.
    kbc._INITIALIZED_PATHS.discard(str(board_db.resolve()))

    with pytest.raises(kbc.KanbanDbReplacedError):
        with kbc.connect_closing(board_db):
            pass

    tables = _tables(board_db)
    assert "tasks" not in tables
    assert "something_else" in tables


def test_first_time_create_still_initializes_a_board(board_db):
    """The guard is about REPLACEMENT — a board that never existed is still created."""
    assert not board_db.exists()
    with kbc.connect_closing(board_db) as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at) "
            "VALUES ('t-1', 'first', 'ready', 1000)"
        )
        conn.commit()

    assert "tasks" in _tables(board_db)
    assert kbc._init_marker_path(board_db).exists()
    # ...and the board it created is reusable, not a one-shot.
    with kbc.connect_closing(board_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_explicit_init_recreates_a_replaced_board(board_db):
    """`hermes kanban init` / board create stay the sanctioned way to start over."""
    _initialize_with_one_task(board_db)
    _delete_db_files(board_db)

    path = kbc.init_db(board_db, allow_recreate=True)

    assert path == board_db
    assert "tasks" in _tables(board_db)
    with kbc.connect_closing(board_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_repair_reports_a_replaced_board_rather_than_missing(board_db):
    """`hermes kanban repair` must not call a replaced board simply 'missing'."""
    _initialize_with_one_task(board_db)
    _delete_db_files(board_db)

    report = kbc.repair_db(board_db)

    assert report.status == "replaced"
    assert report.db_path == board_db.resolve()
