import sqlite3

from gateway.run import _is_fatal_kanban_board_db_error
from hermes_cli.kanban_db import KanbanDbCorruptError


def test_disk_io_error_is_fatal_for_kanban_board_dispatch():
    assert _is_fatal_kanban_board_db_error(
        sqlite3.OperationalError("disk I/O error")
    )


def test_malformed_database_is_fatal_for_kanban_board_dispatch():
    assert _is_fatal_kanban_board_db_error(
        sqlite3.DatabaseError("database disk image is malformed")
    )


def test_integrity_guard_corrupt_error_is_fatal_for_kanban_board_dispatch(tmp_path):
    assert _is_fatal_kanban_board_db_error(
        KanbanDbCorruptError(
            tmp_path / "kanban.db",
            tmp_path / "kanban.db.corrupt.bak",
            "integrity_check returned malformed page tree",
        )
    )


def test_busy_locked_database_is_not_classified_as_corruption():
    assert not _is_fatal_kanban_board_db_error(
        sqlite3.OperationalError("database is locked")
    )
