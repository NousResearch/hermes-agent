import sqlite3

from gateway.run import _is_fatal_kanban_board_db_error


def test_disk_io_error_is_fatal_for_kanban_board_dispatch():
    assert _is_fatal_kanban_board_db_error(
        sqlite3.OperationalError("disk I/O error")
    )


def test_malformed_database_is_fatal_for_kanban_board_dispatch():
    assert _is_fatal_kanban_board_db_error(
        sqlite3.DatabaseError("database disk image is malformed")
    )


def test_busy_locked_database_is_not_classified_as_corruption():
    assert not _is_fatal_kanban_board_db_error(
        sqlite3.OperationalError("database is locked")
    )
