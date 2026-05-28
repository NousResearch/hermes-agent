from __future__ import annotations

import sqlite3
from types import SimpleNamespace


class _FakeCorruptError(RuntimeError):
    pass


class _FakeUnavailableError(sqlite3.DatabaseError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__("wrapped board DB unavailable")


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fake_kb(tmp_path, *, connect_exc=None, dispatch_exc=None):
    db_path = tmp_path / "kanban.db"
    db_path.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    conn = _FakeConn()
    calls = {"connect": 0, "dispatch_once": 0}

    def kanban_db_path(slug):
        assert slug == "superoptions"
        return db_path

    def connect(*, board):
        assert board == "superoptions"
        calls["connect"] += 1
        if connect_exc is not None:
            raise connect_exc
        return conn

    def dispatch_once(conn_arg, **kwargs):
        assert conn_arg is conn
        assert kwargs["board"] == "superoptions"
        calls["dispatch_once"] += 1
        if dispatch_exc is not None:
            raise dispatch_exc
        return SimpleNamespace(spawned=[])

    kb = SimpleNamespace(
        KanbanDbCorruptError=_FakeCorruptError,
        kanban_db_path=kanban_db_path,
        connect=connect,
        dispatch_once=dispatch_once,
    )
    return kb, conn, calls, db_path


def test_kanban_board_tick_quarantines_wal_disk_io_open_failure(tmp_path, caplog):
    """A WAL/open disk I/O failure should pause only that board, without tracebacks."""
    from gateway.run import _dispatch_kanban_board_once

    kb, _conn, calls, db_path = _fake_kb(
        tmp_path,
        connect_exc=sqlite3.OperationalError("disk I/O error"),
    )
    disabled = {}

    with caplog.at_level("ERROR", logger="gateway.run"):
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None

    assert calls["connect"] == 1, "second tick should be suppressed by board quarantine"
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1, messages
    assert "board=superoptions" in messages[0]
    assert str(db_path) in messages[0]
    assert "disk I/O" in messages[0]
    assert "WAL/SHM" in messages[0]
    assert "operation=connect" in messages[0]
    assert caplog.records[0].board_slug == "superoptions"
    assert caplog.records[0].db_path == str(db_path)
    assert caplog.records[0].operation == "connect"
    assert caplog.records[0].error_class == "OperationalError"
    assert caplog.records[0].sqlite_failure == "disk_io"
    assert caplog.records[0].quarantined is True
    assert not caplog.records[0].exc_info, "quarantined board diagnostics must not traceback every tick"


def test_kanban_board_tick_logs_original_error_class_from_board_diagnostic(tmp_path, caplog):
    """connect() wrappers should not hide the original SQLite class/message."""
    from gateway.run import _dispatch_kanban_board_once

    diagnostic = SimpleNamespace(
        category="disk_io",
        quarantine=True,
        operator_action="repair or restore the board DB and WAL/SHM sidecars",
        error_class="OperationalError",
        error_message="disk I/O error",
    )
    kb, _conn, calls, _db_path = _fake_kb(
        tmp_path,
        connect_exc=_FakeUnavailableError(diagnostic),
    )
    disabled = {}

    with caplog.at_level("ERROR", logger="gateway.run"):
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None

    assert calls == {"connect": 1, "dispatch_once": 0}
    assert caplog.records[0].error_class == "OperationalError"
    assert caplog.records[0].error_message == "disk I/O error"
    assert "error_class=OperationalError" in caplog.records[0].getMessage()
    assert "wrapped board DB unavailable" not in caplog.records[0].getMessage()


def test_kanban_board_tick_quarantines_release_stale_claims_read_failure(tmp_path, caplog):
    """A dispatch/read disk I/O failure should close the connection and quarantine the board."""
    from gateway.run import _dispatch_kanban_board_once

    kb, conn, calls, db_path = _fake_kb(
        tmp_path,
        dispatch_exc=sqlite3.OperationalError("disk I/O error"),
    )
    disabled = {}

    with caplog.at_level("ERROR", logger="gateway.run"):
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None

    assert calls == {"connect": 1, "dispatch_once": 1}
    assert conn.closed is True
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1, messages
    assert "board=superoptions" in messages[0]
    assert str(db_path) in messages[0]
    assert "disk I/O" in messages[0]
    assert "WAL/SHM" in messages[0]
    assert "operation=dispatch_once" in messages[0]
    assert caplog.records[0].operation == "dispatch_once"
    assert caplog.records[0].sqlite_failure == "disk_io"
    assert caplog.records[0].quarantined is True
    assert not caplog.records[0].exc_info


def test_kanban_board_tick_retries_busy_locked_without_quarantine(tmp_path, caplog):
    """Lock/busy failures are transient: log once per tick and do not quarantine."""
    from gateway.run import _dispatch_kanban_board_once

    kb, _conn, calls, db_path = _fake_kb(
        tmp_path,
        dispatch_exc=sqlite3.OperationalError("database is locked"),
    )
    disabled = {}

    with caplog.at_level("WARNING", logger="gateway.run"):
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None

    assert calls == {"connect": 2, "dispatch_once": 2}
    assert disabled == {}
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2, messages
    assert all("sqlite_failure=busy_locked" in msg for msg in messages)
    assert all("skipping this tick" in msg for msg in messages)
    assert caplog.records[0].board_slug == "superoptions"
    assert caplog.records[0].db_path == str(db_path)
    assert caplog.records[0].operation == "dispatch_once"
    assert caplog.records[0].error_class == "OperationalError"
    assert caplog.records[0].quarantined is False
    assert not caplog.records[0].exc_info


def test_kanban_board_tick_quarantines_corrupt_guard_error(tmp_path, caplog):
    """The kanban_db integrity/header guard maps to board-local quarantine."""
    from gateway.run import _dispatch_kanban_board_once

    kb, _conn, calls, db_path = _fake_kb(
        tmp_path,
        connect_exc=_FakeCorruptError("not a valid SQLite database"),
    )
    disabled = {}

    with caplog.at_level("ERROR", logger="gateway.run"):
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None
        assert _dispatch_kanban_board_once(
            kb,
            "superoptions",
            disabled_boards=disabled,
            quarantine_retry_after_seconds=300,
        ) is None

    assert calls == {"connect": 1, "dispatch_once": 0}
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1, messages
    assert "sqlite_failure=corrupt" in messages[0]
    assert "operation=connect" in messages[0]
    assert "restore or move aside" in messages[0]
    assert caplog.records[0].board_slug == "superoptions"
    assert caplog.records[0].db_path == str(db_path)
    assert caplog.records[0].error_class == "_FakeCorruptError"
    assert caplog.records[0].sqlite_failure == "corrupt"
    assert caplog.records[0].quarantined is True
    assert not caplog.records[0].exc_info
