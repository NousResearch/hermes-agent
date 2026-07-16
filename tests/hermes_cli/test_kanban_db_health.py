"""Board-wide Kanban SQLite health circuit and error taxonomy."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from tools import kanban_tools


def _sqlite_error(message: str, code: int, name: str) -> sqlite3.OperationalError:
    exc = sqlite3.OperationalError(message)
    exc.sqlite_errorcode = code
    exc.sqlite_errorname = name
    return exc


def _clear_process_state(path: Path) -> None:
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))


def test_classifier_distinguishes_busy_storage_corruption_and_application_errors():
    busy = kb.classify_sqlite_error(
        _sqlite_error("database is locked", sqlite3.SQLITE_BUSY, "SQLITE_BUSY")
    )
    ioerr = kb.classify_sqlite_error(
        _sqlite_error("disk I/O error", sqlite3.SQLITE_IOERR, "SQLITE_IOERR")
    )
    corrupt = kb.classify_sqlite_error(
        _sqlite_error(
            "database disk image is malformed",
            sqlite3.SQLITE_CORRUPT,
            "SQLITE_CORRUPT",
        )
    )
    constraint = kb.classify_sqlite_error(
        _sqlite_error(
            "UNIQUE constraint failed",
            sqlite3.SQLITE_CONSTRAINT,
            "SQLITE_CONSTRAINT",
        )
    )

    assert busy.category == "transient"
    assert busy.retryable is True
    assert busy.fatal is False
    assert ioerr.category == "fatal_storage" and ioerr.fatal is True
    assert corrupt.category == "corruption" and corrupt.fatal is True
    assert constraint.category == "application" and constraint.fatal is False


def test_ioerr_trips_one_persisted_incident_and_stops_reopen_attempts(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    _clear_process_state(db_path)

    calls = 0

    def fail_open(_path):
        nonlocal calls
        calls += 1
        raise _sqlite_error("disk I/O error", sqlite3.SQLITE_IOERR, "SQLITE_IOERR")

    monkeypatch.setattr(kb, "_sqlite_connect", fail_open)

    incidents = set()
    manifests = set()
    for _ in range(6):
        with pytest.raises(kb.KanbanDbHealthError) as excinfo:
            kb.connect(db_path=db_path)
        incidents.add(excinfo.value.incident_id)
        manifests.add(excinfo.value.manifest_path)

    assert calls == 1, "an open circuit must prevent repeated SQLite open attempts"
    assert len(incidents) == 1
    assert len(manifests) == 1
    manifest_path = next(iter(manifests))
    assert manifest_path is not None and manifest_path.exists()
    assert len(list(tmp_path.glob("kanban.db.incident.*.json"))) == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["incident_id"] == next(iter(incidents))
    assert manifest["classification"] == "fatal_storage"


def test_corrupt_guard_reuses_incident_backup_and_manifest(tmp_path):
    db_path = tmp_path / "kanban.db"
    header = b"SQLite format 3\x00".ljust(100, b"\x00")
    db_path.write_bytes(header + (b"broken page" * 256))
    _clear_process_state(db_path)

    errors = []
    for _ in range(3):
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)
        errors.append(excinfo.value)

    assert len({e.incident_id for e in errors}) == 1
    assert len({e.backup_path for e in errors}) == 1
    assert len({e.manifest_path for e in errors}) == 1
    assert len(list(tmp_path.glob("kanban.db.corrupt.*.bak"))) == 1
    assert len(list(tmp_path.glob("kanban.db.incident.*.json"))) == 1


def test_direct_tool_and_cli_diagnostics_report_same_incident(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    _clear_process_state(db_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    def fail_open(_path):
        raise _sqlite_error(
            "database disk image is malformed",
            sqlite3.SQLITE_CORRUPT,
            "SQLITE_CORRUPT",
        )

    monkeypatch.setattr(kb, "_sqlite_connect", fail_open)
    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        kb.connect(db_path=db_path)
    incident_id = excinfo.value.incident_id

    tool_result = json.loads(kanban_tools._handle_show({"task_id": "t_missing"}))
    assert incident_id in tool_result["error"]
    assert "quarantined" in tool_result["error"]

    monkeypatch.setattr(
        kb,
        "init_db",
        lambda *args, **kwargs: pytest.fail(
            "diagnostics must not open a quarantined database"
        ),
    )
    rc = kanban_cli.kanban_command(
        argparse.Namespace(
            kanban_action="diagnostics",
            board=None,
            task=None,
            severity=None,
            json=True,
        )
    )
    captured = capsys.readouterr()
    assert rc == 2
    cli_result = json.loads(captured.out)
    assert cli_result["board_health"]["incident_id"] == incident_id
    assert cli_result["board_health"]["manifest_path"] == str(
        excinfo.value.manifest_path
    )


def test_fatal_write_error_trips_shared_circuit(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    conn = kb.connect(db_path=db_path)
    try:
        with pytest.raises(kb.KanbanDbHealthError) as excinfo:
            with kb.write_txn(conn):
                raise _sqlite_error(
                    "disk I/O error",
                    sqlite3.SQLITE_IOERR,
                    "SQLITE_IOERR",
                )
    finally:
        conn.close()

    health = kb.get_db_health(db_path)
    assert health is not None
    assert health["incident_id"] == excinfo.value.incident_id
    assert health["classification"] == "fatal_storage"


def test_atomic_database_replacement_clears_stale_circuit(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    replacement = tmp_path / "replacement.db"
    kb.init_db(db_path=db_path)
    kb.init_db(db_path=replacement)
    _clear_process_state(db_path)

    def fail_open(_path):
        raise _sqlite_error(
            "disk I/O error",
            sqlite3.SQLITE_IOERR,
            "SQLITE_IOERR",
        )

    real_connect = kb._sqlite_connect
    monkeypatch.setattr(kb, "_sqlite_connect", fail_open)
    with pytest.raises(kb.KanbanDbHealthError):
        kb.connect(db_path=db_path)
    monkeypatch.setattr(kb, "_sqlite_connect", real_connect)

    os.replace(replacement, db_path)
    _clear_process_state(db_path)
    assert kb.get_db_health(db_path) is None
    with kb.connect(db_path=db_path) as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
