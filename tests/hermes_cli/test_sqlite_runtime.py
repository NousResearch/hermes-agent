"""Behavioral tests for exact-interpreter SQLite runtime inspection."""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_cli.sqlite_runtime import (
    is_sqlite_wal_reset_vulnerable,
    probe_sqlite_runtime,
)
from hermes_state import SessionDB


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((3, 6, 23), False),
        ((3, 7, 0), True),
        ((3, 44, 5), True),
        ((3, 44, 6), False),
        ((3, 45, 0), True),
        ((3, 50, 6), True),
        ((3, 50, 7), False),
        ((3, 51, 2), True),
        ((3, 51, 3), False),
        ((3, 53, 1), False),
    ],
)
def test_wal_reset_vulnerability_matrix(
    version: tuple[int, ...],
    expected: bool,
) -> None:
    assert is_sqlite_wal_reset_vulnerable(version) is expected


def test_probe_reports_the_requested_interpreters_linked_sqlite() -> None:
    info = probe_sqlite_runtime(sys.executable)

    assert info is not None
    assert info.executable.resolve() == Path(sys.executable).resolve()
    assert info.base_prefix.resolve() == Path(sys.base_prefix).resolve()
    assert info.python_version == sys.version_info[:3]
    assert info.sqlite_version == sqlite3.sqlite_version_info
    assert info.sqlite_version_string == sqlite3.sqlite_version

    with sqlite3.connect(":memory:") as conn:
        source_id = conn.execute("SELECT sqlite_source_id()").fetchone()[0]
    assert info.sqlite_source_id == source_id


def test_writable_session_db_uses_delete_on_vulnerable_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_state

    monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 51, 2))
    monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.51.2")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", source="test")
        db.append_message("session-1", "user", content="round-trip")
        session = db.get_session("session-1")
        assert session is not None
        assert session["id"] == "session-1"
        assert db.search_messages("round-trip")
    finally:
        db.close()


def test_writable_session_db_refuses_vulnerable_wal_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_state

    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (value TEXT)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 51, 2))
    monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.51.2")

    with pytest.raises(RuntimeError, match="journal_mode=WAL"):
        SessionDB(db_path=db_path)


def test_read_only_session_db_remains_available_on_vulnerable_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_state

    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version_info", (3, 51, 2))
    monkeypatch.setattr(hermes_state.sqlite3, "sqlite_version", "3.51.2")

    db = SessionDB(db_path=db_path, read_only=True)
    db.close()


def test_shared_state_direct_writers_use_delete_on_vulnerable_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.sqlite_runtime as runtime
    import gateway.delivery_ledger as delivery_ledger
    import tools.async_delegation as async_delegation

    monkeypatch.setattr(runtime, "is_sqlite_wal_reset_vulnerable", lambda _version: True)
    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: tmp_path / "delivery-state.db")
    monkeypatch.setattr(async_delegation, "_db_path", lambda: tmp_path / "delegation-state.db")

    delivery_conn = delivery_ledger._connect()
    delivery_conn.close()
    delegation_conn = async_delegation._connect()
    delegation_conn.close()
    assert (tmp_path / "delivery-state.db").exists()
    assert (tmp_path / "delegation-state.db").exists()

    monkeypatch.setattr(runtime, "is_sqlite_wal_reset_vulnerable", lambda _version: False)
    conn = async_delegation._connect()
    conn.close()
    monkeypatch.setattr(runtime, "is_sqlite_wal_reset_vulnerable", lambda _version: True)
    assert async_delegation.get_durable_delegation("missing") is None


def test_hosted_room_state_writers_refuse_vulnerable_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.sqlite_runtime as runtime
    from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
    from gateway.hosted_rooms_common import connect

    monkeypatch.setattr(runtime, "is_sqlite_wal_reset_vulnerable", lambda _version: True)
    state_path = tmp_path / "state.db"

    conn = connect(state_path, db_label="state.db", ready=lambda _conn: True, initialize=lambda _conn: None)
    conn.close()
    checkpoint = HostedRoomPolicyCheckpoint(state_path)
    assert checkpoint.db_path == state_path
    assert state_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX executable probe stub")
def test_probe_uses_child_payload_and_sanitizes_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "reported-python"
    payload = {
        "base_prefix": str(tmp_path / "reported-base"),
        "executable": str(fake_python),
        "python_version": [3, 11, 15],
        "sqlite_version": [9, 8, 7],
        "sqlite_version_string": "9.8.7-child",
        "sqlite_source_id": "child-source-id",
    }
    fake_python.write_text(
        "\n".join([
            "#!/bin/sh",
            '[ "$1" = "-I" ] && [ "$2" = "-c" ] || exit 10',
            '[ -z "${PYTHONHOME+x}" ] || exit 11',
            '[ -z "${PYTHONPATH+x}" ] || exit 12',
            f"printf '%s\\n' {shlex.quote(json.dumps(payload))}",
        ])
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "poison-home"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "poison-path"))

    info = probe_sqlite_runtime(fake_python)

    assert info is not None
    assert info.executable == fake_python
    assert info.base_prefix == tmp_path / "reported-base"
    assert info.sqlite_version == (9, 8, 7)
    assert info.sqlite_version_string == "9.8.7-child"
    assert info.sqlite_source_id == "child-source-id"
