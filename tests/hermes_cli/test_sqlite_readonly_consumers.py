"""Exercise the read-only consumers with real databases and special paths."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import os
import sqlite3

import pytest

from gateway.readiness import _probe_state_db
from hermes_cli.approvals_suggest import _connect_readonly
from hermes_cli.backup import _safe_copy_db, verify_sqlite_integrity
from hermes_cli.kanban_db_notify import count_notify_subs
from hermes_cli.sqlite_safe_read import (
    connect_tracked,
    has_live_connection,
    read_header_bytes_preopen,
)


def _assert_consumers(home: Path, monkeypatch) -> None:
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "state.db"
    with closing(connect_tracked(path, isolation_level=None)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE messages(content TEXT)")
        db.execute("INSERT INTO messages VALUES ('retained content')")
        assert _probe_state_db(home)["status"] == "ok"
        with closing(_connect_readonly(path)) as reader:
            assert reader.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
            assert has_live_connection(path)
        snapshot = home / "copy#literal%23.db"
        assert _safe_copy_db(path, snapshot)
        assert verify_sqlite_integrity(snapshot)["valid"]
        # Oversized branch: use a small limit to execute the structural probe.
        assert verify_sqlite_integrity(snapshot, max_bytes=1)["valid"]
        with closing(sqlite3.connect(snapshot)) as reader:
            assert reader.execute("SELECT content FROM messages").fetchall() == [("retained content",)]
    assert not has_live_connection(path)
    assert not has_live_connection(snapshot)
    assert not (home.parent / "profile").exists()


@pytest.mark.parametrize("dirname", ["profile#literal", "profile%23literal", "profile space-é"])
def test_session_readiness_approvals_and_backup_use_literal_paths(tmp_path, monkeypatch, dirname):
    _assert_consumers(tmp_path / dirname, monkeypatch)


@pytest.mark.platforms("windows")
def test_native_windows_consumers(tmp_path, monkeypatch):
    _assert_consumers(tmp_path / "profile#literal%23 space-é", monkeypatch)


@pytest.mark.platforms("macos")
def test_native_macos_consumers(tmp_path, monkeypatch):
    _assert_consumers(tmp_path / "profile#literal%23 space-é", monkeypatch)


def test_backup_reader_is_tracked_until_closed(tmp_path, monkeypatch):
    path = tmp_path / "live.db"
    destination = tmp_path / "snapshot.db"
    with closing(connect_tracked(path, isolation_level=None)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample(value TEXT)")
        writer.execute("INSERT INTO sample VALUES ('in WAL')")
        from hermes_cli import backup
        from hermes_cli import sqlite_safe_read as safe_read
        from hermes_cli.sqlite_safe_read import open_sqlite_readonly

        observations = []

        class ObservedConnection(sqlite3.Connection):
            def backup(self, target, **kwargs):
                with safe_read._live_lock:
                    observations.append(safe_read._live_connections[str(path.resolve())])
                assert read_header_bytes_preopen(path, length=16) is None
                return super().backup(target, **kwargs)

        def observed_reader(source, **kwargs):
            return open_sqlite_readonly(source, factory=ObservedConnection, **kwargs)

        monkeypatch.setattr("hermes_cli.sqlite_safe_read.open_sqlite_readonly", observed_reader)
        assert backup._safe_copy_db(path, destination)
        assert observations == [2]
        with safe_read._live_lock:
            assert safe_read._live_connections[str(path.resolve())] == 1
    assert not has_live_connection(path)
    with closing(sqlite3.connect(destination)) as reader:
        assert reader.execute("SELECT value FROM sample").fetchall() == [("in WAL",)]


def test_notify_subscription_count_reader_is_tracked_and_closed(tmp_path, monkeypatch):
    path = tmp_path / "board#literal.db"
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("CREATE TABLE kanban_notify_subs(task_id TEXT)")
        writer.execute("INSERT INTO kanban_notify_subs VALUES ('task-1')")
        writer.commit()
    from hermes_cli import sqlite_safe_read as safe_read

    real_open = safe_read.open_sqlite_readonly
    observed = []

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith("SELECT COUNT(*) FROM kanban_notify_subs"):
                observed.append(safe_read.has_live_connection(path))
                assert safe_read.read_header_bytes_preopen(path, length=16) is None
            return super().execute(sql, *args)

    def observed_reader(database, **kwargs):
        return real_open(database, factory=ObservedConnection, **kwargs)

    monkeypatch.setattr(safe_read, "open_sqlite_readonly", observed_reader)
    assert count_notify_subs(db_path=path) == 1
    assert observed == [True]
    assert not has_live_connection(path)


def test_missing_and_corrupt_consumers_close_and_fail_closed(tmp_path):
    from hermes_cli.approvals_suggest import scan_approval_history

    path = tmp_path / "state.db"
    destination = tmp_path / "snapshot.db"
    assert _probe_state_db(tmp_path) == {"status": "ok", "detail": "not initialized"}
    assert scan_approval_history(path) == []
    assert count_notify_subs(db_path=path) == 0
    assert not _safe_copy_db(path, destination)
    assert not verify_sqlite_integrity(path)["valid"]
    assert not path.exists()
    assert not destination.exists()

    path.write_bytes(b"not a database" * 100)
    assert _probe_state_db(tmp_path)["status"] == "degraded"
    with pytest.raises(sqlite3.DatabaseError):
        scan_approval_history(path)
    with pytest.raises(sqlite3.DatabaseError):
        count_notify_subs(db_path=path)
    assert not _safe_copy_db(path, destination)
    assert not verify_sqlite_integrity(path)["valid"]
    assert not has_live_connection(path)
    assert not destination.exists()


def test_busy_writer_backup_deadline_and_reader_cleanup(tmp_path):
    path = tmp_path / "state.db"
    destination = tmp_path / "snapshot.db"
    with closing(connect_tracked(path, isolation_level=None)) as writer:
        writer.execute("CREATE TABLE sample(value TEXT)")
        writer.execute("BEGIN EXCLUSIVE")
        try:
            assert _probe_state_db(tmp_path)["status"] == "degraded"
            assert not _safe_copy_db(path, destination, timeout_seconds=0)
            assert not destination.exists()
            from hermes_cli import sqlite_safe_read as safe_read
            with safe_read._live_lock:
                assert safe_read._live_connections[str(path.resolve())] == 1
        finally:
            writer.rollback()
    assert not has_live_connection(path)


def test_approvals_scan_uses_one_wal_snapshot(tmp_path, monkeypatch):
    import json
    import hermes_cli.approvals_suggest as suggest

    # Windows forbids '?' in filesystem names; retain legal URI metacharacters.
    filename = "approval#%23 .db" if os.name == "nt" else "approval#%23 ?.db"
    path = tmp_path / filename
    with closing(connect_tracked(path, isolation_level=None)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE messages(role TEXT, tool_calls TEXT, timestamp REAL, tool_call_id TEXT, content TEXT)")
        real_blocked = suggest._blocked_tool_call_ids

        def concurrent_denial(reader, since):
            blocked = real_blocked(reader, since)
            calls = json.dumps([{"id": "denied", "function": {"name": "terminal", "arguments": json.dumps({"command": "git push --force origin main"})}}])
            writer.execute("INSERT INTO messages VALUES ('assistant', ?, 1, NULL, NULL)", (calls,))
            writer.execute("INSERT INTO messages VALUES ('tool', NULL, 1, 'denied', 'BLOCKED: User denied command')")
            return blocked

        monkeypatch.setattr(suggest, "_blocked_tool_call_ids", concurrent_denial)
        assert suggest.scan_approval_history(path, days=0) == []
        from hermes_cli import sqlite_safe_read as safe_read
        with safe_read._live_lock:
            assert safe_read._live_connections[str(path.resolve())] == 1
    assert not has_live_connection(path)


def test_readiness_reader_is_tracked_and_closed(tmp_path, monkeypatch):
    from hermes_cli import sqlite_safe_read as safe_read

    path = tmp_path / "state.db"
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("CREATE TABLE sample(value TEXT)")
    real_open = safe_read.open_sqlite_readonly
    observed = []

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith("SELECT name FROM sqlite_master"):
                observed.append(has_live_connection(path))
                assert read_header_bytes_preopen(path, length=16) is None
            return super().execute(sql, *args)

    def observed_reader(database, **kwargs):
        assert kwargs["timeout"] == 1.0
        return real_open(database, factory=ObservedConnection, **kwargs)

    monkeypatch.setattr(safe_read, "open_sqlite_readonly", observed_reader)
    assert _probe_state_db(tmp_path)["status"] == "ok"
    assert observed == [True]
    assert not has_live_connection(path)


def test_shared_state_uri_builder_delegates(tmp_path, monkeypatch):
    from hermes_cli import sqlite_safe_read as safe_read
    from hermes_state_holders import read_only_db_uri

    path = tmp_path / "literal#%23.db"
    calls = []

    def builder(database):
        calls.append(database)
        return "file:delegated?mode=ro"

    monkeypatch.setattr(safe_read, "read_only_db_uri", builder)
    assert read_only_db_uri(path) == "file:delegated?mode=ro"
    assert calls == [path]


def test_legacy_kanban_without_subscriptions_closes_reader(tmp_path):
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("CREATE TABLE old_board(value TEXT)")
    assert count_notify_subs(db_path=path) == 0
    assert not has_live_connection(path)
