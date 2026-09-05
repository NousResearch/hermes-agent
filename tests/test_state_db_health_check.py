"""Health checks must reject real index damage without repairing the source (#63386)."""

import os
import sqlite3
import struct
import subprocess
import sys
from contextlib import closing
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_cli.console_engine import HermesConsoleEngine
from hermes_cli.sessions_cmd import cmd_sessions
from hermes_state import SessionDB
from hermes_state_repair import _db_opens_cleanly, repair_state_db_schema


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", path)
    with SessionDB(db_path=path) as db:
        db.create_session("health-check", source="cli")
        for i in range(10):
            db.append_message("health-check", role="user", content=f"pizza recipe {i}")
    return path


def _canonical_rows(path):
    with closing(sqlite3.connect(path)) as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("sessions", "messages")
        }


def _damage_segments(path, table):
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        # Preserve FTS structure/config records (1 and 10), so empty MATCH
        # still succeeds even though the index's own integrity check fails.
        changed = conn.execute(
            f"UPDATE {table}_data SET block=zeroblob(length(block)) WHERE id > 10"
        ).rowcount
        assert changed > 0
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        assert conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH '\"\"'"
        ).fetchall() == []
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(f"INSERT INTO {table}({table}) VALUES('integrity-check')")


def _damage_delivery_btree(path):
    from gateway.delivery_ledger import _initialize_schema

    with closing(sqlite3.connect(path)) as conn:
        _initialize_schema(conn)
        conn.executemany(
            "INSERT INTO delivery_obligations "
            "(obligation_id, session_key, platform, chat_id, content, state, created_at, updated_at) "
            "VALUES (?, 'health-check', 'test', 'chat', 'reply', 'pending', 1, 1)",
            [(f"obligation-{i}",) for i in range(3)],
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        rootpage = conn.execute(
            "SELECT rootpage FROM sqlite_schema WHERE name='delivery_obligations'"
        ).fetchone()[0]
    # Only this closed, disposable DB is byte-edited. Swap cell pointers in
    # a real table leaf: the records survive, but rowid ordering is invalid.
    data = bytearray(path.read_bytes())
    offset = (rootpage - 1) * page_size
    assert data[offset] == 0x0D
    assert struct.unpack_from(">H", data, offset + 3)[0] >= 2
    first, second = data[offset + 8:offset + 10], data[offset + 10:offset + 12]
    data[offset + 8:offset + 12] = second + first
    path.write_bytes(data)
    with closing(sqlite3.connect(path)) as conn:
        problems = " ".join(row[0] for row in conn.execute("PRAGMA quick_check"))
        assert "Rowid" in problems and "out of order" in problems
    assert _db_opens_cleanly(path) is not None


@pytest.mark.parametrize("table", ["messages_fts", "messages_fts_trigram"])
def test_segment_corruption_is_detected_and_rebuilt_without_losing_rows(state_path, table):
    before = _canonical_rows(state_path)
    assert _db_opens_cleanly(state_path) is None
    _damage_segments(state_path, table)

    reason = _db_opens_cleanly(state_path)
    assert reason is not None, "empty MATCH/write probes must not certify corrupt FTS segments"
    assert table in reason
    assert _canonical_rows(state_path) == before

    report = repair_state_db_schema(state_path)
    assert report["repaired"], report
    assert report["strategy"] == "rebuild_fts"
    assert report["backup_path"]
    assert _db_opens_cleanly(state_path) is None
    assert _canonical_rows(state_path) == before
    with closing(sqlite3.connect(state_path)) as conn:
        hits = conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH 'pizza' ORDER BY rowid"
        ).fetchall()
        expected = conn.execute("SELECT id FROM messages ORDER BY id").fetchall()
        assert hits == expected


@pytest.mark.parametrize("surface", ["cli", "console"])
@pytest.mark.parametrize("damage", ["fts", "btree"])
def test_check_only_reports_failure_without_repairing(state_path, damage, surface, capsys):
    def run_check():
        if surface == "console":
            result = HermesConsoleEngine().execute("sessions repair --check-only", confirmed=True)
            return result.status == "error", result.output
        rc = cmd_sessions(SimpleNamespace(sessions_action="repair", check_only=True))
        return bool(rc), capsys.readouterr().out

    assert run_check()[0] is False
    before_rows = _canonical_rows(state_path)
    if damage == "fts":
        _damage_segments(state_path, "messages_fts")
    else:
        _damage_delivery_btree(state_path)
    before_bytes = state_path.read_bytes()

    failed, output = run_check()
    assert failed, output
    assert "does not open cleanly" in output
    assert _canonical_rows(state_path) == before_rows
    assert state_path.read_bytes() == before_bytes
    assert not list(state_path.parent.glob("*.malformed-backup-*"))
    assert not list(state_path.parent.glob("*repair-scratch*"))


def test_repair_connection_blocks_raw_reads_until_closed(state_path):
    from hermes_cli.sqlite_safe_read import read_header_bytes_preopen
    from hermes_state_repair import _repair_conn

    with _repair_conn(state_path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("BEGIN IMMEDIATE")
        # A raw descriptor close here would release this process's POSIX
        # locks; repair/probe connections must participate in the registry.
        header = read_header_bytes_preopen(state_path)
        script = (
            "import sqlite3,sys; c=sqlite3.connect(sys.argv[1],timeout=0); "
            "c.execute('BEGIN IMMEDIATE')"
        )
        peer = subprocess.run(
            [sys.executable, "-c", script, str(state_path)],
            capture_output=True, text=True, timeout=10, env=os.environ.copy(),
        )
        assert peer.returncode != 0 and "locked" in peer.stderr
        assert header is None
        conn.rollback()
    header = read_header_bytes_preopen(state_path)
    assert header is not None and header.startswith(b"SQLite format 3")
