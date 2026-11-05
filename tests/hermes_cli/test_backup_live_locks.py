"""Taking a snapshot must preserve another connection's live transaction."""
import sqlite3
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
def test_backup_cannot_release_a_live_writers_transaction(tmp_path, monkeypatch):
    from hermes_cli.console_engine import HermesConsoleEngine

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    writer = sqlite3.connect(path, isolation_level=None)
    try:
        writer.execute("CREATE TABLE work(value TEXT)")
        writer.execute("INSERT INTO work VALUES ('saved')")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO work VALUES ('in progress')")
        result = HermesConsoleEngine().execute("backup --quick", confirmed=True)
        assert result.status == "ok", result.output
        result = subprocess.run([
            sys.executable, "-c", """
import sqlite3, sys
db = sqlite3.connect(sys.argv[1], timeout=0)
try:
    db.execute("INSERT INTO work VALUES ('external committed')")
    db.commit()
    print('committed')
except sqlite3.OperationalError as exc:
    if exc.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        raise
    print('blocked')
finally:
    db.close()
""", str(path)], capture_output=True, text=True, check=True, timeout=30)
        assert result.stdout.strip() == "blocked"
        writer.commit()
        assert writer.execute("SELECT value FROM work").fetchall() == [("saved",), ("in progress",)]
    finally:
        writer.rollback()
        writer.close()


@pytest.mark.platforms("posix")
def test_quick_snapshot_keeps_live_session_database_usable(tmp_path, monkeypatch):
    from hermes_cli.backup import create_quick_snapshot
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("work", source="cli")
        db.append_message("work", "user", "saved work")
        assert create_quick_snapshot(hermes_home=tmp_path)
        subprocess.run([
            sys.executable, "-c", """
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('SELECT count(*) FROM messages').fetchone()
db.close()
""", str(path)], capture_output=True, text=True, check=True, timeout=30)
        db.append_message("work", "assistant", "work after snapshot")
        assert [m["content"] for m in db.get_messages("work")] == [
            "saved work", "work after snapshot"]
