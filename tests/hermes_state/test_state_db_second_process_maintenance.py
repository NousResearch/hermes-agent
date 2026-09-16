"""#103339 item 2: structural maintenance from a second PROCESS must refuse while ANY other process holds
state.db, even where SQLite's own lock probe is blind (``journal_mode=DELETE`` reader; malformed file)."""

from __future__ import annotations

import select
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_repair import repair_state_db_schema

_HOLDER = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("SELECT count(*) FROM sessions").fetchall()
print("ready", flush=True)
sys.stdin.read(1)
conn.close()
"""


@pytest.fixture
def delete_mode_db(tmp_path, monkeypatch) -> Path:
    import hermes_state_wal
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "delete")
    db = tmp_path / "state.db"
    handle = SessionDB(db_path=db)
    sid = handle.create_session(session_id=str(uuid.uuid4()), source="cli")
    handle.append_message(sid, role="user", content="seed")
    handle.close()
    assert db.read_bytes()[18] == 1, "fixture must be a rollback-journal DB"
    return db


@pytest.mark.linux_only
def test_repair_refuses_delete_mode_db_held_open_by_another_process(delete_mode_db):
    """A held DELETE-mode reader takes only SHARED, so ``BEGIN IMMEDIATE`` succeeds and the lock probe sees
    nothing; the holder scan must still refuse — REINDEX/VACUUM from a second process is the #103339 class."""
    db = delete_mode_db
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(db)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([holder.stdout], [], [], 10)[0] and holder.stdout.readline().strip() == "ready"
        with open(db, "r+b") as fh:  # schema page garbage under the holder: the shape repair is invoked on
            fh.seek(100)
            fh.write(b"\xff" * (4096 - 100))
        report = repair_state_db_schema(db, backup=False)
    finally:
        holder.stdin.write("x")
        holder.stdin.close()
        holder.wait(timeout=10)
    assert report["repaired"] is False
    assert "stop the gateway" in (report["error"] or "").lower()


def test_holder_scan_sees_through_a_symlinked_home(tmp_path):
    """psutil/libproc report the resolved pathname; a holder opened via the real path must be found
    when the scan is asked about the alias, or maintenance proceeds under a live writer. The Linux
    /proc leg already compares inodes; the textual psutil leg (macOS) is the one this pins."""
    import sqlite3

    from hermes_state_holders import foreign_state_db_holders

    real = tmp_path / "real-home"
    real.mkdir()
    alias = tmp_path / "alias-home"
    alias.symlink_to(real, target_is_directory=True)
    db = real / "state.db"
    sqlite3.connect(db).execute("CREATE TABLE t(x)").connection.close()
    holder = subprocess.Popen(
        [sys.executable, "-c",
         f"import sqlite3, sys, time; c = sqlite3.connect({str(db)!r}); c.execute('BEGIN IMMEDIATE'); "
         "print('held', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert any(pid == holder.pid for pid, _ in foreign_state_db_holders(alias / "state.db"))
    finally:
        holder.kill()
        holder.wait()
