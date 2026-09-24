"""`hermes sessions set-journal-mode` converts an existing WAL store offline and refuses under a foreign holder.

#100896 (@ruangraung): `database.journal_mode: delete` never self-applies to a store that is already WAL because
open never live-downgrades; this command is the sanctioned offline path and must fail closed while any other
process holds the file — or while it cannot prove the file is quiet at all.
"""
import argparse
import sqlite3
import subprocess
import sys

import pytest

from hermes_cli.sessions_cmd import cmd_sessions
from hermes_cli.sessions_cmd_journal_mode import _refusal


def _wal_store(path):
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")
    conn.close()
    assert path.read_bytes()[18:20] == b"\x02\x02"


def _args(mode, db=None, force=True):
    return argparse.Namespace(sessions_action="set-journal-mode", mode=mode, db=db, force=force)


def test_set_journal_mode_converts_wal_store_offline(tmp_path, monkeypatch, capsys):
    db = tmp_path / "state.db"
    _wal_store(db)
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", db)

    assert cmd_sessions(_args("delete")) == 0

    assert db.read_bytes()[18:20] == b"\x01\x01", "header must report rollback-journal mode"
    assert not (tmp_path / "state.db-wal").exists()
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 3
    conn.close()
    assert "wal → delete" in capsys.readouterr().out


@pytest.mark.parametrize("force", [False, True], ids=["normal", "force"])
def test_set_journal_mode_refuses_while_another_process_holds_the_store(
    force, tmp_path, monkeypatch, capsys
):
    db = tmp_path / "state.db"
    _wal_store(db)
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", db)
    holder = subprocess.Popen(
        [
            sys.executable, "-c",
            (
                "import os,sqlite3,sys,time; "
                "c=sqlite3.connect(sys.argv[1]); c.execute('SELECT 1'); "
                "print(f'held:{os.getpid()}', flush=True); time.sleep(60)"
            ),
            str(db),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        marker, pid_text = holder.stdout.readline().strip().split(":", 1)
        assert marker == "held"
        sqlite_pid = int(pid_text)
        # --force never waives a process the scan actually found.
        assert cmd_sessions(_args("delete", force=force)) == 1
    finally:
        holder.kill()
        holder.wait()

    out = capsys.readouterr().out
    assert f"pid {sqlite_pid}" in out
    assert db.read_bytes()[18:20] == b"\x02\x02", "a refused switch must leave the file untouched"

    # A failed scan (pid <= 0 sentinel) is refused too, and is the ONLY thing --force waives.
    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda path: [(-1, "scan failed")])
    assert cmd_sessions(_args("delete", force=force)) == (0 if force else 1)
    assert db.read_bytes()[18:20] == (b"\x01\x01" if force else b"\x02\x02")
    assert ("scan: scan failed" in capsys.readouterr().out) is not force


@pytest.mark.parametrize("target,current,cross_vm,expected", [
    ("delete", "wal", False, None),
    # WAL shared memory corrupts on virtiofs/9p, so ENABLING it there is refused.
    ("wal", "delete", True, "cross-VM filesystems"),
    ("delete", "wal", True, None),
    # A garbage or unrecognised header is reported, never handed to sqlite3 for a raw traceback.
    ("delete", "not-a-database", False, "not a Hermes SQLite store"),
    ("delete", "unknown(3/3)", False, "not a Hermes SQLite store"),
])
def test_refusal_admission_invariants(target, current, cross_vm, expected):
    reason = _refusal(target, current, on_cross_vm_fs=cross_vm)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


@pytest.mark.parametrize("kind", ["garbage-file", "directory"])
def test_set_journal_mode_reports_unusable_db_paths_without_a_traceback(kind, tmp_path, capsys):
    if kind == "garbage-file":
        bad = tmp_path / "notes.txt"
        bad.write_bytes(b"this is not a database, not even close\n")
    else:
        bad = tmp_path / "a-directory"
        bad.mkdir()

    assert cmd_sessions(_args("delete", db=str(bad))) == 1
    assert "✗" in capsys.readouterr().out
