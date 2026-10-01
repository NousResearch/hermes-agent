"""Regression: a state.db write-lock timeout names the process holding the lock.

Before, ``database is locked (another Hermes process held the state.db write lock for over 60s)``
identified the victim only; every Hermes process has the DB open, so the descriptor scan could not
single out the writer. ``/proc/locks`` can.
"""

import os
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from hermes_state_lockowners import parse_proc_locks, state_db_write_lock_holders


def test_parse_proc_locks_keeps_only_write_locks_on_our_inodes_and_decodes_the_wal_write_byte():
    inodes = {(os.makedev(0x103, 0x02), 4194737): "-shm", (os.makedev(0x103, 0x02), 4228330): ""}
    text = textwrap.dedent("""\
        1: POSIX  ADVISORY  WRITE 594094 103:02:4194737 120 120
        2: POSIX  ADVISORY  READ 594094 103:02:4194737 125 125
        3: POSIX  ADVISORY  READ 99493 103:02:4194737 128 128
        4: OFDLCK ADVISORY  WRITE -1 103:02:4228330 1073741825 1073741825
        5: POSIX  ADVISORY  WRITE 4242 103:02:99999 120 120
        6: -> POSIX  ADVISORY  WRITE 777 103:02:4194737 120 120
    """)
    assert parse_proc_locks(text, inodes) == [
        (594094, "WAL write", "-shm"),
        (-1, "RESERVED", ""),
    ]


@pytest.mark.linux_only
def test_live_writer_in_another_process_is_named_by_pid(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=wal")
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()

    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sqlite3, sys, time
            c = sqlite3.connect({str(db)!r}, isolation_level=None)
            c.execute("BEGIN IMMEDIATE")
            c.execute("INSERT INTO t VALUES (1)")
            print("held", flush=True)
            time.sleep(30)
        """)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        deadline = time.monotonic() + 5
        lines = []
        while time.monotonic() < deadline:
            lines = state_db_write_lock_holders(db)
            if lines:
                break
            time.sleep(0.05)
        assert any(f"PID {holder.pid} " in line and "WAL write" in line for line in lines), lines
    finally:
        holder.kill()
        holder.wait()
    assert state_db_write_lock_holders(db) == []
