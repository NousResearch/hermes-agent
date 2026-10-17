"""Regression: a state.db write-lock timeout names the process holding the lock.

Before, ``database is locked (another Hermes process held the state.db write lock for over 60s)``
identified the victim only; every Hermes process has the DB open, so the descriptor scan could not
single out the writer. ``/proc/locks`` can.
"""

import io
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


@pytest.mark.platforms("linux")
def test_holder_skipped_by_one_proc_locks_pass_is_still_named(tmp_path, monkeypatch):
    """/proc/locks is served over several read()s, so churn elsewhere can skip an entry in one pass."""
    import hermes_state_lockowners

    db = tmp_path / "state.db"
    db.write_bytes(b"")
    st = os.stat(db)
    held = (f"1: POSIX  ADVISORY  WRITE {os.getpid()} "
            f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} 1073741825 1073741825\n")
    passes = iter(["", held, ""])
    monkeypatch.setattr(hermes_state_lockowners, "open", lambda *a, **k: io.StringIO(next(passes)), raising=False)
    lines = state_db_write_lock_holders(db)
    assert len(lines) == 1 and f"PID {os.getpid()} " in lines[0] and "RESERVED" in lines[0], lines


@pytest.mark.platforms("linux")
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


@pytest.mark.platforms("linux")
def test_translated_container_device_still_identifies_verified_wal_writer(tmp_path, monkeypatch):
    """The MAJ:MIN in /proc/locks can differ from stat() inside an Incus container."""
    import hermes_state_lockowners as lockowners

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(x)")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        shm_st = os.stat(str(db) + "-shm")
        # Report a different kernel device but the *same* inode and holder PID.
        other_minor = os.minor(shm_st.st_dev) + 1
        fake_lock = (f"1: POSIX  ADVISORY  WRITE {os.getpid()} "
                     f"{os.major(shm_st.st_dev):02x}:{other_minor:02x}:{shm_st.st_ino} 120 120\n")
        real_open = open

        def translated_open(path, *args, **kwargs):
            path = os.fspath(path)
            if path == "/proc/locks":
                return io.StringIO(fake_lock)
            if path.startswith(f"/proc/{os.getpid()}/fdinfo/"):
                return io.StringIO("lock: " + fake_lock)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(lockowners, "open", translated_open, raising=False)

        lines = state_db_write_lock_holders(db)
        assert len(lines) == 1, lines
        assert f"PID {os.getpid()} " in lines[0] and "WAL write" in lines[0], lines
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.platforms("linux")
def test_translated_device_inode_without_verified_owner_is_not_attributed(tmp_path, monkeypatch):
    """A matching inode by itself is insufficient; the PID must hold the target fd."""
    import hermes_state_lockowners as lockowners

    db = tmp_path / "state.db"
    db.write_bytes(b"")
    st = os.stat(db)
    other_minor = os.minor(st.st_dev) + 1
    fake_lock = (f"1: POSIX  ADVISORY  WRITE {os.getpid()} "
                 f"{os.major(st.st_dev):02x}:{other_minor:02x}:{st.st_ino} 1073741825 1073741825\n")
    monkeypatch.setattr(lockowners, "open", lambda *a, **k: io.StringIO(fake_lock), raising=False)
    assert state_db_write_lock_holders(db) == []


@pytest.mark.platforms("linux")
def test_target_open_but_kernel_lock_from_other_device_is_not_attributed(tmp_path, monkeypatch):
    """Even a PID holding the target open must have THAT lock in its fdinfo."""
    import hermes_state_lockowners as lockowners

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(x)")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        shm_st = os.stat(str(db) + "-shm")
        other_minor = os.minor(shm_st.st_dev) + 1
        fake_lock = (f"1: POSIX  ADVISORY  WRITE {os.getpid()} "
                     f"{os.major(shm_st.st_dev):02x}:{other_minor:02x}:{shm_st.st_ino} 120 120\n")
        real_open = open

        def kernel_only_open(path, *args, **kwargs):
            if os.fspath(path) == "/proc/locks":
                return io.StringIO(fake_lock)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(lockowners, "open", kernel_only_open, raising=False)
        # The process has the target shm open, but its *actual* fdinfo reports
        # a different lock identity. An inode + open FD must not suffice.
        assert state_db_write_lock_holders(db) == []
    finally:
        conn.rollback()
        conn.close()
