"""Live Windows proof for #120205: structural maintenance sees a real foreign SQLite holder."""

import sqlite3
import subprocess
import sys

import pytest

from hermes_state_holders import foreign_state_db_holders, held_store_refusal

pytestmark = pytest.mark.platforms("windows")


def test_restart_manager_finds_real_foreign_state_db_holder(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os,sqlite3,sys; "
                "c=sqlite3.connect(sys.argv[1]); "
                "c.execute('SELECT count(*) FROM t').fetchone(); "
                "print(f'held:{os.getpid()}', flush=True); sys.stdin.readline()"
            ),
            str(db),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        marker, pid_text = holder.stdout.readline().strip().split(":", 1)
        assert marker == "held"
        sqlite_pid = int(pid_text)

        holders = foreign_state_db_holders(db)
        assert any(pid == sqlite_pid for pid, _ in holders), holders
        refusal = held_store_refusal(db, command="optimize-storage")
        assert refusal is not None and f"PID {sqlite_pid}" in refusal
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=30)

    assert foreign_state_db_holders(db) == []

@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_restart_manager_finds_foreign_sqlite_sidecar_holder(tmp_path, suffix):
    """A foreign sidecar-only handle is still a holder of this SQLite resource family."""
    db = tmp_path / "state.db"
    db.touch()
    sidecar = tmp_path / f"state.db{suffix}"
    sidecar.write_bytes(b"sidecar sentinel")

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os,sys; "
                "f=open(sys.argv[1], 'rb'); "
                "print(f'held:{os.getpid()}', flush=True); sys.stdin.readline()"
            ),
            str(sidecar),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        marker, pid_text = holder.stdout.readline().strip().split(":", 1)
        assert marker == "held"
        holder_pid = int(pid_text)

        holders = foreign_state_db_holders(db)
        assert any(pid == holder_pid for pid, _ in holders), holders
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=30)

    assert foreign_state_db_holders(db) == []
