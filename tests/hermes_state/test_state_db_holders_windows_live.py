"""Live Windows proof for #120205: structural maintenance sees a real foreign SQLite holder."""

import sqlite3
import subprocess
import sys

import pytest

from hermes_state_holders import foreign_state_db_holders, held_store_refusal

pytestmark = pytest.mark.windows_only


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
                "import sqlite3,sys; "
                "c=sqlite3.connect(sys.argv[1]); "
                "c.execute('SELECT count(*) FROM t').fetchone(); "
                "print('held', flush=True); sys.stdin.readline()"
            ),
            str(db),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        holders = foreign_state_db_holders(db)
        assert any(pid == holder.pid for pid, _ in holders), holders
        refusal = held_store_refusal(db, command="optimize-storage")
        assert refusal is not None and f"PID {holder.pid}" in refusal
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=30)

    assert foreign_state_db_holders(db) == []
