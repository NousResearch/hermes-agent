"""Invariant: importing a backup must not publish a NEW state.db inode while a
live process still holds the deleted one (#110179).

``_import_db_member`` guards the replace path with ``_safe_restore_db``, but the
"target is missing" branch treated absence as proof of no holders. A gateway that
had the database open when it was unlinked keeps writing the deleted inode, so an
atomic publish there re-creates exactly the #90950 split brain the branch below
avoids. The holder here is a real subprocess with a real SQLite connection.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import zipfile

import pytest

from hermes_cli import backup_restore as backup_mod

pytestmark = pytest.mark.platforms("linux")  # /proc holder scan is Linux-only

_HOLDER = (
    "import sqlite3, sys, time\n"
    "conn = sqlite3.connect(sys.argv[1])\n"
    "conn.execute('SELECT count(*) FROM t').fetchone()\n"
    "sys.stdout.write('held\\n')\n"
    "sys.stdout.flush()\n"
    "time.sleep(300)\n"
)

_MEMBER = "hermes/state.db"


def _make_db(path, marker):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.execute("INSERT INTO t VALUES (?)", (marker,))
    conn.commit()
    conn.close()


def test_import_refuses_to_publish_over_a_deleted_but_held_database(tmp_path):
    target = tmp_path / "state.db"
    _make_db(target, "live")
    archive = tmp_path / "backup.zip"
    payload = tmp_path / "archived.db"
    _make_db(payload, "from-archive")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(payload, _MEMBER)

    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(target)],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    stdout = holder.stdout
    assert stdout is not None
    try:
        assert stdout.readline().strip() == "held"
        target.unlink()  # the deleted-inode window: holder keeps writing it
        assert backup_mod._foreign_db_holder_pids(target) == [holder.pid]

        with zipfile.ZipFile(archive) as zf:
            with pytest.raises(OSError) as excinfo:
                backup_mod._import_db_member(zf, _MEMBER, target)

        assert str(holder.pid) in str(excinfo.value)
        assert not target.exists(), "a new inode was published under a live holder"
    finally:
        holder.terminate()  # by PID: the process this test spawned
        holder.wait(timeout=10)
        stdout.close()

    # No holder left: the ordinary atomic publish still applies.
    with zipfile.ZipFile(archive) as zf:
        backup_mod._import_db_member(zf, _MEMBER, target)
    conn = sqlite3.connect(str(target))
    try:
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "from-archive"
    finally:
        conn.close()
