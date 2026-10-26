"""A stopped or killed update pre-flight must not leave full-size staging copies behind.

Desktop stops a pre-flight that outlives its deadline; each abandoned
``state.db.pre-update-emergency-*.partial`` is a whole copy of state.db, and
reporters accumulated 11.7 GB of them (#124972, #128605).
"""
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.waiting import wait_until

SCRIPT = Path(__file__).resolve().parents[2] / "hermes_cli" / "backup_sqlite.py"


def _home_with_db(tmp_path: Path, rows: int, journal_mode: str = "WAL") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as db:
        db.execute(f"PRAGMA journal_mode={journal_mode}")
        db.execute("CREATE TABLE messages (body TEXT)")
        db.executemany("INSERT INTO messages VALUES (?)", [("x" * 4000,)] * rows)
    return home


def _staging(home: Path) -> list[str]:
    return sorted(p.name for p in home.glob("state.db.pre-update-emergency-*.partial*"))


@pytest.mark.platforms("posix")
def test_sigterm_mid_copy_removes_the_staging_copy(tmp_path):
    home = _home_with_db(tmp_path, 10, journal_mode="DELETE")
    # The barrier: an exclusive lock on a rollback-journal source keeps the copy
    # busy-waiting until the child is signalled, however fast the disk is.
    holder = sqlite3.connect(home / "state.db", isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        proc = subprocess.Popen([sys.executable, "-I", "-S", str(SCRIPT), str(home)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            wait_until(lambda: _staging(home) or proc.poll() is not None, timeout=30, what="the staging copy")
            assert proc.poll() is None and _staging(home), "the pre-flight must still be copying"
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    # 128 + SIGTERM is the handler's exit; a finished or failed copy exits 0 or 1.
    assert proc.returncode == 128 + signal.SIGTERM, stderr
    assert stdout == b""
    assert _staging(home) == []


def test_a_new_preflight_reclaims_staging_abandoned_by_a_killed_one(tmp_path):
    home = _home_with_db(tmp_path, 10)
    abandoned = home / "state.db.pre-update-emergency-killed.partial"
    abandoned.write_bytes(b"x" * 4096)
    hours_ago = time.time() - 3 * 3600
    os.utime(abandoned, (hours_ago, hours_ago))
    in_flight = home / "state.db.pre-update-emergency-running.partial"
    in_flight.write_bytes(b"x" * 4096)

    result = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), str(home)],
                            capture_output=True, text=True, timeout=60, check=False)

    assert result.returncode == 0, result.stderr
    assert _staging(home) == [in_flight.name]
