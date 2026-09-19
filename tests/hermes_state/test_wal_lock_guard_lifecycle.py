"""The WAL lock guard tracks the lifecycle of the handle it protects.

Three shapes the first cut got wrong (review on #110544): a writer reopened after ``close()``
raced a live caller came back unguarded; the periodic checkpoint's guard refresh could land after
``close()`` and pin an OFD lock with no connection behind it; and refcounts keyed on a reusable
descriptor NUMBER mistook a recycled fd for a surviving lock. Linux-only: OFD locks + ``/proc``.
"""

import os
import sqlite3
import subprocess
import sys

import pytest

import hermes_state_lockguard as lg
from hermes_state import SessionDB
from tests.hermes_state._wal_generation_harness import make_db, pin_wal, require_wal

pytestmark = pytest.mark.linux_only


def _foreign_exclusive_ok(path: str) -> bool:
    """Another process tries the EXCLUSIVE a close-time WAL reset needs; True = nothing guards."""
    code = (
        "import fcntl, os, struct, sys\n"
        f"fd = os.open({path!r}, os.O_RDWR)\n"
        "lk = struct.pack('@hhqqi', fcntl.F_WRLCK, 0, 0x40000002, 510, 0)\n"
        "try:\n    fcntl.fcntl(fd, 37, lk); print('EXCLUSIVE_ACQUIRED')\n"
        "except BlockingIOError:\n    print('REFUSED')\n"
    )
    return "EXCLUSIVE_ACQUIRED" in subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def test_reopened_writer_is_guarded_again(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    require_wal(db)
    db.close()
    db.append_message("s", role="user", content="after close")  # #94736 teardown/worker reopen
    try:
        assert db._conn is not None and db._wal_lock_guard, "reopen returned an unguarded writer"
        assert not _foreign_exclusive_ok(str(db.db_path))
    finally:
        db.close()
    assert _foreign_exclusive_ok(str(db.db_path))  # a true last close lifts the guard


def test_guard_never_outlives_the_handle_under_fd_reuse(tmp_path, monkeypatch):
    """A+B live -> close A (its fd number is recycled by C) -> close B: C must still be guarded,
    and once C closes nothing may be left locked."""
    pin_wal(monkeypatch)
    path = tmp_path / "state.db"
    a = make_db(path, "s", "seed")
    require_wal(a)
    b = SessionDB(db_path=path)
    a.close()
    c = SessionDB(db_path=path)
    b.close()
    try:
        assert not _foreign_exclusive_ok(str(path)), "C recorded as guarded while nothing locks"
        for name in ("state.db", "state.db-shm"):  # the stray close the guard exists for
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        subprocess.run([sys.executable, "-c",
                        f"import sqlite3; c = sqlite3.connect({str(path)!r}); "
                        "c.execute('select count(*) from messages').fetchone(); c.close()"], check=True)
        c.append_message("s", role="user", content="still writes")
    finally:
        c.close()
    assert _foreign_exclusive_ok(str(path)), "a lock survived the last handle's close"
    assert not lg._HANDLES
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
