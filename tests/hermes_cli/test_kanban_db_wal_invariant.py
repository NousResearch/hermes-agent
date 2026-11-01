"""Post-commit file-length invariant (torn-extend detector) under WAL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.sqlite_safe_read import file_length_matches_header


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_write_txn_tolerates_wal_lagging_the_main_file(kanban_home):
    """A committed page still sitting in the -wal is not a torn extend.

    ``_check_file_length_invariant`` runs *after* COMMIT, so a false positive
    here raises ``sqlite3.DatabaseError`` for a transaction that is already
    durable -- and anything retrying on ``DatabaseError`` writes it twice.
    Under WAL the main file lags its page count by design until a checkpoint,
    which is the normal state of a busy board, so the invariant must not fire.

    Reported from two platforms on #53819 (ext4/NVMe and APFS/NVMe), always
    off by exactly one page with ``integrity_check`` clean. The WAL exemption
    in ``_check_file_length_invariant`` is what prevents it; this locks it in.
    """
    conn = kbc.connect()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        if mode != "wal":
            pytest.skip(
                f"runtime keeps journal_mode={mode} (WAL is withheld on SQLite "
                f"{sqlite3.sqlite_version}); this false positive is WAL-only"
            )
        # Stop the -wal draining, so the main file stays behind its page count.
        conn.execute("PRAGMA wal_autocheckpoint=0")

        with kb.write_txn(conn) as c:
            c.executemany(
                "INSERT INTO task_events (task_id, kind, payload, created_at)"
                " VALUES (?, ?, ?, ?)",
                [("t_lag", "heartbeat", "x" * 2000, 0) for _ in range(400)],
            )

        # Guard the guard: if the write stops producing the lag, this test
        # would keep passing while covering nothing.
        assert file_length_matches_header(conn) is False, (
            "expected the main file to lag its page count under WAL -- "
            "without that lag this test no longer covers the false positive"
        )
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 400
    finally:
        conn.close()
