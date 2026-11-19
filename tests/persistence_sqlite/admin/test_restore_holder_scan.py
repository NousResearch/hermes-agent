"""The backup admin's unlink+move restore fallback refuses when the holder scan cannot answer.

The fallback replaces the destination inode and deletes its sidecars, which splits the database
under any process that still holds it. A scan that could not finish (no psutil, an unreadable
/proc, a failed Windows Restart Manager call) is "unknown", never "no holders".
"""

import sqlite3

from persistence.sqlite.admin import _backup


def _make_db(path, marker):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (marker,))
    conn.close()


def test_fallback_restore_refuses_when_holder_scan_is_inconclusive(tmp_path, monkeypatch):
    src, dst = tmp_path / "snapshot.db", tmp_path / "state.db"
    _make_db(src, "snapshot")
    _make_db(dst, "live")
    with open(dst, "r+b") as fh:  # a damaged header makes the backup-API route fail into the fallback
        fh.write(b"\x00" * 100)
    before = dst.read_bytes()
    monkeypatch.setattr(_backup, "foreign_state_db_holders", lambda path: [(-1, "open-file scan unavailable")])

    assert _backup._restore_db_pages(src, dst) is False
    assert dst.read_bytes() == before
