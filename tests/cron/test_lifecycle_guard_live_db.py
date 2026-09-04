"""``_read_referenced_script`` must never raw-open a path with a live
SQLite connection in this process (#102589).

A bare ``os.open()``/``os.close()`` on such a path cancels every POSIX
advisory lock this process holds on that file, for every fd, even though
the caller never touched the tracked connection. That drops the gateway's
WAL protection out from under it the moment a terminal command merely
*mentions* ``state.db``'s path.
"""

from __future__ import annotations

import cron.lifecycle_guard as lifecycle_guard
from hermes_cli import sqlite_safe_read


def test_read_referenced_script_fails_closed_on_live_connection(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"not a real sqlite file, presence is enough")

    def _must_not_open(*_args, **_kwargs):
        raise AssertionError(
            "must not raw os.open() a path with a live SQLite connection"
        )

    monkeypatch.setattr(lifecycle_guard.os, "open", _must_not_open)

    sqlite_safe_read.track_connection(db)
    try:
        text, unsafe = lifecycle_guard._read_referenced_script(db)
    finally:
        sqlite_safe_read.untrack_connection(db)

    assert text is None
    assert unsafe is True


def test_read_referenced_script_reads_normally_without_live_connection(tmp_path):
    script = tmp_path / "s.sh"
    script.write_text("echo ok\n", encoding="utf-8")

    text, unsafe = lifecycle_guard._read_referenced_script(script)

    assert text == "echo ok\n"
    assert unsafe is False
