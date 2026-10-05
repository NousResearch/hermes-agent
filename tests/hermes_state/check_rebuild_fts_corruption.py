"""Standalone check for the #133375 rebuild_fts corruption-class fix.

Runs without pytest/conftest (which fight the live-install tree). Reproduces the exact
production failure: every FTS 'rebuild' command raises sqlite3.DatabaseError (SQLITE_CORRUPT).
Before the fix, that error escaped the per-index loop un-rolled-back. After the fix, the
call returns 0 (no progress), rolls back, and logs the offline-repair hint.
"""
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hermes_state import SessionDB  # noqa: E402


def corrupting_execute(real_execute):
    def execute(sql, *args, **kwargs):
        if "VALUES('rebuild')" in sql:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return real_execute(sql, *args, **kwargs)
    return execute


def main():
    logging.basicConfig(level=logging.ERROR)
    tmp = Path(tempfile.mkdtemp(prefix="fts-check-"))
    db = SessionDB(db_path=tmp / "state.db")
    assert db._fts_enabled, "FTS5 unavailable in this build"
    db.create_session("s1", source="test")
    db.append_message("s1", "user", "hello world")

    # Baseline: a healthy rebuild works.
    assert db.rebuild_fts() >= 1, "healthy rebuild should make progress"

    # Now simulate the corruption class (#133375).
    db._conn.execute = corrupting_execute(db._conn.execute)
    result = db.rebuild_fts()  # must NOT propagate the DatabaseError
    assert result == 0, f"expected 0 (no progress), got {result}"
    assert db._conn.in_transaction is False, "failed statement must be rolled back"
    db.close()
    print("PASS: rebuild_fts catches corruption-class error, rolls back, returns 0")


if __name__ == "__main__":
    main()
