"""#86515: the lock-free mode=ro read pool needs a CONFIRMED WAL header. When the on-disk probe at open
is blocked by a concurrent opener, apply_wal_with_fallback reports "wal" as the assumed mode with nothing
touched; SessionDB must not turn that assumption into pooled readers on a file that is really DELETE."""
import sqlite3

from hermes_state import SessionDB


def test_blocked_probe_on_delete_file_does_not_enable_read_pool(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("s", "cli")
    db.close()
    offline = sqlite3.connect(str(path))  # sole opener: a DELETE switch here is safe
    assert offline.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() == "delete"
    offline.close()

    sibling = sqlite3.connect(str(path), isolation_level=None)
    sibling.execute("BEGIN EXCLUSIVE")  # blocks the header read: the probe comes back None
    try:
        conn = db._open_writer_conn()
    finally:
        sibling.execute("ROLLBACK")
        sibling.close()
    try:
        assert db._wal_active is False
        assert db._checkout_read_conn() is None  # reads queue on the writer lock, never SQLITE_BUSY
    finally:
        conn.close()


def test_confirmed_wal_file_keeps_read_pool(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        if sqlite3.connect(str(path)).execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
            return  # vulnerable-SQLite or WAL-refusing filesystem: nothing to confirm
        assert db._wal_active is True
    finally:
        db.close()
