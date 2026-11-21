"""SQLite safe-copy: read-only open failure retries read-write (WAL without -shm)."""

import sqlite3


def test_ro_open_failure_falls_back_to_rw_backup(tmp_path, monkeypatch):
    """A failed read-only open must retry via a read-write backup().

    Read-only clients cannot open a WAL database whose -shm file is
    absent or needs recovery ("unable to open database file"). The
    read-write retry still snapshots consistently — including rows
    that only live in the -wal, which dropping the database from the
    snapshot would lose.
    """
    from hermes_cli import backup_sqlite

    src = tmp_path / "wal.db"
    dst = tmp_path / "copy.db"

    conn = sqlite3.connect(str(src))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2)")
    conn.commit()
    # Keep the connection open so the rows live in the -wal file only.

    real_connect = sqlite3.connect

    def fail_ro_connect(database, *args, **kwargs):
        if isinstance(database, str) and database.startswith("file:"):
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(backup_sqlite.sqlite3, "connect", fail_ro_connect)

    assert backup_sqlite._safe_copy_db(src, dst) is True
    conn.close()

    copy = real_connect(str(dst))
    assert copy.execute("SELECT count(*) FROM t").fetchone() == (2,)
    copy.close()

def test_fails_closed_when_read_only_and_read_write_open_both_fail(tmp_path, monkeypatch):
    """No consistent snapshot is possible, so drop the database and say so.

    Copying only the main file loses WAL data, so _safe_copy_db must not
    paper over two failed opens with a raw file copy.
    """
    from hermes_cli import backup_sqlite

    src = tmp_path / "test.db"
    dst = tmp_path / "copy.db"
    src.write_bytes(b"SQLite format 3\x00" + bytes(512))

    def always_fail(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(backup_sqlite.sqlite3, "connect", always_fail)

    assert backup_sqlite._safe_copy_db(src, dst) is False
    assert not dst.exists()

def test_locked_source_is_not_retried_read_write(tmp_path, monkeypatch):
    """A source that is locked, not unopenable, gets exactly one open attempt.

    Retrying read-write there would spend the busy deadline a second time
    over the same lock without being any more able to take it.
    """
    from hermes_cli import backup_sqlite

    src = tmp_path / "locked.db"
    dst = tmp_path / "copy.db"
    src.touch()

    clock = iter((100.0, 100.5, 101.1))

    class BusySource:
        def backup(_destination, *, pages, progress, sleep):
            progress(sqlite3.SQLITE_BUSY, 0, 1)
            progress(sqlite3.SQLITE_BUSY, 0, 1)

        def close(self):
            pass

    class Destination:
        def close(self):
            pass

    opened = []

    def fake_connect(*args, **kwargs):
        opened.append((args, kwargs))
        return BusySource() if len(opened) == 1 else Destination()

    monkeypatch.setattr(backup_sqlite.sqlite3, "connect", fake_connect)
    monkeypatch.setattr(backup_sqlite.time, "monotonic", lambda: next(clock))

    assert backup_sqlite._safe_copy_db(src, dst, timeout_seconds=1.0) is False
    # Source open + destination open, and no read-write retry on top.
    assert len(opened) == 2
