"""Tests for the WAL-safe SQLite copy behind full and quick backups, and the zeroed-database probe."""

import sqlite3
from pathlib import Path


class TestSafeCopyDb:
    def test_copies_valid_database(self, tmp_path):
        from hermes_cli.backup import _safe_copy_db
        src = tmp_path / "test.db"
        dst = tmp_path / "copy.db"

        conn = sqlite3.connect(str(src))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
        conn.commit()
        conn.close()

        result = _safe_copy_db(src, dst)
        assert result is True

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT x FROM t").fetchall()
        conn.close()
        assert rows == [(42,)]

    def test_aborts_when_source_remains_busy_past_deadline(
        self, tmp_path, monkeypatch
    ):
        from hermes_cli import backup as backup_mod

        src = tmp_path / "locked.db"
        dst = tmp_path / "copy.db"
        src.touch()
        dst.write_bytes(b"partial")

        clock = iter((100.0, 100.5, 101.1))

        class BusySourceConnection(sqlite3.Connection):
            def backup(self, target, *, pages=-1, progress=None, name="main", sleep=0.250):
                assert progress is not None
                assert pages > 0
                assert sleep > 0
                progress(sqlite3.SQLITE_BUSY, 0, 1)
                progress(sqlite3.SQLITE_BUSY, 0, 1)

        destination_closed = []

        class DestinationConnection(sqlite3.Connection):
            def close(self):
                super().close()
                destination_closed.append(True)

        real_connect = sqlite3.connect
        factories = iter((BusySourceConnection, DestinationConnection))
        real_unlink = Path.unlink

        def assert_closed_before_unlink(path, *args, **kwargs):
            assert destination_closed
            return real_unlink(path, *args, **kwargs)

        connect_calls = []

        def fake_connect(*args, **kwargs):
            connect_calls.append((args, kwargs.copy()))
            kwargs["factory"] = next(factories)
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(backup_mod.sqlite3, "connect", fake_connect)
        monkeypatch.setattr(backup_mod.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(Path, "unlink", assert_closed_before_unlink)

        assert backup_mod._safe_copy_db(src, dst, timeout_seconds=1.0) is False
        assert connect_calls[0][1]["timeout"] == 0.0
        assert not dst.exists()

    def test_is_zeroed_sqlite_file_detects_nul_header(self, tmp_path):
        from hermes_cli.backup import is_zeroed_sqlite_file
        p = tmp_path / "state.db"
        p.write_bytes(bytes(4096))  # all NULs
        assert is_zeroed_sqlite_file(p) is True
