"""StoreHolder open and close on real files: owner-only file modes, POSIX locks kept across a second open, no tracked connection leaked by a failed open, the checkpoint a close requests, FTS search on a read-only open, and the read-only open's retry of a transient WAL I/O error."""

import os
import stat
from unittest import mock

import pytest

from persistence.sqlite import _holder
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import search_messages
from persistence.sqlite.conversation._store.sessions import create_session


# =========================================================================
# Connection lifecycle
# =========================================================================


class TestConnectionLifecycle:
    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_writable_state_db_is_owner_only_under_permissive_umask(self, tmp_path):
        """state.db and any live SQLite sidecars must not inherit 0644 modes."""
        db_path = tmp_path / "state.db"

        old_umask = os.umask(0o022)
        try:
            session_db = StoreHolder(db_path=db_path)
        finally:
            os.umask(old_umask)

        try:
            state_files = [
                path
                for path in (
                    db_path,
                    db_path.with_name(db_path.name + "-wal"),
                    db_path.with_name(db_path.name + "-shm"),
                )
                if path.exists()
            ]
            assert state_files
            assert all(
                stat.S_IMODE(path.stat().st_mode) == 0o600
                for path in state_files
            )
        finally:
            session_db.close()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_writable_state_db_tightens_existing_loose_mode(self, tmp_path):
        """Opening a legacy 0644 profile store repairs it in place."""
        db_path = tmp_path / "state.db"
        initial = StoreHolder(db_path=db_path)
        initial.close()
        os.chmod(db_path, 0o644)

        session_db = StoreHolder(db_path=db_path)
        try:
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        finally:
            session_db.close()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX fcntl locks")
    def test_writable_state_db_keeps_locks_across_second_open(self, tmp_path):
        """Opening a second StoreHolder in this process must not unlink live sidecars.

        POSIX locks are owned per (process, inode): closing any descriptor for
        state.db drops every lock this process holds on it, including the locks
        of the first StoreHolder's connection. A sibling process reading the
        database after that close takes the shared-memory DMS exclusively on its
        own close, checkpoints, and unlinks -wal/-shm while the first handle
        keeps using the deleted inodes.
        """
        import subprocess
        import sys

        from hermes_state_dbfile import iter_deleted_sqlite_sidecar_holders

        db_path = tmp_path / "state.db"
        first = StoreHolder(db_path=db_path)
        second = StoreHolder(db_path=db_path)
        try:
            assert not iter_deleted_sqlite_sidecar_holders(db_path)
            subprocess.run(
                [sys.executable, "-c",
                 "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
                 "c.execute('SELECT count(*) FROM sessions').fetchone(); c.close()",
                 str(db_path)],
                check=True, timeout=30,
            )
            assert not iter_deleted_sqlite_sidecar_holders(db_path), (
                "a second StoreHolder open or a sibling reader unlinked the live "
                "WAL/SHM inodes out from under this process"
            )
        finally:
            second.close()
            first.close()

    def test_failed_writable_open_does_not_leak_tracked_connection(
        self, tmp_path, monkeypatch
    ):
        """A failed schema init must close the connection opened before it."""
        from hermes_cli.sqlite_safe_read import has_live_connection

        db_path = tmp_path / "state.db"
        opened = []
        real_connect = _holder._connect_tracked_db

        def capture_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        monkeypatch.setattr(_holder, "_connect_tracked_db", capture_connect)
        monkeypatch.setattr(
            _holder,
            "_init_schema",
            mock.Mock(side_effect=RuntimeError("schema init failed")),
        )

        try:
            with pytest.raises(RuntimeError, match="schema init failed"):
                StoreHolder(db_path=db_path)
            assert has_live_connection(db_path) is False
        finally:
            for conn in opened:
                try:
                    conn.close()
                except Exception:
                    pass

    def test_failed_wal_read_open_does_not_leak_tracked_connection(
        self, tmp_path, monkeypatch
    ):
        """A post-open read setup failure must close its unregistered conn."""
        from hermes_cli import sqlite_safe_read

        db_path = tmp_path / "state.db"
        db = StoreHolder(db_path=db_path)
        opened = []
        real_connect = _holder._connect_tracked_db
        real_pragmas = _holder.apply_database_pragmas

        def capture_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        def fail_pragmas(*args, **kwargs):
            raise RuntimeError("read setup failed")

        monkeypatch.setattr(_holder, "_connect_tracked_db", capture_connect)
        monkeypatch.setattr(_holder, "apply_database_pragmas", fail_pragmas)
        before = dict(sqlite_safe_read._live_connections)
        db._wal_active = True

        try:
            with pytest.raises(RuntimeError, match="read setup failed"):
                db._get_read_conn()
            assert sqlite_safe_read._live_connections == before
        finally:
            monkeypatch.setattr(_holder, "apply_database_pragmas", real_pragmas)
            for conn in opened:
                try:
                    conn.close()
                except Exception:
                    pass
            db.close()

    def test_read_only_close_never_requests_wal_checkpoint(self, tmp_path):
        db_path = tmp_path / "state.db"
        writable = StoreHolder(db_path=db_path)
        create_session(writable, "s1", source="cli")
        writable.close()

        executed = []
        read_only = StoreHolder(db_path=db_path, read_only=True)
        read_only._conn.set_trace_callback(executed.append)
        read_only.close()

        assert not any("wal_checkpoint" in sql.lower() for sql in executed)

    def test_writable_close_uses_passive_checkpoint(self, tmp_path):
        db_path = tmp_path / "state.db"
        writable = StoreHolder(db_path=db_path)
        executed = []
        writable._conn.set_trace_callback(executed.append)

        writable.close()

        # close() must NOT TRUNCATE: transient per-cron-run connections firing
        # full WAL resets race the gateway's live writer and corrupt B-tree
        # pages (issue #45383). It uses PASSIVE instead.
        assert not any(
            "pragma wal_checkpoint(truncate)" == " ".join(sql.lower().split())
            for sql in executed
        )
        assert any(
            "pragma wal_checkpoint(passive)" == " ".join(sql.lower().split())
            for sql in executed
        )

    def test_read_only_connection_keeps_fts_search_available(self, tmp_path):
        db_path = tmp_path / "state.db"
        writable = StoreHolder(db_path=db_path)
        create_session(writable, "fts-read-only", source="cli")
        append_message(writable,
            "fts-read-only",
            role="user",
            content="readonlywoodpecker 大别山项目",
        )
        writable.close()

        read_only = StoreHolder(db_path=db_path, read_only=True)
        try:
            base_matches = search_messages(read_only, "readonlywoodpecker")
            trigram_matches = search_messages(read_only, "大别山")
        finally:
            read_only.close()

        assert [match["session_id"] for match in base_matches] == [
            "fts-read-only"
        ]
        assert [match["session_id"] for match in trigram_matches] == [
            "fts-read-only"
        ]

    def test_failed_read_only_open_does_not_leak_tracked_connection(
        self, tmp_path
    ):
        """A malformed store makes the RO FTS probe raise DatabaseError.
        The connection must be closed on that failure path: a leaked tracked
        connection blocks _backup_db_file's raw-copy for the process
        lifetime, so the writable heal that follows would repair WITHOUT its
        forensic backup."""
        import sqlite3

        from hermes_cli.sqlite_safe_read import has_live_connection

        db_path = tmp_path / "state.db"
        writable = StoreHolder(db_path=db_path)
        create_session(writable, "s1", source="cli")
        append_message(writable, "s1", role="user", content="leak probe")
        writable.close()

        # Corrupt sqlite_master: duplicate messages_fts definition. Any
        # statement on a fresh connection then raises "malformed database
        # schema" (DatabaseError, not the OperationalError the probe eats).
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA writable_schema=ON")
        row = conn.execute(
            "SELECT type,name,tbl_name,rootpage,sql FROM sqlite_master "
            "WHERE name='messages_fts'"
        ).fetchone()
        assert row is not None
        conn.execute(
            "INSERT INTO sqlite_master (type,name,tbl_name,rootpage,sql) "
            "VALUES (?,?,?,?,?)",
            row,
        )
        conn.execute("PRAGMA writable_schema=OFF")
        conn.close()

        with pytest.raises(sqlite3.DatabaseError):
            StoreHolder(db_path=db_path, read_only=True)

        assert has_live_connection(db_path) is False

        # The writable heal must still take its forensic backup.
        healed = StoreHolder(db_path=db_path, read_only=False)
        healed.close()
        assert list(tmp_path.glob("*malformed-backup*"))

    def test_read_only_open_retries_transient_wal_ioerr(self, tmp_path, monkeypatch):
        """A transient SQLITE_IOERR on a read-only open must retry, not raise.

        A ``mode=ro`` connection cannot perform WAL recovery (recovery would
        need to write the -shm index, which read-only mode refuses), so a
        concurrent checkpoint / WAL reset / frame-flush on the writer side can
        surface "disk I/O error" to a reader on a perfectly healthy database
        (#100436). The transition window is millisecond-scale; a bounded retry
        must let the open succeed instead of 500-ing the /api/sessions poll
        and every other read-only opener.
        """
        import sqlite3

        from hermes_cli.sqlite_safe_read import has_live_connection

        db_path = tmp_path / "state.db"
        writable = StoreHolder(db_path=db_path)
        create_session(writable, "wal-race", source="cli")
        writable.close()

        real_connect = _holder._connect_tracked_db
        attempts = []

        def flaky_connect(*args, **kwargs):
            attempts.append(kwargs.get("uri"))
            if len(attempts) == 1:
                # First open lands inside the writer's WAL transition window.
                raise sqlite3.OperationalError("disk I/O error")
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(_holder, "_connect_tracked_db", flaky_connect)
        # Keep the test fast: one backoff tick is enough; the retry budget
        # itself is exercised by the attempt count below.
        monkeypatch.setattr(_holder, "_READ_ONLY_IOERR_RETRY_BACKOFF_S", 0.0)

        read_only = StoreHolder(db_path=db_path, read_only=True)
        try:
            assert read_only._fts_enabled is True
            search_messages(read_only, "wal-race")
        finally:
            read_only.close()

        assert len(attempts) >= 2, "the transient IOERR must be retried"
        assert has_live_connection(db_path) is False  # no leaked connections

    def test_read_only_open_exhausts_retry_budget_for_persistent_ioerr(
        self, tmp_path, monkeypatch
    ):
        """A persistent SQLITE_IOERR must exhaust the budget and raise.

        The retry exists to ride out a millisecond WAL transition — a
        storage layer that keeps failing after the full budget is genuinely
        broken and must surface the error (and not loop forever).
        """
        import sqlite3

        db_path = tmp_path / "state.db"
        writable = StoreHolder(db_path=db_path)
        create_session(writable, "broken-disk", source="cli")
        writable.close()

        attempts = []

        def bad_connect(*args, **kwargs):
            attempts.append(1)
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(_holder, "_connect_tracked_db", bad_connect)
        monkeypatch.setattr(_holder, "_READ_ONLY_IOERR_RETRY_BACKOFF_S", 0.0)
        budget = _holder._READ_ONLY_IOERR_RETRY_ATTEMPTS

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            StoreHolder(db_path=db_path, read_only=True)

        # budget + 1 = the initial attempt plus `budget` retries.
        assert len(attempts) == budget + 1
