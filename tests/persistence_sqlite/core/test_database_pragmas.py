"""The SQLite connection setup the backend's StoreHolder applies: the journal-mode probe in ``apply_wal_with_fallback``, ``sqlite_support.apply_database_pragmas`` honouring the ``database`` config section, and the configured pragmas reaching the holder's writer, reader and read-only connections."""

import pytest

import hermes_state_wal as state_wal


# ---------------------------------------------------------------------------
# apply_wal_with_fallback — read-only probe tests
# ---------------------------------------------------------------------------


class TestApplyWalProbe:
    """Unit tests for the journal_mode probe in apply_wal_with_fallback."""

    @pytest.fixture(autouse=True)
    def _assume_fixed_sqlite(self, monkeypatch):
        """These cases cover the fixed-SQLite WAL path (not the #69784 gate)."""

        monkeypatch.setattr(
            state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
        )

    def test_sets_wal_on_fresh_connection(self, tmp_path):
        """Probe sees 'delete', then set-pragma runs and returns 'wal'."""
        import sqlite3
        from hermes_state_wal import apply_wal_with_fallback

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                return super().execute(sql, params)

        db_path = tmp_path / "fresh.db"
        conn = _TracingConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        assert result == "wal"
        assert any("journal_mode=WAL" in sql for sql in conn.executed), (
            "set-pragma must fire on a fresh (non-WAL) connection"
        )

    def test_apply_wal_concurrent_connects_no_eio(self, tmp_path):
        """20 threads calling connect() on the same DB must not see disk I/O error."""
        import sys
        import threading
        import sqlite3
        from hermes_state_wal import apply_wal_with_fallback

        db_path = tmp_path / "concurrent.db"
        errors = []

        def _connect_cycle():
            for _ in range(5):
                try:
                    conn = sqlite3.connect(str(db_path))
                    apply_wal_with_fallback(conn)
                    conn.close()
                except sqlite3.OperationalError as exc:
                    if "disk i/o error" in str(exc).lower():
                        errors.append(exc)

        threads = [threading.Thread(target=_connect_cycle) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"disk I/O errors from concurrent connects: {errors}"

        # Linux-only: no (deleted) WAL/SHM FDs should accumulate.
        if sys.platform == "linux":
            import os

            fd_dir = f"/proc/{os.getpid()}/fd"
            deleted_fds = []
            for fd_name in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd_name))
                    if "(deleted)" in target and (
                        "wal" in target.lower() or "shm" in target.lower()
                    ):
                        deleted_fds.append(target)
                except OSError:
                    pass
            assert not deleted_fds, f"stale deleted WAL/SHM FDs: {deleted_fds}"


class TestApplyDatabasePragmas:
    """Config-driven WAL-sizing pragma application (database: section)."""

    @staticmethod
    def _patch_cfg(monkeypatch, cfg):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: cfg,
        )

    def test_honors_wal_autocheckpoint_from_config(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state_wal import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            self._patch_cfg(monkeypatch, {"database": {"wal_autocheckpoint": 250}})
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 250
        finally:
            conn.close()

    def test_honors_journal_size_limit_from_config(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state_wal import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            self._patch_cfg(
                monkeypatch, {"database": {"journal_size_limit": 10485760}}
            )
            apply_database_pragmas(conn, db_label="test.db")
            assert (
                conn.execute("PRAGMA journal_size_limit").fetchone()[0] == 10485760
            )
        finally:
            conn.close()

    def test_noop_when_database_section_missing(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state_wal import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            self._patch_cfg(monkeypatch, {})
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        finally:
            conn.close()

    def test_never_touches_journal_mode(self, tmp_path, monkeypatch):
        """journal_mode is owned by apply_wal_with_fallback — a database:
        journal_mode entry must NOT cause a second, unguarded mode switch."""
        import sqlite3
        from hermes_state_wal import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            self._patch_cfg(monkeypatch, {"database": {"journal_mode": "delete"}})
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()

    def test_ignores_non_integer_values(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state_wal import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            before = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            self._patch_cfg(
                monkeypatch, {"database": {"wal_autocheckpoint": "lots"}}
            )
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == before
        finally:
            conn.close()

    def test_ignores_non_integer_performance_values(self, tmp_path, monkeypatch):
        """Garbage cache_size/mmap_size/temp_store values must be rejected."""
        import sqlite3
        from hermes_state_wal import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            before = {
                name: conn.execute(f"PRAGMA {name}").fetchone()[0]
                for name in ("cache_size", "mmap_size", "temp_store")
            }
            self._patch_cfg(
                monkeypatch,
                {
                    "database": {
                        "cache_size": "big",
                        "mmap_size": [256],
                        "temp_store": "ram please",
                    }
                },
            )
            apply_database_pragmas(conn, db_label="test.db")
            after = {
                name: conn.execute(f"PRAGMA {name}").fetchone()[0]
                for name in ("cache_size", "mmap_size", "temp_store")
            }
            assert after == before
        finally:
            conn.close()


class TestPerformancePragmasEndToEnd:
    """E2E guard for PR #71755: config-gated cache_size / mmap_size /
    temp_store must reach EVERY connection type (writer, read-only
    cross-profile attach, WAL per-thread reader) — and default installs
    (no ``database:`` keys) must see byte-identical SQLite defaults.

    NOTE: SQLite's compiled-in default for ``cache_size`` is already
    ``-2000``, so the configured value here is ``-16000`` — a value the
    test can actually discriminate from the default (a reverted prod
    change must FAIL this test, not accidentally pass it).
    """

    PRAGMAS = ("cache_size", "mmap_size", "temp_store")
    CONFIGURED = {"cache_size": -16000, "mmap_size": 1048576, "temp_store": 2}

    @staticmethod
    def _read(conn):
        return {
            name: conn.execute(f"PRAGMA {name}").fetchone()[0]
            for name in ("cache_size", "mmap_size", "temp_store")
        }

    @staticmethod
    def _sqlite_defaults(tmp_path):
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "baseline.db"))
        try:
            return {
                name: conn.execute(f"PRAGMA {name}").fetchone()[0]
                for name in ("cache_size", "mmap_size", "temp_store")
            }
        finally:
            conn.close()

    def _fresh_home(self, tmp_path, monkeypatch, config_text=None):

        # Local venvs may bundle a WAL-reset-vulnerable SQLite (e.g. 3.46.0),
        # which would silently disable WAL and skip the per-thread reader
        # path. Force WAL eligibility so _get_read_conn is truly exercised
        # (established pattern used by the WAL tests above).
        monkeypatch.setattr(
            state_wal, "is_sqlite_wal_reset_vulnerable",
            lambda version_info=None: False,
        )
        home = tmp_path / "hermes_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        if config_text is not None:
            (home / "config.yaml").write_text(config_text)
        return home

    def test_configured_pragmas_reach_all_connection_types(
        self, tmp_path, monkeypatch
    ):
        from persistence.sqlite._holder import StoreHolder

        home = self._fresh_home(
            tmp_path,
            monkeypatch,
            "database:\n"
            "  cache_size: -16000\n"
            "  temp_store: 2\n"
            "  mmap_size: 1048576\n",
        )
        db_path = home / "state.db"
        db = StoreHolder(db_path=db_path)
        try:
            # Writer connection.
            assert self._read(db._conn) == self.CONFIGURED
            # WAL per-thread reader.
            rconn = db._get_read_conn()
            assert rconn is not None, "WAL reader expected on local filesystem"
            assert self._read(rconn) == self.CONFIGURED
        finally:
            db.close()

        # Read-only cross-profile attach.
        ro = StoreHolder(db_path=db_path, read_only=True)
        try:
            assert self._read(ro._conn) == self.CONFIGURED
        finally:
            ro.close()

    def test_defaults_unchanged_without_config(self, tmp_path, monkeypatch):
        """No database: keys in config.yaml → SQLite defaults untouched."""
        from persistence.sqlite._holder import StoreHolder

        defaults = self._sqlite_defaults(tmp_path)
        home = self._fresh_home(tmp_path, monkeypatch, config_text=None)
        db_path = home / "state.db"
        db = StoreHolder(db_path=db_path)
        try:
            assert self._read(db._conn) == defaults
            rconn = db._get_read_conn()
            if rconn is not None:
                assert self._read(rconn) == defaults
        finally:
            db.close()

        ro = StoreHolder(db_path=db_path, read_only=True)
        try:
            assert self._read(ro._conn) == defaults
        finally:
            ro.close()
