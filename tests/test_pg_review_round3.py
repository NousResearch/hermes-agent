"""Regression tests for the third-round review blockers.

Each pins a specific failure identified at head a84b63b9d3:

1. ``_open_session_db_for_profile(None, read_only=True)`` detected a configured
   PostgreSQL DSN and then called ``SessionDB(read_only=True)``, whose own gate
   was ``if not read_only and db_path is None`` — so it selected the constructor
   mode that bypasses PostgreSQL. Dashboard status/session listing, cron
   history, usage analytics, and resume lookup all read through that helper, so
   live writes could be in PostgreSQL while every helper read consulted the
   local SQLite file.

2. ``_PostgresConnection.cursor()`` still routed through ``_call_with_retry``,
   which may reconnect. ``_reconnect()`` builds the replacement with
   ``autocommit=True``, so a statement issued after a mid-transaction swap
   self-commits outside the caller's ``BEGIN`` and without its
   transaction-scoped advisory locks.

3. ``resolve_postgres_dsn()`` read the active profile's selector through
   ``load_config()``, which degrades malformed YAML to defaults rather than
   raising. A fresh process whose only PostgreSQL selection lived in a
   now-malformed ``config.yaml`` silently opened SQLite.

4. ``open_store_for_profile()`` required ``sessions.postgres_dsn`` in the target
   profile's ``config.yaml``, but the documented deployment shape puts the
   credential-bearing DSN in that profile's ``.env``. A profile configured as
   documented could not be opened by a cross-profile reader.
"""

from __future__ import annotations


from pathlib import Path

import pytest

import hermes_state_pg_schema as pg_schema
from hermes_state_common import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 1. read_only must not silently select the SQLite backend
# ---------------------------------------------------------------------------


class TestReadOnlyDoesNotBypassPostgres:
    def test_read_only_still_resolves_the_postgres_backend(self, monkeypatch):
        """A read-only SessionDB must open PostgreSQL when one is configured.

        Behavioural, not source-shaped: construct with ``read_only=True`` and
        an explicit DSN, and assert the resulting handle is PostgreSQL-backed.
        """
        import hermes_state_postgres as hsp
        from hermes_state import SessionDB

        # A read-only open now validates and locks down the connection, so the
        # double must answer the schema/version probes rather than be an inert
        # sentinel.
        class _ReadyConn:
            def __init__(self):
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                outer = self

                class _Cur:
                    def fetchone(self):
                        if "information_schema" in sql:
                            return (1,)
                        if "from schema_version" in sql.lower():
                            return {"version": SCHEMA_VERSION}
                        return None

                    def fetchall(self):
                        return []
                return _Cur()

            def commit(self):
                return None

            def close(self):
                return None

        sentinel_conn = _ReadyConn()
        monkeypatch.setattr(
            hsp, "connect_postgres", lambda dsn: sentinel_conn
        )
        monkeypatch.setattr(pg_schema, "init_postgres_schema", lambda conn, ver: None)
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version",
            lambda c: max((m.version for m in pg_schema._PG_ONLY_MIGRATIONS),
                          default=0),
        )

        db = SessionDB(
            read_only=True, postgres_dsn="postgresql://example/db"
        )
        try:
            assert getattr(db, "_is_postgres", False) is True, (
                "a read_only SessionDB fell back to SQLite despite an explicit "
                "PostgreSQL DSN; every dashboard/helper read would then consult "
                "a different physical store than the live write path"
            )
            assert db._conn is sentinel_conn
        finally:
            db._conn = None  # avoid close() on the sentinel

    def test_maybe_open_postgres_serves_read_only_callers(self, monkeypatch):
        """A read-only request must still resolve to the PostgreSQL store."""
        import hermes_state_postgres as hsp

        opened: dict = {}

        class _ReadyConn:
            def execute(self, sql, params=()):
                class _Cur:
                    def fetchone(self):
                        if "information_schema" in sql:
                            return (1,)
                        if "from schema_version" in sql.lower():
                            return {"version": SCHEMA_VERSION}
                        return None

                    def fetchall(self):
                        return []
                return _Cur()

            def commit(self):
                return None

            def close(self):
                return None

        def _fake_connect(dsn):
            opened["dsn"] = dsn
            return _ReadyConn()

        monkeypatch.setattr(hsp, "connect_postgres", _fake_connect)
        monkeypatch.setattr(pg_schema, "init_postgres_schema", lambda conn, ver: None)
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version",
            lambda c: max((m.version for m in pg_schema._PG_ONLY_MIGRATIONS),
                          default=0),
        )

        conn = hsp.maybe_open_postgres(
            True, 1, dsn_override="postgresql://example/db"
        )

        assert conn is not None, (
            "maybe_open_postgres returned None for a read_only caller, sending "
            "it to SQLite even though a DSN was resolved"
        )
        assert opened["dsn"] == "postgresql://example/db"

    def test_dashboard_helper_opens_postgres_for_default_profile(
        self, monkeypatch, tmp_path
    ):
        """`_open_session_db_for_profile(None, read_only=True)` must not open SQLite.

        This is the exact call shape the dashboard's read paths use.
        """
        from hermes_cli import web_server_sessions as web_server

        sentinel = object()
        monkeypatch.setattr(
            "hermes_state_postgres.resolve_postgres_dsn",
            lambda *a, **k: "postgresql://example/db",
        )

        captured: dict = {}

        class _FakeSessionDB:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        monkeypatch.setattr("hermes_state.SessionDB", _FakeSessionDB)

        db = web_server._open_session_db_for_profile(None, read_only=True)

        assert isinstance(db, _FakeSessionDB), (
            "the helper fell through to the SQLite path (_open_session_db_at_path) "
            "even though a PostgreSQL DSN resolved"
        )
        assert captured["kwargs"]["postgres_dsn"] == "postgresql://example/db"


# ---------------------------------------------------------------------------
# 2. No transparent reconnect once BEGIN has succeeded
# ---------------------------------------------------------------------------


class _FakePsycopgError(Exception):
    pass


class _ClosedConn:
    """A psycopg-shaped connection that reports itself closed."""

    closed = 1

    def __init__(self):
        self.cursor_calls = 0
        self.executed: list = []

    def cursor(self):
        self.cursor_calls += 1
        return _RecordingCursor(self)

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


class _RecordingCursor:
    description = None
    rowcount = 0

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        self._conn.executed.append(sql)
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def close(self):
        return None


class TestNoReconnectInsideTransaction:
    def _make_conn(self):
        from hermes_state_postgres import _PostgresConnection

        raw = _ClosedConn()
        return _PostgresConnection(raw, dsn="postgresql://example/db"), raw

    def test_begin_then_connection_loss_fails_closed(self, monkeypatch):
        """BEGIN on A -> A closes -> next statement must NOT run on a new conn."""
        import hermes_state_postgres as hsp

        conn, raw = self._make_conn()

        # Mark the transaction open, as a successful BEGIN would.
        conn._in_transaction = True

        reconnects: list = []

        def _boom_reconnect(self):
            reconnects.append(True)
            raise AssertionError(
                "reconnected inside an open transaction — the replacement is "
                "autocommit, so the next statement would self-commit outside "
                "the caller's BEGIN and without its advisory locks"
            )

        monkeypatch.setattr(hsp._PostgresConnection, "_reconnect", _boom_reconnect)

        with pytest.raises(RuntimeError, match="lost inside an open transaction"):
            conn.execute("INSERT INTO sessions (id) VALUES (%s)", ("x",))

        assert not reconnects, "a reconnect was attempted mid-transaction"
        assert raw.executed == [], (
            "a statement executed despite the connection being dead inside a "
            "transaction; it would have landed outside the caller's BEGIN"
        )

    def test_outside_a_transaction_reconnect_is_still_allowed(self, monkeypatch):
        """The safety rule must not disable ordinary reconnect for idle handles."""
        import hermes_state_postgres as hsp

        conn, _raw = self._make_conn()
        conn._in_transaction = False

        healthy = _ClosedConn()
        healthy.closed = 0

        def _swap(self):
            self._conn = healthy

        monkeypatch.setattr(hsp._PostgresConnection, "_reconnect", _swap)

        conn.execute("SELECT 1")
        assert healthy.executed, (
            "a dead idle connection was not repaired; transparent reconnect "
            "outside a transaction is still wanted"
        )

    def test_commit_clears_transaction_state(self):
        """After commit the connection is reconnectable again."""
        conn, _raw = self._make_conn()
        conn._in_transaction = True
        conn.commit()
        assert conn._in_transaction is False

    def test_rollback_clears_transaction_state(self):
        conn, _raw = self._make_conn()
        conn._in_transaction = True
        conn.rollback()
        assert conn._in_transaction is False


# ---------------------------------------------------------------------------
# 3. Malformed ACTIVE config must fail closed
# ---------------------------------------------------------------------------


class TestActiveConfigFailsClosed:
    def test_malformed_active_config_raises(self, tmp_path, monkeypatch):
        """A broken active config.yaml must not resolve to SQLite.

        ``load_config()`` degrades malformed YAML to defaults by design, so the
        selector has to be validated with a strict read first.
        """
        import hermes_state_postgres as hsp

        home = tmp_path / "hermes_home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "sessions:\n  state_backend: [unclosed\n", encoding="utf-8"
        )
        monkeypatch.setattr(hsp, "get_hermes_home", lambda: home, raising=False)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_STATE_BACKEND", raising=False)
        monkeypatch.delenv("HERMES_STATE_DATABASE_URL", raising=False)
        monkeypatch.delenv("HERMES_STATE_POSTGRES_DSN", raising=False)

        with pytest.raises(RuntimeError, match="not usable|could not be read or parsed"):
            hsp.resolve_postgres_dsn()

    def test_absent_active_config_is_a_legitimate_sqlite_selection(
        self, tmp_path, monkeypatch
    ):
        import hermes_state_postgres as hsp

        home = tmp_path / "hermes_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_STATE_BACKEND", raising=False)
        monkeypatch.delenv("HERMES_STATE_DATABASE_URL", raising=False)
        monkeypatch.delenv("HERMES_STATE_POSTGRES_DSN", raising=False)

        # No config.yaml at all -> no selection -> SQLite, no raise.
        hsp._assert_active_config_parseable()

    def test_non_mapping_active_config_fails_closed(self, tmp_path, monkeypatch):
        """A structurally invalid root cannot authorize a SQLite fallback.

        A list or scalar at the top level means the operator wrote something
        whose intent cannot be determined — exactly as unknown as unparseable
        YAML. Reading it as "nothing configured" would silently route the
        process to SQLite while the operator may have meant PostgreSQL,
        splitting session history across two physical stores.
        """
        import hermes_state_postgres as hsp

        for body in ("- a\n- b\n", "just-a-string\n"):
            home = tmp_path / f"home_{abs(hash(body))}"
            home.mkdir()
            (home / "config.yaml").write_text(body, encoding="utf-8")
            monkeypatch.setattr(hsp, "get_hermes_home", lambda h=home: h,
                                raising=False)
            monkeypatch.setenv("HERMES_HOME", str(home))

            with pytest.raises(RuntimeError, match="not usable"):
                hsp._assert_active_config_parseable()


# ---------------------------------------------------------------------------
# 4. Named-profile reads must accept the documented .env DSN
# ---------------------------------------------------------------------------


class TestProfileEnvDsnIsHonoured:
    def test_dsn_read_from_profile_env(self, tmp_path):
        """The documented shape: backend in YAML, DSN only in that profile's .env."""
        from hermes_state_postgres import _dsn_from_profile_env

        pdir = tmp_path / "profiles" / "b"
        pdir.mkdir(parents=True)
        (pdir / ".env").write_text(
            "# comment\n"
            "SOME_OTHER=ignored\n"
            "HERMES_STATE_DATABASE_URL=postgresql://u:p@h:5432/bdb\n",
            encoding="utf-8",
        )

        assert _dsn_from_profile_env(pdir) == "postgresql://u:p@h:5432/bdb"

    def test_quoted_values_are_unwrapped(self, tmp_path):
        from hermes_state_postgres import _dsn_from_profile_env

        pdir = tmp_path / "profiles" / "b"
        pdir.mkdir(parents=True)
        (pdir / ".env").write_text(
            'HERMES_STATE_POSTGRES_DSN="postgresql://h/db"\n', encoding="utf-8"
        )

        assert _dsn_from_profile_env(pdir) == "postgresql://h/db"

    def test_database_url_wins_over_postgres_dsn(self, tmp_path):
        """Same precedence the active-process resolver uses."""
        from hermes_state_postgres import _dsn_from_profile_env

        pdir = tmp_path / "profiles" / "b"
        pdir.mkdir(parents=True)
        (pdir / ".env").write_text(
            "HERMES_STATE_POSTGRES_DSN=postgresql://h/second\n"
            "HERMES_STATE_DATABASE_URL=postgresql://h/first\n",
            encoding="utf-8",
        )

        assert _dsn_from_profile_env(pdir) == "postgresql://h/first"

    def test_absent_env_returns_empty(self, tmp_path):
        from hermes_state_postgres import _dsn_from_profile_env

        pdir = tmp_path / "profiles" / "b"
        pdir.mkdir(parents=True)
        assert _dsn_from_profile_env(pdir) == ""

    def test_reading_profile_env_does_not_mutate_process_env(self, tmp_path, monkeypatch):
        """Resolution must not leak the target's credential into os.environ."""
        import os

        from hermes_state_postgres import _dsn_from_profile_env

        monkeypatch.delenv("HERMES_STATE_DATABASE_URL", raising=False)
        pdir = tmp_path / "profiles" / "b"
        pdir.mkdir(parents=True)
        (pdir / ".env").write_text(
            "HERMES_STATE_DATABASE_URL=postgresql://h/bdb\n", encoding="utf-8"
        )

        _dsn_from_profile_env(pdir)

        assert os.environ.get("HERMES_STATE_DATABASE_URL") is None, (
            "reading a profile's .env leaked its DSN into the process "
            "environment, where a concurrent SessionDB() could observe it"
        )


    def test_seam_opens_a_profile_whose_dsn_lives_only_in_its_env(
        self, tmp_path, monkeypatch
    ):
        """The composed case the review asked for.

        Profile B selects PostgreSQL in its config.yaml and keeps the DSN ONLY
        in its own .env — the documented deployment shape, since the DSN
        carries a password. Profile A must be able to open B through the seam
        and reach the same physical store.

        Exercises ``open_store_for_profile`` end to end rather than the .env
        parser alone: a helper that works while the seam ignores it is exactly
        the gap this test exists to close.
        """
        import hermes_state_postgres as hsp
        from hermes_cli import profiles as profiles_mod

        pdir = tmp_path / "profiles" / "b"
        pdir.mkdir(parents=True)
        (pdir / "config.yaml").write_text(
            "sessions:\n  state_backend: postgres\n", encoding="utf-8"
        )
        (pdir / ".env").write_text(
            "HERMES_STATE_DATABASE_URL=postgresql://u:p@h:5432/bdb\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: True)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: str(pdir))

        # The active process (profile A) must not be the source of the answer.
        monkeypatch.delenv("HERMES_STATE_DATABASE_URL", raising=False)
        monkeypatch.delenv("HERMES_STATE_POSTGRES_DSN", raising=False)
        monkeypatch.delenv("HERMES_STATE_BACKEND", raising=False)

        seen: dict = {}

        class _FakeSessionDB:
            _is_postgres = True

            def __init__(self, *args, **kwargs):
                seen["postgres_dsn"] = kwargs.get("postgres_dsn")

            def close(self):
                return None

        monkeypatch.setattr("hermes_state.SessionDB", _FakeSessionDB)

        hsp.open_store_for_profile("b")

        assert seen.get("postgres_dsn") == "postgresql://u:p@h:5432/bdb", (
            "the seam did not reach profile B's own .env, so a profile "
            "configured in the documented shape (backend in YAML, DSN in .env) "
            "is unreadable by any cross-profile reader"
        )


class TestRawPropertyRespectsTransaction:
    """The `raw` property is a fifth reconnect path into the same hazard.

    `_call_with_retry` was hardened, but `raw` called `_ensure_live()`
    unconditionally. Callers that bypass the adapter — advisory-lock SQL,
    migration paths — would receive a REPLACEMENT connection while the caller
    believed it was inside a transaction. That replacement is
    `autocommit=True`, so statements issued on it self-commit outside the
    caller's BEGIN and without its transaction-scoped advisory locks.

    Found by live failure injection: closing the client handle mid-transaction
    and asserting the post-close row never becomes durable. The earlier
    `pg_terminate_backend` variant could not find it — the server rejects the
    statement either way, so it passed with the protection removed.
    """

    def _make_conn(self):
        from hermes_state_postgres import _PostgresConnection

        raw = _ClosedConn()
        return _PostgresConnection(raw, dsn="postgresql://example/db")

    def test_raw_does_not_reconnect_inside_a_transaction(self, monkeypatch):
        import hermes_state_postgres as hsp

        conn = self._make_conn()
        conn._in_transaction = True

        def _boom(self):
            raise AssertionError(
                "raw reconnected inside an open transaction; the replacement "
                "is autocommit, so any statement issued on it self-commits "
                "outside the caller's BEGIN"
            )

        monkeypatch.setattr(hsp._PostgresConnection, "_reconnect", _boom)

        with pytest.raises(RuntimeError, match="lost inside an open transaction"):
            _ = conn.raw

    def test_raw_still_repairs_an_idle_dead_handle(self, monkeypatch):
        """Outside a transaction, `raw` must keep its healing behaviour."""
        import hermes_state_postgres as hsp

        conn = self._make_conn()
        conn._in_transaction = False

        healthy = _ClosedConn()
        healthy.closed = 0

        def _swap(self):
            self._conn = healthy

        monkeypatch.setattr(hsp._PostgresConnection, "_reconnect", _swap)

        assert conn.raw is healthy
