"""Read-only PostgreSQL opens must not be able to change the store.

A read-only open serves the dashboard's status/session listing, cron history,
usage analytics, and resume lookup. Review finding at head `cac143fa40`:

    Removing `read_only` as a backend selector fixes the split-store read.
    However, `maybe_open_postgres(read_only=True, ...)` still unconditionally
    calls `init_postgres_schema()`. A status/resume/analytics reader can
    therefore create or reconcile schema through a path presented as
    read-only, and the returned Postgres handle has no engine- or
    adapter-enforced write prohibition.

Three invariants follow, and each is pinned here:

  1. a read-only open runs NO DDL — provisioning belongs to writable opens;
  2. a read-only open against an absent or behind-this-build schema FAILS
     rather than mutating it or serving a store it cannot correctly read;
  3. the returned handle carries an engine-enforced write prohibition.

Invariant 4 is the previous round's fix, guarded here against regression:
`read_only` must still resolve the PostgreSQL backend, never fall back to
SQLite.
"""

from __future__ import annotations

import sqlite3

import pytest

import hermes_state_pg_schema as pg_schema
from hermes_state_common import SCHEMA_VERSION


class _FakeCursor:
    def __init__(self, conn, rows=None):
        self._conn = conn
        self._rows = rows if rows is not None else []

    def execute(self, sql, params=()):
        self._conn.executed.append(sql.strip())
        return self

    def executescript(self, sql):
        self._conn.executed.append("SCRIPT:" + sql.strip()[:40])
        return self

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        return None


class _FakeConn:
    """Minimal connection double that records every statement it is given."""

    def __init__(self, *, has_sessions=True, version=SCHEMA_VERSION):
        self.executed: list[str] = []
        self.commits = 0
        self._has_sessions = has_sessions
        self._version = version

    def execute(self, sql, params=()):
        self.executed.append(sql.strip())
        low = sql.lower()
        if "information_schema.tables" in low:
            return _FakeCursor(self, [(1,)] if self._has_sessions else [])
        if "from schema_version" in low:
            return _FakeCursor(self, [{"version": self._version}])
        return _FakeCursor(self)

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1

    def close(self):
        return None


def _expected_version():
    from hermes_state_pg_schema import _PG_ONLY_MIGRATIONS

    return max((m.version for m in _PG_ONLY_MIGRATIONS), default=0)


# ---------------------------------------------------------------------------
# 1. A read-only open runs no DDL
# ---------------------------------------------------------------------------


class TestReadOnlyOpenRunsNoDDL:
    def test_read_only_does_not_call_schema_init(self, monkeypatch):
        """Provisioning through a path presented as read-only is the bug."""
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        called: list[str] = []

        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(
            pg_schema, "init_postgres_schema",
            lambda c, v: called.append("init"),
        )
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version", lambda c: _expected_version()
        )

        hsp.maybe_open_postgres(True, 1, dsn_override="postgresql://h/db")

        assert called == [], (
            "a read-only open ran init_postgres_schema; a status/analytics "
            "reader must not be able to create or reconcile schema"
        )

    def test_writable_open_still_initialises_schema(self, monkeypatch):
        """The owner path must keep provisioning — don't over-correct."""
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        called: list[str] = []

        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(
            pg_schema, "init_postgres_schema",
            lambda c, v: called.append("init"),
        )

        hsp.maybe_open_postgres(False, 1, dsn_override="postgresql://h/db")

        assert called == ["init"], "writable opens must still provision schema"

    def test_read_only_issues_no_ddl_statements(self, monkeypatch):
        """Belt and braces: inspect what actually reached the connection."""
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version", lambda c: _expected_version()
        )

        hsp.maybe_open_postgres(True, 1, dsn_override="postgresql://h/db")

        ddl = [
            s for s in conn.executed
            if s.upper().startswith(("CREATE", "ALTER", "DROP", "INSERT",
                                     "UPDATE", "DELETE", "SCRIPT:"))
        ]
        assert ddl == [], f"read-only open issued mutating statements: {ddl}"


# ---------------------------------------------------------------------------
# 2. Fail closed on an unusable store
# ---------------------------------------------------------------------------


class TestReadOnlyFailsClosed:
    def test_absent_schema_raises_instead_of_provisioning(self, monkeypatch):
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=False)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)

        with pytest.raises(RuntimeError, match="no Hermes schema found"):
            hsp.maybe_open_postgres(True, 1, dsn_override="postgresql://h/db")

    def test_schema_behind_this_build_raises(self, monkeypatch):
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version",
            lambda c: _expected_version() - 1,
        )

        with pytest.raises(RuntimeError, match="migration version"):
            hsp.maybe_open_postgres(True, 1, dsn_override="postgresql://h/db")

    def test_schema_ahead_of_this_build_is_allowed(self, monkeypatch):
        """A newer store still satisfies an older reader's queries.

        Refusing it would break every mixed-version deployment mid-rollout.
        The schema only ever grows, so 'ahead' is safe; only 'behind' is not.
        """
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version",
            lambda c: _expected_version() + 5,
        )

        assert hsp.maybe_open_postgres(
            True, 1, dsn_override="postgresql://h/db"
        ) is conn

    def test_error_message_does_not_leak_the_password(self, monkeypatch):
        """DSNs carry credentials; a refusal message must not print them."""
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=False)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        secret = "hunter2seekrit"

        with pytest.raises(RuntimeError) as excinfo:
            hsp.maybe_open_postgres(
                True, 1,
                dsn_override=f"postgresql://user:{secret}@host:5432/db?sslmode=require",
            )

        assert secret not in str(excinfo.value)
        assert "host:5432/db" in str(excinfo.value), (
            "the message should still identify WHICH store was refused"
        )


# ---------------------------------------------------------------------------
# 3. Enforced write prohibition
# ---------------------------------------------------------------------------


class TestReadOnlyWriteProhibition:
    def test_session_is_set_read_only_before_anything_else(self, monkeypatch):
        """The prohibition must be in force before any other statement runs.

        Engine-level (`default_transaction_read_only`) rather than an
        adapter-side SQL classifier: a parser has permanent false-negative
        holes and protects nothing against code reaching the raw connection.
        """
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version", lambda c: _expected_version()
        )

        hsp.maybe_open_postgres(True, 1, dsn_override="postgresql://h/db")

        assert conn.executed, "no statements were issued at all"
        first = conn.executed[0].lower()
        assert "default_transaction_read_only" in first and " on" in first, (
            f"the read-only prohibition was not the first statement; got: "
            f"{conn.executed[0]!r}"
        )

    def test_writable_open_does_not_set_read_only(self, monkeypatch):
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(pg_schema, "init_postgres_schema", lambda c, v: None)

        hsp.maybe_open_postgres(False, 1, dsn_override="postgresql://h/db")

        assert not any(
            "default_transaction_read_only" in s.lower() for s in conn.executed
        ), "a writable open must not put the session into read-only mode"


# ---------------------------------------------------------------------------
# 4. Regression guard for the PREVIOUS round's fix
# ---------------------------------------------------------------------------


class TestReadOnlyStillSelectsPostgres:
    def test_read_only_does_not_fall_back_to_sqlite(self, monkeypatch):
        """read_only must never be a backend selector again.

        Gating the backend on it sent every dashboard reader to the local
        state.db while writes went to PostgreSQL.
        """
        import hermes_state_postgres as hsp

        conn = _FakeConn(has_sessions=True)
        monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
        monkeypatch.setattr(
            pg_schema, "postgres_migration_version", lambda c: _expected_version()
        )

        assert hsp.maybe_open_postgres(
            True, 1, dsn_override="postgresql://h/db"
        ) is conn, "read_only fell back to SQLite (returned None)"


class _LedgerConnection:
    """Real version rows; only PostgreSQL's read-only/catalog syntax is adapted."""

    def __init__(self, path, shared_version, migration_version):
        self.raw = sqlite3.connect(path)
        self.raw.row_factory = sqlite3.Row
        self.raw.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY)")
        self.raw.execute("CREATE TABLE pg_migration_version(version INTEGER)")
        self.raw.execute("INSERT INTO pg_migration_version VALUES (?)", (migration_version,))
        if shared_version is not None:
            self.raw.execute("CREATE TABLE schema_version(version INTEGER)")
            self.raw.execute("INSERT INTO schema_version VALUES (?)", (shared_version,))
        self.raw.commit()
        self.executed = []
        self.closed = False

    def execute(self, sql, params=()):
        self.executed.append(sql)
        if sql == "SET default_transaction_read_only = on":
            return self.raw.execute("PRAGMA query_only = ON")
        if "information_schema.tables" in sql:
            return self.raw.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'")
        return self.raw.execute(sql, params)

    def commit(self):
        self.raw.commit()

    def close(self):
        self.raw.close()
        self.closed = True


def _assert_only_read_probes(conn):
    assert conn.executed[0] == "SET default_transaction_read_only = on"
    assert all(statement.lstrip().upper().startswith(("SELECT ", "SET ")) for statement in conn.executed)


@pytest.mark.parametrize("shared_delta, migration_delta", [(None, 0), (-1, 0), (-1, 1)])
def test_current_pg_ledger_cannot_hide_missing_or_stale_shared_schema(
    tmp_path, monkeypatch, shared_delta, migration_delta,
):
    import hermes_state_postgres as hsp

    shared_version = None if shared_delta is None else SCHEMA_VERSION + shared_delta
    conn = _LedgerConnection(tmp_path / "ledger.db", shared_version, _expected_version() + migration_delta)
    monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
    secret = "test-ledger-password"
    with pytest.raises(RuntimeError, match="shared schema version") as caught:
        hsp.maybe_open_postgres(
            True, SCHEMA_VERSION, dsn_override=f"postgresql://user:{secret}@host/db",
        )
    assert conn.closed, "a refused read-only open must release its connection"
    assert secret not in str(caught.value)
    _assert_only_read_probes(conn)
    # Missing schema stays absent; the reader cannot manufacture a current ledger.
    with sqlite3.connect(tmp_path / "ledger.db") as check:
        if shared_version is None:
            assert check.execute("SELECT 1 FROM sqlite_master WHERE name = 'schema_version'").fetchone() is None
        else:
            assert check.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == shared_version


@pytest.mark.parametrize("shared_delta, migration_delta", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_readonly_accepts_current_or_newer_versions_in_both_ledgers(
    tmp_path, monkeypatch, shared_delta, migration_delta,
):
    import hermes_state_postgres as hsp

    shared_version = SCHEMA_VERSION + shared_delta
    migration_version = _expected_version() + migration_delta
    conn = _LedgerConnection(tmp_path / "ledger.db", shared_version, migration_version)
    monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
    try:
        assert hsp.maybe_open_postgres(True, SCHEMA_VERSION, dsn_override="postgresql://host/db") is conn
        assert not conn.closed
        _assert_only_read_probes(conn)
        assert conn.raw.execute("PRAGMA query_only").fetchone()[0] == 1
        assert conn.raw.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == shared_version
        assert conn.raw.execute("SELECT MAX(version) FROM pg_migration_version").fetchone()[0] == migration_version
    finally:
        conn.close()


def test_readonly_closes_when_shared_version_cannot_be_read(tmp_path, monkeypatch):
    import hermes_state_postgres as hsp

    conn = _LedgerConnection(tmp_path / "ledger.db", "invalid", _expected_version())
    monkeypatch.setattr(hsp, "connect_postgres", lambda dsn: conn)
    with pytest.raises(RuntimeError, match="could not read the shared schema version"):
        hsp.maybe_open_postgres(True, SCHEMA_VERSION, dsn_override="postgresql://host/db")
    assert conn.closed
    _assert_only_read_probes(conn)
