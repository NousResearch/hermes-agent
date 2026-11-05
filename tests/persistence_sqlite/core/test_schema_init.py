"""Schema initialisation on a real StoreHolder: the journal mode, the explicit telegram topic schema, SCHEMA_SQL as the source of truth for live columns, one canonical async_delegations shape whatever opens the store first, ``_schema._reconcile_columns`` error handling under the open's lock patience, state_meta, and the Insights tool-call index recreated on reopen."""

import sqlite3

import pytest

from persistence.sqlite import _holder, _schema
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture()
def db(tmp_path):
    """Create a StoreHolder with a temp database file."""
    db_path = tmp_path / "test_state.db"
    holder = StoreHolder(db_path=db_path)
    yield holder
    holder.close()


class TestSchemaInit:
    def test_wal_mode(self, db):
        """Prefer WAL on fixed SQLite; DELETE on WAL-reset-vulnerable builds (#69784)."""
        from persistence.sqlite._state.wal import is_sqlite_wal_reset_vulnerable

        cursor = db._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0].lower()
        if is_sqlite_wal_reset_vulnerable():
            assert mode == "delete"
        else:
            assert mode == "wal"

    def test_telegram_topic_binding_roundtrip_requires_explicit_schema(self, backend, scope_a):
        store = backend.open(scope_a)
        db = store.holder()
        create_session(db,
            session_id="topic-session",
            source="telegram",
            user_id="208214988",
        )

        assert store.routing.get_topic_binding(chat_id="208214988", thread_id="17585") is None

        store.routing.bind_topic(
            chat_id="208214988",
            thread_id="17585",
            user_id="208214988",
            session_key="telegram:dm:208214988:thread:17585",
            session_id="topic-session",
        )

        binding = store.routing.get_topic_binding(chat_id="208214988", thread_id="17585")
        assert binding is not None
        assert binding.chat_id == "208214988"
        assert binding.thread_id == "17585"
        assert binding.user_id == "208214988"
        assert binding.session_key == "telegram:dm:208214988:thread:17585"
        assert binding.session_id == "topic-session"
        assert db.get_meta("telegram_dm_topic_schema_version") == "3"

    def test_schema_sql_is_source_of_truth(self, db):
        """Every column in SCHEMA_SQL exists in the live database.

        This is the architectural invariant: SCHEMA_SQL declares the
        desired schema, _reconcile_columns ensures it matches reality.
        """
        from persistence.sqlite._schema_ddl import SCHEMA_SQL

        expected = _schema._parse_schema_columns(SCHEMA_SQL, db.db_path.parent)
        for table_name, declared_cols in expected.items():
            live_cols = {
                r[1]
                for r in db._conn.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            }
            for col_name in declared_cols:
                assert col_name in live_cols, (
                    f"Column {col_name} declared in SCHEMA_SQL for {table_name} "
                    f"but missing from live DB. Live columns: {live_cols}"
                )


class TestAsyncDelegationsSchemaAgreement:
    """One durable-shape authority for async_delegations (#94691).

    The delegation tool used to carry its own CREATE TABLE + ALTER column
    list; it drifted from SCHEMA_SQL (a same-name column with a different
    shape depending on which authority touched the database first). The
    tool's initializer now routes through the canonical reconciler, so
    every opening order must land on the same canonical shape — compared
    over FULL PRAGMA table_info metadata (type, notnull, dflt_value, pk),
    not just column names, and with the canonical index set pinned.
    """

    def _table_info(self, conn):
        return {
            row[1]: (row[2], row[3], row[4], row[5])
            for row in conn.execute(
                "PRAGMA table_info(async_delegations)"
            ).fetchall()
        }

    def _canonical_shape(self):
        ref = __import__("sqlite3").connect(":memory:")
        try:
            from persistence.sqlite._schema_ddl import SCHEMA_SQL

            ref.executescript(SCHEMA_SQL)
            return self._table_info(ref), {
                row[0]
                for row in ref.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='async_delegations' AND sql IS NOT NULL"
                ).fetchall()
            }
        finally:
            ref.close()

    def _ledger_write(self, async_delegation, db_path):
        """One delegation ledger write, which opens ``db_path`` through the store."""
        from unittest.mock import patch

        with patch.object(async_delegation, "_db_path", lambda: db_path):
            async_delegation.mark_completion_delivered("schema-probe")

    def _legacy_db(self, db_path):
        """A database created before origin_session_id existed, carrying a
        pre-existing delegation row that must survive every opening order."""
        import sqlite3

        from persistence.sqlite._schema_ddl import SCHEMA_SQL

        legacy_sql = SCHEMA_SQL.replace(
            "    origin_session_id TEXT NOT NULL DEFAULT ''\n", ""
        ).replace(
            "    delivery_claimed_at REAL,\n",
            "    delivery_claimed_at REAL\n",
        )
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(legacy_sql)
            conn.execute(
                "INSERT INTO async_delegations (delegation_id, origin_session, origin_ui_session_id, state, dispatched_at, updated_at) VALUES ('legacy-1', 'sess-a', '', 'completed', 1.0, 1.0)"
            )
            conn.commit()
        finally:
            conn.close()

    def _assert_canonical(self, conn):
        expected_cols, expected_indexes = self._canonical_shape()
        live_cols = self._table_info(conn)
        assert live_cols == expected_cols
        live_indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='async_delegations' AND sql IS NOT NULL"
            ).fetchall()
        }
        assert live_indexes == expected_indexes
        row = conn.execute(
            "SELECT delegation_id, IFNULL(origin_session_id, '<null>') FROM async_delegations WHERE delegation_id='legacy-1'"
        ).fetchone()
        if row is not None:
            # The legacy row survived and the canonical '' default
            # backfilled the added column (SQLite ADD COLUMN ... DEFAULT
            # populates existing rows with the default).
            assert row[1] == ""

    def test_fresh_session_db_then_tool(self, tmp_path):
        import sqlite3

        from tools import async_delegation

        db_path = tmp_path / "state.db"
        db = StoreHolder(db_path=db_path)
        db.close()

        conn = sqlite3.connect(db_path)
        try:
            shape_before = self._table_info(conn)
            self._ledger_write(async_delegation, db_path)
            assert self._table_info(conn) == shape_before
            self._assert_canonical(conn)
        finally:
            conn.close()

    def test_legacy_store_then_session_db(self, tmp_path):
        db_path = tmp_path / "legacy-state.db"
        self._legacy_db(db_path)

        db = StoreHolder(db_path=db_path)
        try:
            self._assert_canonical(db._conn)
        finally:
            db.close()

    def test_legacy_store_then_tool_then_session_db(self, tmp_path):
        import sqlite3

        from tools import async_delegation

        db_path = tmp_path / "legacy-tool" / "state.db"
        db_path.parent.mkdir()
        self._legacy_db(db_path)
        self._ledger_write(async_delegation, db_path)

        db = StoreHolder(db_path=db_path)
        try:
            self._assert_canonical(db._conn)
        finally:
            db.close()


class TestReconcileColumnsErrorHandling:
    """_reconcile_columns must not bury migration failures (#79531/#80037).

    A locked ALTER used to be swallowed at DEBUG: startup "succeeded" with a
    half-reconciled schema and every session-list read then 500ed with
    "no such column" until an unrelated writable open. The contract now:
    duplicate-column races stay quiet, lock/busy propagates (so the open-time
    lock patience retries the whole init), everything else warns.
    """

    class _FailingAlterCursor:
        """Pass through to a real cursor, failing ALTER TABLE with ``exc``."""

        def __init__(self, real_cursor, exc):
            self._real = real_cursor
            self._exc = exc

        def execute(self, sql, *args, **kwargs):
            if sql.lstrip().upper().startswith("ALTER TABLE"):
                raise self._exc
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def _db_missing_column(self, tmp_path):
        """A store whose sessions table lacks last_read_at."""
        db_path = tmp_path / "state.db"
        seed = StoreHolder(db_path=db_path)
        seed.close()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("ALTER TABLE sessions DROP COLUMN last_read_at")
            conn.commit()
        finally:
            conn.close()
        return db_path

    def test_locked_alter_propagates(self, tmp_path):
        """database-is-locked must escape _reconcile_columns, not vanish.

        Propagation is what lets _connect_and_init_with_lock_patience retry
        the whole init with jittered backoff instead of serving a store
        that is silently behind SCHEMA_SQL.
        """
        db_path = self._db_missing_column(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = self._FailingAlterCursor(
                conn.cursor(),
                sqlite3.OperationalError("database is locked"),
            )
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                _schema._reconcile_columns(cursor, db_path.parent)
        finally:
            conn.close()

    def test_duplicate_column_race_stays_quiet(self, tmp_path, caplog):
        """A duplicate-column race is expected and must not warn or raise."""
        import logging

        db_path = self._db_missing_column(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = self._FailingAlterCursor(
                conn.cursor(),
                sqlite3.OperationalError(
                    "duplicate column name: last_read_at"
                ),
            )
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                _schema._reconcile_columns(cursor, db_path.parent)
        finally:
            conn.close()
        assert not [
            r for r in caplog.records if "reconcile" in r.getMessage()
        ]

    def test_other_alter_failures_warn(self, tmp_path, caplog):
        """Schema mistakes (e.g. un-ADDable NOT NULL) log at WARNING."""
        import logging

        db_path = self._db_missing_column(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = self._FailingAlterCursor(
                conn.cursor(),
                sqlite3.OperationalError(
                    "Cannot add a NOT NULL column with default value NULL"
                ),
            )
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                _schema._reconcile_columns(cursor, db_path.parent)
        finally:
            conn.close()
        warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and "reconcile" in r.getMessage()
        ]
        assert warnings, "un-ADDable column failure must be logged at WARNING+"

    def test_locked_alter_is_retried_by_open_lock_patience(self, tmp_path, monkeypatch):
        """End-to-end: a transiently locked ALTER heals on open retry.

        The lock-patience wrapper retries on OperationalError raised out of
        _connect_and_init; before this fix _reconcile_columns caught the
        error internally so the retry never saw it and the store stayed
        stale forever.
        """
        db_path = self._db_missing_column(tmp_path)

        original = _schema._reconcile_columns
        calls = {"n": 0}

        def flaky_reconcile(cursor, home):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return original(cursor, home)

        monkeypatch.setattr(_schema, "_reconcile_columns", flaky_reconcile)
        # Keep the retry fast — patience budget is 20s by default.
        monkeypatch.setattr(_holder.StoreHolder, "_WRITE_RETRY_SLOW_MIN_S", 0.001)
        monkeypatch.setattr(_holder.StoreHolder, "_WRITE_RETRY_SLOW_MAX_S", 0.005)

        healed = StoreHolder(db_path=db_path)
        try:
            cols = {
                r[1]
                for r in healed._conn.execute(
                    'PRAGMA table_info("sessions")'
                ).fetchall()
            }
        finally:
            healed.close()
        assert calls["n"] >= 2, "lock patience must retry the init"
        assert "last_read_at" in cols


class TestStateMeta:
    def test_get_meta_missing_returns_none(self, db):
        assert db.get_meta("nonexistent") is None

    def test_set_then_get_meta(self, db):
        db.set_meta("foo", "bar")
        assert db.get_meta("foo") == "bar"


class TestInsightsToolCallIndex:
    """The Insights assistant tool-call scan has a predicate-aligned index.

    ``InsightsEngine._get_tool_usage`` / ``_get_skill_usage`` filter messages by
    ``role = 'assistant' AND tool_calls IS NOT NULL``.  A partial index over that
    predicate keeps the scan off the full ``messages`` table on a large state.db.
    """

    _INDEX = "idx_messages_assistant_calls_by_session"

    def _index_defn(self, conn):
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (self._INDEX,),
        ).fetchone()
        return row["sql"] if row else None

    def test_index_created_on_existing_db(self, tmp_path):
        """Reopening a DB that predates the index must create it (SCHEMA_SQL is
        re-run on every open; role/tool_calls are original base columns)."""
        db_path = tmp_path / "legacy.db"
        db = StoreHolder(db_path=db_path)
        # Simulate a database created before the index shipped.
        db._conn.execute(f"DROP INDEX IF EXISTS {self._INDEX}")
        db._conn.commit()
        assert self._index_defn(db._conn) is None
        db.close()

        db2 = StoreHolder(db_path=db_path)
        try:
            assert self._index_defn(db2._conn) is not None, (
                "index not recreated when reopening an existing database"
            )
        finally:
            db2.close()
