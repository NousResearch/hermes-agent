"""Unconditional session_model_usage PK heal (#73823, salvage of #73838).

Installs whose state.db reached ``schema_version >= 22`` before the
``task`` dimension was added carry a 5-column PRIMARY KEY on
``session_model_usage``. Later 6-column keys include ``task`` but not
``pricing_version``. The column reconciler ADDs missing columns as bare
nullables, but SQLite cannot ALTER a primary key, so upserts that name
the new key column in ON CONFLICT fail until this rebuild.

``_heal_session_model_usage_pk`` runs unconditionally on every open
(same pattern as ``_heal_gateway_routing_pk``) and rebuilds the table
once, inside an FK-off window (OR IGNORE does not suppress FK
violations and the connection enables foreign_keys before init).
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION

LEGACY_SQL = """
    CREATE TABLE session_model_usage (
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        model TEXT NOT NULL,
        billing_provider TEXT NOT NULL DEFAULT '',
        billing_base_url TEXT NOT NULL DEFAULT '',
        billing_mode TEXT NOT NULL DEFAULT '',
        api_call_count INTEGER NOT NULL DEFAULT 0,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
        cache_write_tokens INTEGER NOT NULL DEFAULT 0,
        reasoning_tokens INTEGER NOT NULL DEFAULT 0,
        estimated_cost_usd REAL NOT NULL DEFAULT 0,
        actual_cost_usd REAL NOT NULL DEFAULT 0,
        cost_status TEXT,
        cost_source TEXT,
        first_seen REAL,
        last_seen REAL,
        PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode)
    )
"""


def _make_stale_v22_db(tmp_path, usage_rows=(), sessions=("s1",)):
    """Build a state.db in the stale-v22+ shape: current schema everywhere,
    but session_model_usage carrying the legacy 5-column PK with ``task``
    reconciler-appended OUTSIDE the key, and schema_version already at
    current — so the version-gated v22 rebuild can never run."""
    db_path = tmp_path / "state.db"
    # Born-current DB for everything else...
    db = SessionDB(db_path=db_path)
    for sid in sessions:
        db.create_session(sid, "cli")
    db.close()
    # ...then regress session_model_usage to the legacy shape.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE session_model_usage")
    conn.execute(LEGACY_SQL)
    # Mimic the column reconciler: task appended OUTSIDE the primary key.
    conn.execute('ALTER TABLE session_model_usage ADD COLUMN "task" TEXT')
    conn.executemany(
        "INSERT INTO session_model_usage "
        "(session_id, model, input_tokens, output_tokens) VALUES (?, ?, ?, ?)",
        list(usage_rows),
    )
    # Version already current: proves the heal does not depend on the gate.
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    conn.commit()
    conn.close()
    return db_path


def _pk_cols(db):
    rows = db._conn.execute(
        'PRAGMA table_info("session_model_usage")'
    ).fetchall()
    return sorted(r["name"] for r in rows if r["pk"])


class TestSessionModelUsagePkHeal:
    def test_stale_v22_pk_rebuilt_and_accounting_restored(self, tmp_path):
        """The broken-PK table is rebuilt on open even though schema_version
        is already current, and the usage upsert works again."""
        db_path = _make_stale_v22_db(
            tmp_path, usage_rows=[("s1", "m-old", 10, 20)]
        )
        db = SessionDB(db_path=db_path)
        try:
            assert "task" in _pk_cols(db)
            assert "pricing_version" in _pk_cols(db)
            # Existing rows survive the rebuild (task backfilled to '').
            row = db._conn.execute(
                "SELECT task, input_tokens FROM session_model_usage "
                "WHERE session_id='s1' AND model='m-old'"
            ).fetchone()
            assert row is not None
            assert row["task"] == ""
            assert row["input_tokens"] == 10
            # The killed write path works again: the upsert used to abort
            # the whole transaction with an ON CONFLICT mismatch.
            db.update_token_counts(
                "s1", input_tokens=5, output_tokens=7,
                model="m-new", billing_provider="p", api_call_count=1,
            )
            row = db._conn.execute(
                "SELECT input_tokens FROM session_model_usage "
                "WHERE session_id='s1' AND model='m-new'"
            ).fetchone()
            assert row is not None and row["input_tokens"] == 5
        finally:
            db.close()

    def test_orphan_rows_survive_fk_enforcement(self, tmp_path):
        """The rebuild copies rows inside an FK-off window: an orphaned
        usage row (session pruned while accounting was broken) must not
        abort the heal — OR IGNORE does NOT suppress FK violations."""
        db_path = _make_stale_v22_db(
            tmp_path,
            usage_rows=[("s1", "m1", 1, 1), ("ghost-session", "m1", 2, 2)],
        )
        db = SessionDB(db_path=db_path)
        try:
            assert "task" in _pk_cols(db)
            assert "pricing_version" in _pk_cols(db)
            rows = db._conn.execute(
                "SELECT session_id FROM session_model_usage ORDER BY session_id"
            ).fetchall()
            assert [r["session_id"] for r in rows] == ["ghost-session", "s1"]
            # FK enforcement is restored after the heal window.
            assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        finally:
            db.close()

    def test_healthy_db_is_a_noop(self, tmp_path):
        """A DB born with the composite PK is left untouched (idempotence)."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s1", "cli")
            db.update_token_counts(
                "s1", input_tokens=3, model="m", billing_provider="p",
                api_call_count=1,
            )
            assert "task" in _pk_cols(db)
            assert "pricing_version" in _pk_cols(db)
            # Re-running the heal directly is a no-op.
            cur = db._conn.cursor()
            db._heal_session_model_usage_pk(cur)
            row = db._conn.execute(
                "SELECT input_tokens FROM session_model_usage "
                "WHERE session_id='s1'"
            ).fetchone()
            assert row is not None and row["input_tokens"] == 3
        finally:
            db.close()

    def test_no_legacy_leftover_table(self, tmp_path):
        """The rename-copy-drop leaves no *_legacy_pk residue behind."""
        db_path = _make_stale_v22_db(tmp_path, usage_rows=[("s1", "m1", 1, 1)])
        db = SessionDB(db_path=db_path)
        try:
            left = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='session_model_usage_legacy_pk'"
            ).fetchone()
            assert left is None
        finally:
            db.close()

    def test_six_column_task_pk_gains_pricing_version(self, tmp_path):
        """Current-main 6-column PKs (task in the key, no revision) rebuild."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", "cli")
        db.close()
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE session_model_usage")
        conn.execute("""
            CREATE TABLE session_model_usage (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                billing_provider TEXT NOT NULL DEFAULT '',
                billing_base_url TEXT NOT NULL DEFAULT '',
                billing_mode TEXT NOT NULL DEFAULT '',
                task TEXT NOT NULL DEFAULT '',
                api_call_count INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                actual_cost_usd REAL NOT NULL DEFAULT 0,
                cost_status TEXT,
                cost_source TEXT,
                first_seen REAL,
                last_seen REAL,
                PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
            )
        """)
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model, task, input_tokens) "
            "VALUES ('s1', 'm-old', '', 10)"
        )
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        conn.commit()
        conn.close()

        db = SessionDB(db_path=db_path)
        try:
            assert "pricing_version" in _pk_cols(db)
            assert "task" in _pk_cols(db)
            db.update_token_counts(
                "s1", input_tokens=4, model="m-old", billing_provider="p",
                pricing_version="1.aaaaaaaaaaaa", api_call_count=1,
            )
            db.update_token_counts(
                "s1", input_tokens=6, model="m-old", billing_provider="p",
                pricing_version="1.bbbbbbbbbbbb", api_call_count=1,
            )
            rows = db._conn.execute(
                "SELECT pricing_version, input_tokens FROM session_model_usage "
                "WHERE session_id='s1' AND model='m-old' ORDER BY pricing_version"
            ).fetchall()
            by_version = {r["pricing_version"]: r["input_tokens"] for r in rows}
            assert by_version[""] == 10
            assert by_version["1.aaaaaaaaaaaa"] == 4
            assert by_version["1.bbbbbbbbbbbb"] == 6
        finally:
            db.close()
