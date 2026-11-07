"""The raw ``state.db`` schema catch-up on the SQLite backend: one declared order (columns, the canonical DDL, columns again) brings an old store up to date, an index on a missing column included."""

import sqlite3
from contextlib import closing

from persistence.sqlite import _schema
from persistence.sqlite._schema import reconcile_state_schema
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def _lagging_store_with_history(home, rows=5):
    """A real store with some history, its indexed ``last_read_at`` column dropped."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / "state.db"
    with open_persistence(profile_scope(home=home)) as store:
        seed_session(store, "kept", "cli")
        seed_messages(store, "kept", *({"role": "user", "content": f"history {i} " * 40} for i in range(rows)))
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
        conn.execute("ALTER TABLE sessions DROP COLUMN last_read_at")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return path


def _session_columns(path):
    with closing(sqlite3.connect(path)) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}


def test_every_raw_schema_catch_up_handles_an_old_indexed_column(tmp_path):
    """Every raw caller catches an old store (indexed column missing) up through ``apply_declared_schema``, the one
    owner of the columns, SCHEMA_SQL, columns order."""
    path = _lagging_store_with_history(tmp_path / "profiles" / "scribe")
    with closing(sqlite3.connect(path)) as conn:
        reconcile_state_schema(conn, path.parent)
        assert not conn.in_transaction
    assert "last_read_at" in _session_columns(path)
    assert hasattr(_schema, "apply_declared_schema")
