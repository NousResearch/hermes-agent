"""Schema migration tests for memory_store.db.

Covers:
- Fresh DB has schema_version row at current and encoding_version column.
- Pre-existing DB without schema_version / encoding_version is migrated
  cleanly when MemoryStore opens it; pre-existing rows get encoding_version=1.
- Re-opening an already-current DB is a no-op (idempotent).
- New facts inserted post-migration get encoding_version=current.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from plugins.memory.holographic.store import (
    MemoryStore,
    _CURRENT_ENCODING_VERSION,
    _CURRENT_SCHEMA_VERSION,
)


DIM = 256  # tests don't need full dim


@pytest.fixture
def tmp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def test_fresh_db_has_schema_version(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        row = store._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == _CURRENT_SCHEMA_VERSION
    finally:
        store.close()


def test_fresh_db_has_encoding_version_column(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        cols = {r[1] for r in store._conn.execute(
            "PRAGMA table_info(facts)"
        ).fetchall()}
        assert "encoding_version" in cols
    finally:
        store.close()


def test_new_facts_get_current_encoding_version(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        fid = store.add_fact("Walker Anderson is a person.", category="identity")
        row = store._conn.execute(
            "SELECT encoding_version FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row[0] == _CURRENT_ENCODING_VERSION
    finally:
        store.close()


def test_legacy_db_without_schema_version_migrates(tmp_db_path):
    """Simulate a pre-migration DB: facts table exists but no schema_version
    table and no encoding_version column. Opening it via MemoryStore should
    add both, with existing rows defaulted to encoding_version=1.
    """
    # Build a legacy-shaped DB by hand.
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript("""
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT DEFAULT 'unknown',
            aliases TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE fact_entities (
            fact_id INTEGER REFERENCES facts(fact_id),
            entity_id INTEGER REFERENCES entities(entity_id),
            PRIMARY KEY (fact_id, entity_id)
        );
    """)
    conn.execute(
        "INSERT INTO facts (content) VALUES (?)",
        ("Pre-migration fact about Walker Anderson.",),
    )
    conn.commit()
    conn.close()

    # Open via MemoryStore — migration should run.
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        # schema_version exists and matches.
        row = store._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == _CURRENT_SCHEMA_VERSION

        # encoding_version column exists.
        cols = {r[1] for r in store._conn.execute(
            "PRAGMA table_info(facts)"
        ).fetchall()}
        assert "encoding_version" in cols

        # ADR-002: retrieval_count column dropped by v1→v2 migration.
        assert "retrieval_count" not in cols

        # Pre-existing row defaulted to 1.
        row = store._conn.execute(
            "SELECT encoding_version FROM facts WHERE content LIKE 'Pre-migration%'"
        ).fetchone()
        assert row[0] == 1
    finally:
        store.close()


def test_reopen_is_idempotent(tmp_db_path):
    """Opening, closing, and re-opening the same DB does not change schema."""
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    fid = store.add_fact("Idempotent test fact.", category="general")
    snapshot_pragma = store._conn.execute(
        "PRAGMA table_info(facts)"
    ).fetchall()
    snapshot_version = store._conn.execute(
        "SELECT version FROM schema_version LIMIT 1"
    ).fetchone()[0]
    store.close()

    store2 = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        assert store2._conn.execute(
            "PRAGMA table_info(facts)"
        ).fetchall() == snapshot_pragma
        assert store2._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()[0] == snapshot_version
        # Fact survived.
        row = store2._conn.execute(
            "SELECT encoding_version FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row[0] == _CURRENT_ENCODING_VERSION
    finally:
        store2.close()
