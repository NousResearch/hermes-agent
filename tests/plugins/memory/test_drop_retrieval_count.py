"""ADR-002: facts.retrieval_count is dropped by the v1→v2 migration.

Coverage:
  - Fresh DB has no retrieval_count column.
  - Legacy v1 DB (with the column populated) is migrated cleanly: column
    is gone, schema_version stamps v2, surviving column data is preserved.
  - search_facts and list_facts continue to function without the column.
  - The result dicts no longer carry the retrieval_count key.
  - Migration is idempotent — running it twice is a no-op.
  - FTS5 index is rebuilt — keyword search still finds pre-migration content.
  - Production code carries no remaining `retrieval_count` references
    outside the migration function itself.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from plugins.memory.holographic.store import (
    MemoryStore,
    _CURRENT_SCHEMA_VERSION,
    _migrate_v1_to_v2,
)


DIM = 256


@pytest.fixture
def tmp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _facts_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(facts)").fetchall()}


def test_fresh_db_has_no_retrieval_count(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        assert "retrieval_count" not in _facts_columns(store._conn)
    finally:
        store.close()


def test_v1_db_migrates_and_drops_column(tmp_db_path):
    """A v1 DB with retrieval_count populated migrates cleanly to v2."""
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (1);
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
            hrr_vector BLOB,
            encoding_version INTEGER NOT NULL DEFAULT 1
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
        "INSERT INTO facts (content, retrieval_count, trust_score) "
        "VALUES (?, ?, ?)",
        ("Pre-migration fact about Walker Anderson.", 17, 0.85),
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        cols = _facts_columns(store._conn)
        assert "retrieval_count" not in cols
        assert {"trust_score", "helpful_count", "content"}.issubset(cols)

        # Schema version stamped at current.
        version = store._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()[0]
        assert version == _CURRENT_SCHEMA_VERSION

        # Surviving columns preserved across the table-rebuild.
        row = store._conn.execute(
            "SELECT content, trust_score FROM facts "
            "WHERE content LIKE 'Pre-migration%'"
        ).fetchone()
        assert row[0] == "Pre-migration fact about Walker Anderson."
        assert row[1] == 0.85
    finally:
        store.close()


def test_search_facts_works_post_migration(tmp_db_path):
    """ADR-002 removes the retrieval_count UPDATE from search_facts —
    confirm the FTS5 path still returns results without raising."""
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Apollo Energy Group runs LNG logistics.",
                       category="general")
        store.add_fact("Walker Anderson manages logistics.", category="identity")
        results = store.search_facts("Apollo")
        assert len(results) >= 1
        assert "retrieval_count" not in results[0]
    finally:
        store.close()


def test_list_facts_dict_lacks_retrieval_count(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Apollo Energy Group runs LNG logistics.",
                       category="general")
        results = store.list_facts()
        assert len(results) >= 1
        assert "retrieval_count" not in results[0]
        assert "trust_score" in results[0]
    finally:
        store.close()


def test_migration_is_idempotent_after_first_run(tmp_db_path):
    """Calling _migrate_v1_to_v2 again on a v2 DB must be a no-op."""
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Idempotent test.", category="general")
        before = store._conn.execute(
            "SELECT fact_id, content, trust_score FROM facts"
        ).fetchall()
        # Second migration call against the already-migrated DB.
        _migrate_v1_to_v2(store._conn)
        after = store._conn.execute(
            "SELECT fact_id, content, trust_score FROM facts"
        ).fetchall()
        assert before == after
        assert "retrieval_count" not in _facts_columns(store._conn)
    finally:
        store.close()


def test_fts_index_survives_migration(tmp_db_path):
    """The migration drops + recreates the FTS5 virtual table; pre-migration
    content must still be findable via the rebuild."""
    # Build a v1 DB with FTS5 and a fact, then open via MemoryStore to migrate.
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (1);
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
            hrr_vector BLOB,
            encoding_version INTEGER NOT NULL DEFAULT 1
        );
        CREATE VIRTUAL TABLE facts_fts USING fts5(
            content, tags, content=facts, content_rowid=fact_id
        );
        CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts(rowid, content, tags)
                VALUES (new.fact_id, new.content, new.tags);
        END;
    """)
    conn.execute(
        "INSERT INTO facts (content) VALUES (?)",
        ("Findable Apollo Energy fact.",),
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        # FTS rebuild during migration should re-index existing rows.
        results = store.search_facts("Apollo")
        contents = [r["content"] for r in results]
        assert "Findable Apollo Energy fact." in contents
    finally:
        store.close()


def test_no_remaining_retrieval_count_references_outside_migration():
    """Production code (not the migration function or the doctor's
    absence-assertion) must not reference retrieval_count. The grep
    equivalent of ADR-002's removal contract."""
    plugin_dir = Path(__file__).resolve().parent.parent.parent.parent / (
        "plugins/memory/holographic"
    )
    # Files that legitimately reference the column name:
    #   - store.py: the _migrate_v1_to_v2 function and its docstring
    #   - doctor.py: the schema_version check that asserts absence
    allowed = {"store.py", "doctor.py"}
    offenders: list[tuple[str, int, str]] = []
    for py_file in sorted(plugin_dir.glob("*.py")):
        if py_file.name in allowed:
            continue
        for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
            if "retrieval_count" in line:
                offenders.append((py_file.name, lineno, line.strip()))
    assert not offenders, (
        "retrieval_count must not appear in retrieval/ranking code: "
        + "; ".join(f"{f}:{ln} {ln_text!r}" for f, ln, ln_text in offenders)
    )
