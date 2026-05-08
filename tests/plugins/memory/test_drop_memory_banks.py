"""ADR-003: memory_banks table is dropped by the v1→v2 migration.

The table was a write-only side effect of every fact mutation with zero
readers in any retrieval/ranking code. ADR-003 drops it and removes the
seven _rebuild_bank call sites. HRR retrieval (per-fact unbind) is
unaffected.

Coverage:
  - Fresh DB has no memory_banks table.
  - Legacy v1 DB with the table populated is migrated cleanly.
  - All seven former call sites still execute without error.
  - HRR retrieval (probe path) still functions — proves bank-rebuild
    removal didn't break the per-fact unbind path.
  - Production code carries no remaining `memory_banks` or
    `_rebuild_bank` references outside the migration scope.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import (
    MemoryStore,
    _CURRENT_SCHEMA_VERSION,
)


DIM = 2048


@pytest.fixture
def tmp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _has_memory_banks(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='memory_banks'"
    ).fetchone())


def test_fresh_db_has_no_memory_banks(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        assert not _has_memory_banks(store._conn)
    finally:
        store.close()


def test_v1_db_with_memory_banks_migrates(tmp_db_path):
    """A v1 DB carrying memory_banks rows migrates to v2 with the table gone."""
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
            aliases TEXT DEFAULT ''
        );
        CREATE TABLE fact_entities (
            fact_id INTEGER, entity_id INTEGER,
            PRIMARY KEY (fact_id, entity_id)
        );
        CREATE TABLE memory_banks (
            bank_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_name TEXT NOT NULL UNIQUE,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            fact_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO memory_banks (bank_name, vector, dim, fact_count)
            VALUES ('cat:general', X'00', 2048, 0);
    """)
    conn.commit()
    conn.close()

    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        assert not _has_memory_banks(store._conn)
        version = store._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()[0]
        assert version == _CURRENT_SCHEMA_VERSION
    finally:
        store.close()


def test_add_fact_no_longer_writes_memory_banks(tmp_db_path):
    """add_fact used to call _rebuild_bank; confirm it doesn't anymore."""
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Apollo Energy Group runs LNG.", category="general")
        # Table is gone, so any attempt to query it raises. The fact survives.
        assert not _has_memory_banks(store._conn)
    finally:
        store.close()


def test_update_fact_works_post_migration(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        fid = store.add_fact("Original content about Apollo.", category="general")
        ok = store.update_fact(fid, content="Updated content about Apollo.")
        assert ok is True
        row = store._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row[0] == "Updated content about Apollo."
    finally:
        store.close()


def test_remove_fact_works_post_migration(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        fid = store.add_fact("Apollo Energy Group runs LNG.", category="general")
        ok = store.remove_fact(fid)
        assert ok is True
        gone = store._conn.execute(
            "SELECT fact_id FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert gone is None
    finally:
        store.close()


def test_rebuild_all_vectors_works_post_migration(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Apollo Energy Group runs LNG.", category="general")
        store.add_fact("Walker Anderson manages logistics.", category="identity")
        n = store.rebuild_all_vectors()
        assert n == 2
    finally:
        store.close()


def test_rename_entity_works_post_migration(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Apollo Energy Resources is a multi-division business.",
                       category="identity")
        eid = store._conn.execute(
            "SELECT entity_id FROM entities WHERE name = 'Apollo Energy Resources'"
        ).fetchone()[0]
        result = store.rename_entity(eid, "Apollo Energy Group")
        # categories_rebuilt is preserved as informational metadata describing
        # which categories were affected (no banks are actually rebuilt).
        assert "identity" in result["categories_rebuilt"]
        assert result["facts_re_encoded"] >= 1
    finally:
        store.close()


def test_merge_entities_works_post_migration(tmp_db_path):
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Apollo Energy Resources runs LNG.", category="general")
        store.add_fact("Apollo Energy Group is the canonical name.",
                       category="identity")
        src = store._conn.execute(
            "SELECT entity_id FROM entities WHERE name='Apollo Energy Resources'"
        ).fetchone()[0]
        tgt = store._conn.execute(
            "SELECT entity_id FROM entities WHERE name='Apollo Energy Group'"
        ).fetchone()[0]
        result = store.merge_entities(src, tgt)
        assert result["facts_re_pointed"] >= 1
    finally:
        store.close()


def test_canonicalize_existing_facts_works_post_migration(tmp_db_path):
    """canonicalize_existing_facts no longer does a trailing bank rebuild."""
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM)
    try:
        store.add_fact("Apollo Energy Group runs LNG.", category="general")

        def canonicalizer(text: str) -> str:
            return text.replace("Apollo Energy Group", "AEG")

        result = store.canonicalize_existing_facts(canonicalizer)
        assert result["changed"] >= 1
    finally:
        store.close()


def test_hrr_retrieval_still_works_post_migration(tmp_db_path):
    """The per-fact unbind retrieval path is unaffected by ADR-003."""
    store = MemoryStore(db_path=tmp_db_path, hrr_dim=DIM, default_trust=0.85)
    retriever = FactRetriever(
        store=store,
        temporal_decay_half_life=0,
        reinforce_on_retrieval=False,
        hrr_dim=DIM,
        hrr_weight=0.3,
    )
    try:
        store.add_fact("Apollo Energy Group runs LNG logistics.",
                       category="general")
        store.add_fact("Walker Anderson manages logistics.", category="identity")
        results = retriever.probe(entity="Apollo Energy Group", limit=5)
        assert len(results) >= 1
    finally:
        store.close()


def test_no_remaining_memory_banks_references_outside_migration():
    """Production code (not the migration or the doctor's absence
    assertion) must not reference memory_banks or _rebuild_bank."""
    plugin_dir = Path(__file__).resolve().parent.parent.parent.parent / (
        "plugins/memory/holographic"
    )
    # store.py contains the migration; doctor.py contains the absence check.
    allowed = {"store.py", "doctor.py"}
    offenders: list[tuple[str, int, str]] = []
    for py_file in sorted(plugin_dir.glob("*.py")):
        if py_file.name in allowed:
            continue
        for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
            if "memory_banks" in line or "_rebuild_bank" in line:
                offenders.append((py_file.name, lineno, line.strip()))
    assert not offenders, (
        "memory_banks/_rebuild_bank must not appear in retrieval/ranking code: "
        + "; ".join(f"{f}:{ln} {ln_text!r}" for f, ln, ln_text in offenders)
    )


def test_no_rebuild_bank_method_on_memory_store():
    """The method itself must be gone from the public surface."""
    assert not hasattr(MemoryStore, "_rebuild_bank")
