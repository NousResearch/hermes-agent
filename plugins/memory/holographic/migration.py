"""Additive, idempotent schema migration for the cognitive memory layer.

Rules: never delete data, never rewrite existing columns, CREATE IF NOT EXISTS
only, safe to run on every MemoryStore init and on pre-existing databases.
"""

from __future__ import annotations

import sqlite3

_COGNITIVE_COLUMNS = (
    ("mem_type", "TEXT DEFAULT 'general'"),
    ("lifecycle", "TEXT DEFAULT 'active'"),
    ("salience", "REAL DEFAULT 0.5"),
    ("confidence", "REAL DEFAULT 0.5"),
    ("supersedes", "INTEGER DEFAULT NULL"),
    ("superseded_by", "INTEGER DEFAULT NULL"),
    ("source", "TEXT DEFAULT ''"),
    ("created_session", "TEXT DEFAULT ''"),
    ("updated_session", "TEXT DEFAULT ''"),
    ("norm_hash", "TEXT DEFAULT ''"),
    ("slot_key", "TEXT DEFAULT ''"),
)

_COGNITIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_links (
    link_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id    INTEGER NOT NULL REFERENCES facts(fact_id),
    linked_fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
    relation   TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fact_links_fact ON fact_links(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_links_relation ON fact_links(relation);
CREATE INDEX IF NOT EXISTS idx_facts_lifecycle ON facts(lifecycle);
CREATE INDEX IF NOT EXISTS idx_facts_mem_type ON facts(mem_type);
CREATE INDEX IF NOT EXISTS idx_facts_norm_hash ON facts(norm_hash);

CREATE TABLE IF NOT EXISTS dream_state (
    state_key  TEXT PRIMARY KEY,
    state_val  TEXT DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_VALID_RELATIONS = frozenset({
    "derived_from", "confirmed_by", "contradicted_by",
    "supersedes", "superseded_by", "related",
})


def ensure_cognitive_schema(conn: sqlite3.Connection) -> dict:
    """Apply additive migration; returns {added_columns, ok} report."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    added = []
    for name, ddl in _COGNITIVE_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {ddl}")
            added.append(name)
    conn.executescript(_COGNITIVE_SCHEMA)
    conn.commit()
    return {"added_columns": added, "ok": True}


def valid_relation(relation: str) -> bool:
    """Lineage relation allowlist."""
    return relation in _VALID_RELATIONS
