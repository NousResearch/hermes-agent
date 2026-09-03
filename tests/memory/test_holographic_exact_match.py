"""Tests for the LIKE-wildcard fix in the holographic memory store
(issue #32848 part 5, PR #45640).

Before #45640, ``_resolve_entity`` used:

    SELECT entity_id FROM entities WHERE name LIKE ?

LIKE treats ``_`` and ``%`` as wildcards, so a lookup for ``test_entity`` would
also match ``testXentity``, ``testAentity``, etc. — even though the preceding
comment said "Exact name match".

The fix changed LIKE to ``= `` (exact, case-insensitive via COLLATE NOCASE).

These tests verify the fix against a real sqlite3 in-memory DB, with the same
schema MemoryStore uses.
"""

import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore


def _make_store_with_rows(rows):
    """Build a MemoryStore-like object backed by an in-memory DB with the
    entities table schema MemoryStore actually uses."""
    store = MemoryStore.__new__(MemoryStore)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE entities (
            entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            entity_type TEXT DEFAULT 'unknown',
            aliases     TEXT DEFAULT '',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.executemany(
        "INSERT INTO entities (name) VALUES (?)",
        [(name,) for name in rows],
    )
    conn.commit()
    store._conn = conn
    return store


@pytest.fixture
def store_lookalikes_only():
    """A store whose entities table contains ONLY wildcard lookalikes for
    'test_entity' — deliberately no exact 'test_entity' row.

    Under the old LIKE query, looking up 'test_entity' would match one of the
    '_'-wildcard lookalikes ('testXentity' / 'testAentity'). Under the fixed `=`
    query it must not, so these rows are what force the regression to fire.
    """
    return _make_store_with_rows(
        ["testXentity", "testAentity", "test_entity_v2", "100%"]
    )


def test_exact_lookup_does_not_match_lookalike(store_lookalikes_only):
    """'test_entity' must NOT resolve to a wildcard lookalike.

    The fixture has no exact 'test_entity' row, so if LIKE still treated '_'
    as a wildcard this lookup would return the id of 'testXentity'/'testAentity'
    instead of falling through to creation.
    """
    # Snapshot pre-existing ids BEFORE the call: the fixed code falls through
    # to creation, so the returned id must be a brand-new row, never a lookalike.
    existing_ids = {
        r["entity_id"]
        for r in store_lookalikes_only._conn.execute("SELECT entity_id, name FROM entities")
    }
    result = store_lookalikes_only._resolve_entity("test_entity")
    assert result not in existing_ids, (
        "test_entity unexpectedly matched a wildcard lookalike"
    )
    # It fell through to creation, so an exact row now exists.
    row = store_lookalikes_only._conn.execute(
        "SELECT entity_id FROM entities WHERE name = 'test_entity' COLLATE NOCASE"
    ).fetchone()
    assert row is not None
    assert result == row["entity_id"]


def test_lookalike_name_still_resolves_to_itself(store_lookalikes_only):
    """A real lookalike name still resolves to its own row."""
    result = store_lookalikes_only._resolve_entity("testXentity")
    row = store_lookalikes_only._conn.execute(
        "SELECT entity_id FROM entities WHERE name = 'testXentity'"
    ).fetchone()
    assert result == row["entity_id"]


def test_exact_match_is_case_insensitive():
    """COLLATE NOCASE keeps lookups case-insensitive (the documented contract)."""
    store = _make_store_with_rows(["Test_Entity"])
    result = store._resolve_entity("test_entity")
    row = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = 'Test_Entity'"
    ).fetchone()
    assert result == row["entity_id"]


def test_exact_match_is_not_partial():
    """'test_entity' must not match the longer 'test_entity_v2'."""
    store = _make_store_with_rows(["test_entity", "test_entity_v2"])
    result = store._resolve_entity("test_entity")
    row = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = 'test_entity'"
    ).fetchone()
    assert result == row["entity_id"]
    assert result != store._resolve_entity("test_entity_v2")
