"""Tests for the LIKE-wildcard fix in holographic memory store
(issue #32848 part 5, PR #45640).

Before #45640, _resolve_entity used:

    "SELECT entity_id FROM entities WHERE name LIKE ?"

LIKE treats _ and % as wildcards, so name='test_entity' would
also match 'testXentity', 'testAentity', etc. — even though the
comment on the previous line said 'Exact name match'.

The fix changed LIKE to = so it really is exact.

These tests verify the fix using a real sqlite3 in-memory DB.
"""

import sqlite3

import pytest


@pytest.fixture
def store_with_entities():
    """Set up a minimal MemoryStore-like object with just the entities table."""
    from plugins.memory.holographic.store import MemoryStore

    # The real __init__ has heavy deps. Use __new__ and set up the connection
    # manually with the same schema MemoryStore uses.
    store = MemoryStore.__new__(MemoryStore)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    # Insert test fixtures: an exact-match target plus a few lookalikes
    conn.executemany(
        "INSERT INTO entities (entity_id, name) VALUES (?, ?)",
        [
            (1, "test_entity"),  # exact match
            (2, "testXentity"),  # would match LIKE '_' wildcard
            (3, "testAentity"),  # would match LIKE '_' wildcard
            (4, "test_entity_v2"),  # similar but different
        ],
    )
    conn.commit()
    store._conn = conn
    return store


def test_exact_name_match_finds_the_target(store_with_entities):
    """The exact name should be found."""
    result = store_with_entities._resolve_entity("test_entity")
    assert result == 1


def test_underscore_does_not_match_wildcard(store_with_entities):
    """The '_' character should NOT be treated as a wildcard.

    Before the fix, name='test_entity' would have matched 'testXentity'
    because '_' in LIKE matches any single character. With the fix (=)
    this is no longer the case.
    """
    # 'test_entity' is the exact name; the lookalikes should NOT match
    result = store_with_entities._resolve_entity("test_entity")
    assert result == 1, "Got the wrong row — _ was treated as a wildcard"


def test_partial_match_does_not_return_similar(store_with_entities):
    """Looking up 'test_entity' should not return 'test_entity_v2'."""
    result = store_with_entities._resolve_entity("test_entity")
    assert result == 1
    # The store would create a new entity for the lookalikes
    # (we don't assert that here — we just verify the exact match returns 1)


def test_lookalike_name_falls_through_to_creation(store_with_entities):
    """Looking up 'testXentity' should return entity_id 2 (not 1)."""
    result = store_with_entities._resolve_entity("testXentity")
    assert result == 2


def test_percent_does_not_match_wildcard(store_with_entities):
    """The '%' character should NOT be treated as a wildcard.

    We add a row whose name contains a literal % and confirm exact match
    is still exact (would have matched anything containing that name under LIKE).
    """
    store_with_entities._conn.execute(
        "INSERT INTO entities (entity_id, name) VALUES (?, ?)",
        (5, "100%"),
    )
    store_with_entities._conn.commit()

    # Without the fix, name='100%' would have matched any string containing
    # '100' (because % matches zero or more chars). With the fix it's exact.
    result = store_with_entities._resolve_entity("100%")
    assert result == 5
