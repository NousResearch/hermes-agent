"""Regression tests for the holographic memory store's entity resolution.

Focus: ``MemoryStore._resolve_entity`` must match entity names and aliases
*literally*. SQLite ``LIKE`` treats ``_`` and ``%`` as wildcards, so feeding a
raw name in as a LIKE pattern silently merges distinct entities (e.g. the
quoted identifier ``user_id`` matching a pre-existing ``user1id``), which then
corrupts fact links and the HRR vectors derived from them.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory_store.db"))
    try:
        yield s
    finally:
        s._conn.close()


def _make_entity(store, name, aliases=""):
    cur = store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES (?, ?)", (name, aliases)
    )
    store._conn.commit()
    return int(cur.lastrowid)


def test_underscore_name_does_not_match_distinct_entity(store):
    """``user_id`` must not resolve to a pre-existing ``user1id``.

    With ``WHERE name LIKE ?`` the underscore acted as a single-char wildcard
    and collapsed the two into one entity_id.
    """
    existing = _make_entity(store, "user1id")
    resolved = store._resolve_entity("user_id")

    assert resolved != existing
    # A genuinely new, distinct entity row should have been created.
    name = store._conn.execute(
        "SELECT name FROM entities WHERE entity_id = ?", (resolved,)
    ).fetchone()["name"]
    assert name == "user_id"


def test_percent_name_does_not_wildcard_match(store):
    """A name containing ``%`` must be matched literally, not as a wildcard."""
    other = _make_entity(store, "100 dollars")
    resolved = store._resolve_entity("100%")

    assert resolved != other


def test_exact_name_match_is_case_insensitive(store):
    """The literal match must still be case-insensitive, as documented."""
    existing = _make_entity(store, "Guido")

    assert store._resolve_entity("guido") == existing
    assert store._resolve_entity("GUIDO") == existing


def test_alias_match_is_literal_and_case_insensitive(store):
    """Alias membership must match the whole comma-delimited token literally."""
    entity = _make_entity(store, "Python", aliases="cpython,user1id")

    # Exact alias resolves to the owning entity (case-insensitive).
    assert store._resolve_entity("cpython") == entity
    assert store._resolve_entity("CPython") == entity
    assert store._resolve_entity("user1id") == entity

    # A wildcard-shaped near-miss must NOT match the alias.
    resolved = store._resolve_entity("user_id")
    assert resolved != entity


def test_distinct_quoted_identifiers_get_distinct_entities_via_add_fact(store):
    """End-to-end: two quoted identifiers must not collapse during add_fact."""
    store.add_fact('the column "user1id" is the primary key')
    store.add_fact('the column "user_id" is deprecated')

    names = {
        row["name"]
        for row in store._conn.execute("SELECT name FROM entities").fetchall()
    }
    assert {"user1id", "user_id"} <= names

    ids = [
        store._resolve_entity("user1id"),
        store._resolve_entity("user_id"),
    ]
    assert ids[0] != ids[1]
