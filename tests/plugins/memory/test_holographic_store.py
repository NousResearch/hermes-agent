"""Tests for the holographic memory provider's SQLite fact store.

Focus: entity resolution must match names/aliases literally. Entity names are
extracted from free text (including quoted spans) and can legitimately contain
the SQL LIKE wildcards ``%`` and ``_``; resolution must not treat those as
patterns and silently merge facts about distinct real-world entities.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore, _escape_like


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory_store.db"))
    yield s
    s._conn.close()


def test_escape_like_escapes_wildcards_and_escape_char():
    assert _escape_like("Pro%llo") == r"Pro\%llo"
    assert _escape_like("C_t") == r"C\_t"
    # Backslash is escaped first so the escape char is not double-handled.
    assert _escape_like(r"a\b") == r"a\\b"


def test_resolve_entity_percent_wildcard_does_not_merge(store):
    """A name containing ``%`` must not collide with an unrelated entity."""
    apollo = store._resolve_entity("Project Apollo")
    # Before the fix this resolved via ``name LIKE 'Pro%llo'`` and returned the
    # existing "Project Apollo" id; now it must create a distinct entity.
    other = store._resolve_entity("Pro%llo")
    assert other != apollo


def test_resolve_entity_underscore_wildcard_does_not_merge(store):
    """``_`` matches any single char in LIKE — it must be treated literally."""
    cat = store._resolve_entity("Cat")
    other = store._resolve_entity("C_t")
    assert other != cat


def test_resolve_entity_exact_match_is_case_insensitive(store):
    """Exact resolution stays case-insensitive (documented behavior)."""
    first = store._resolve_entity("Project Apollo")
    again = store._resolve_entity("project apollo")
    assert again == first


def test_resolve_entity_alias_wildcard_does_not_merge(store):
    """Wildcards in an alias query must not match an unrelated alias literally."""
    store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES (?, ?)",
        ("Big Cat", "Tiger,Lion"),
    )
    store._conn.commit()
    big_cat = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = 'Big Cat'"
    ).fetchone()["entity_id"]

    # Exact alias still resolves to the owning entity.
    assert store._resolve_entity("Tiger") == big_cat
    # ``Ti%er`` must NOT match the "Tiger" alias; a new entity is created.
    assert store._resolve_entity("Ti%er") != big_cat


def test_add_fact_keeps_wildcard_named_entities_distinct(store):
    """End-to-end: quoted entity names with wildcards stay separate."""
    store.add_fact('Notes on "Project Apollo" and its history.')
    store.add_fact('Unrelated notes on "Pro%llo" the codename.')

    rows = store._conn.execute(
        "SELECT name FROM entities WHERE name IN ('Project Apollo', 'Pro%llo')"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert names == {"Project Apollo", "Pro%llo"}
