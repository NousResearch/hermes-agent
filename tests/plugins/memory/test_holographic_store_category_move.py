"""Regression tests for the holographic memory store's bank rebuild logic.

Focuses on ``MemoryStore.update_fact`` when a fact is recategorized: the
old category's HRR memory bank must be rebuilt so it no longer bundles the
moved fact's vector, otherwise probe()/related() retrieval returns stale,
contaminated results for the old category.
"""

import pytest

# The HRR memory banks are only built when numpy is available; without it
# ``_rebuild_bank`` is a no-op and there is nothing to assert.
pytest.importorskip("numpy")

from plugins.memory.holographic.store import MemoryStore


def _bank_fact_count(store: MemoryStore, category: str):
    """Return the persisted fact_count for a category bank, or None if absent."""
    row = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = ?",
        (f"cat:{category}",),
    ).fetchone()
    return row["fact_count"] if row is not None else None


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory_store.db"))
    yield s
    s._conn.close()


def test_category_move_rebuilds_old_bank(store):
    """Moving a fact to a new category must shrink the old category's bank."""
    stay = store.add_fact("the sky appears blue at noon", category="weather")
    move = store.add_fact("rain is wet and cold", category="weather")

    # Both facts live in "weather": the bank bundles two vectors.
    assert _bank_fact_count(store, "weather") == 2

    # Recategorize one fact into a brand-new category.
    assert store.update_fact(move, category="science") is True

    # The new category's bank picks up the moved fact...
    assert _bank_fact_count(store, "science") == 1
    # ...and the old category's bank is rebuilt so it no longer counts the
    # moved fact (previously it stayed stale at 2 and contaminated recall).
    assert _bank_fact_count(store, "weather") == 1


def test_category_move_to_empty_old_bank_is_dropped(store):
    """If the moved fact was the last in its category, the old bank is removed."""
    only = store.add_fact("a lone fact in its category", category="solo")
    assert _bank_fact_count(store, "solo") == 1

    assert store.update_fact(only, category="general") is True

    # The old bank had no remaining facts, so it is deleted entirely.
    assert _bank_fact_count(store, "solo") is None
    assert _bank_fact_count(store, "general") == 1


def test_update_without_category_change_keeps_single_bank(store):
    """A content-only update must not disturb other categories' banks."""
    a = store.add_fact("first note about gardening", category="hobby")
    store.add_fact("second note about gardening", category="hobby")
    store.add_fact("an unrelated work note", category="work")

    assert store.update_fact(a, content="first note about gardening, revised") is True

    # The fact stayed in "hobby"; both category banks keep their counts.
    assert _bank_fact_count(store, "hobby") == 2
    assert _bank_fact_count(store, "work") == 1
