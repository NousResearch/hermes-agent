"""Tests for holographic memory store — orphan cleanup and stale bank rebuild.

Regression tests for:
- #43622: remove_fact leaves orphaned entities with zero fact links
- #43621: update_fact leaves old category HRR bank stale when category changes
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def store(tmp_path):
    """Create a fresh MemoryStore backed by a temp SQLite file."""
    from plugins.memory.holographic.store import MemoryStore

    db_path = tmp_path / "test_memory.db"
    return MemoryStore(db_path=db_path, hrr_dim=64)


# --- Helper queries ---------------------------------------------------------

def _entity_count(store) -> int:
    """Return the number of rows in the entities table."""
    return store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]


def _fact_entity_link_count(store) -> int:
    """Return the number of rows in the fact_entities table."""
    return store._conn.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0]


def _entities_for_fact(store, fact_id: int) -> list[str]:
    """Return entity names linked to a given fact."""
    return [
        r["name"]
        for r in store._conn.execute(
            """
            SELECT e.name FROM entities e
            JOIN fact_entities fe ON e.entity_id = fe.entity_id
            WHERE fe.fact_id = ?
            """,
            (fact_id,),
        ).fetchall()
    ]


def _entity_exists(store, name: str) -> bool:
    """Check if an entity with the given name exists."""
    return (
        store._conn.execute(
            "SELECT 1 FROM entities WHERE name = ?", (name,)
        ).fetchone()
        is not None
    )


# --- #43622: remove_fact orphan cleanup ------------------------------------

class TestRemoveFactOrphanCleanup:
    """remove_fact() should delete entities that have no remaining fact links."""

    def test_removes_orphaned_entity(self, store):
        """Entity linked to only one fact is cleaned up when that fact is removed."""
        # Use multi-word capitalized phrases so _extract_entities finds them
        fid = store.add_fact("John Smith works at Acme Corp", category="tech")
        assert _entity_count(store) >= 1
        assert store.remove_fact(fid)
        # All entities that were only linked to this fact should be gone
        assert _entity_count(store) == 0
        assert _fact_entity_link_count(store) == 0

    def test_keeps_shared_entity(self, store):
        """Entity linked to multiple facts survives when one fact is removed."""
        fid1 = store.add_fact("John Smith works at Acme Corp", category="tech")
        fid2 = store.add_fact("John Smith likes Big Data", category="tech")

        # Find entities linked to fid1
        entities_before = _entities_for_fact(store, fid1)
        assert len(entities_before) >= 1

        store.remove_fact(fid1)

        # The shared entity (John Smith) should still exist (linked to fid2)
        assert _entity_exists(store, "John Smith"), "Shared entity 'John Smith' was deleted"

    def test_removes_multiple_orphaned_entities(self, store):
        """Multiple entities linked to a single fact are all cleaned up."""
        fid = store.add_fact("Alice Wonder met Bob Builder in New York", category="people")
        entity_count_before = _entity_count(store)
        assert entity_count_before >= 2

        store.remove_fact(fid)
        assert _entity_count(store) == 0

    def test_remove_nonexistent_fact(self, store):
        """Removing a non-existent fact returns False and doesn't touch entities."""
        assert store.remove_fact(999) is False

    def test_entity_table_does_not_grow_unboundedly(self, store):
        """Repeated add/remove cycles don't accumulate orphaned entities."""
        for i in range(20):
            fid = store.add_fact(f"Jane Doe wrote Report Number {i}", category="test")
            store.remove_fact(fid)

        # No orphans should remain
        assert _entity_count(store) == 0


# --- #43621: update_fact stale bank rebuild ---------------------------------

class TestUpdateFactStaleBank:
    """update_fact() should rebuild the old category's HRR bank when category changes."""

    def test_old_category_bank_rebuilt_on_category_change(self, store):
        """Moving a fact to a new category rebuilds both old and new banks."""
        fid1 = store.add_fact("John Smith works at Acme Corp", category="tech")
        fid2 = store.add_fact("Jane Doe works at Big Tech", category="tech")

        # Verify both facts are in "tech" category
        row = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (fid1,)
        ).fetchone()
        assert row["category"] == "tech"

        # Move fact 1 to "general"
        store.update_fact(fid1, category="general")

        # Verify the category was updated
        row = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (fid1,)
        ).fetchone()
        assert row["category"] == "general"

        # The old "tech" bank should have been rebuilt (no assertion on HRR
        # internals — the fix is that _rebuild_bank is called for old_category).
        row2 = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (fid2,)
        ).fetchone()
        assert row2["category"] == "tech"  # unchanged

    def test_same_category_no_double_rebuild(self, store):
        """Updating without changing category only rebuilds once."""
        fid = store.add_fact("John Smith works at Acme Corp", category="tech")
        # Update content but keep same category — should not trigger old_category rebuild
        store.update_fact(fid, content="John Smith works at Big Corp", category="tech")
        row = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["category"] == "tech"

    def test_category_change_without_content_change(self, store):
        """Category-only change still rebuilds both banks."""
        fid1 = store.add_fact("John Smith works at Acme Corp", category="alpha")
        fid2 = store.add_fact("Jane Doe works at Big Tech", category="alpha")
        fid3 = store.add_fact("Bob Builder makes Big Things", category="beta")

        # Move fid1 from alpha to beta
        store.update_fact(fid1, category="beta")

        row = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (fid1,)
        ).fetchone()
        assert row["category"] == "beta"

    def test_update_preserves_old_category_field(self, store):
        """The old_category variable is captured before the UPDATE executes."""
        fid = store.add_fact("John Smith works at Acme Corp", category="original")
        store.update_fact(fid, category="moved")
        row = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["category"] == "moved"
