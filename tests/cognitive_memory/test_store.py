"""Tests for cognitive_memory.store module."""

import time
from unittest.mock import patch

import pytest

from cognitive_memory.store import (
    CognitiveStore,
    Memory,
    ScoredMemory,
    _serialize_embedding,
    _deserialize_embedding,
)


@pytest.fixture
def store(tmp_path):
    """Create a temporary CognitiveStore for testing."""
    db_path = str(tmp_path / "test_cognitive.db")
    s = CognitiveStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def sample_embedding():
    return [0.1, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip(self):
        original = [0.1, 0.2, 0.3, 0.4]
        blob = _serialize_embedding(original)
        restored = _deserialize_embedding(blob)
        for a, b in zip(original, restored):
            assert abs(a - b) < 1e-6

    def test_empty_vector(self):
        blob = _serialize_embedding([])
        assert _deserialize_embedding(blob) == []

    def test_large_vector(self):
        original = [float(i) / 1000 for i in range(1536)]
        restored = _deserialize_embedding(_serialize_embedding(original))
        assert len(restored) == 1536


# ---------------------------------------------------------------------------
# Add and Get
# ---------------------------------------------------------------------------


class TestAddAndGet:
    def test_add_returns_id(self, store, sample_embedding):
        mid = store.add_memory("test content", embedding=sample_embedding)
        assert isinstance(mid, int)
        assert mid > 0

    def test_get_memory_returns_correct_content(self, store, sample_embedding):
        mid = store.add_memory(
            "hello world",
            embedding=sample_embedding,
            scope="/test",
            importance=0.8,
            categories=["greeting"],
        )
        mem = store.get_memory(mid)
        assert mem is not None
        assert mem.content == "hello world"
        assert mem.scope == "/test"
        assert mem.importance == pytest.approx(0.8)
        assert mem.categories == ["greeting"]
        assert mem.forgotten is False

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_memory(9999) is None

    def test_add_without_embedding(self, store):
        mid = store.add_memory("no embedding")
        mem = store.get_memory(mid)
        assert mem is not None
        assert mem.embedding is None

    def test_add_defaults(self, store):
        mid = store.add_memory("defaults")
        mem = store.get_memory(mid)
        assert mem.scope == "/"
        assert mem.importance == pytest.approx(0.5)
        assert mem.categories == []
        assert mem.access_count == 0

    def test_multiple_adds_get_unique_ids(self, store):
        id1 = store.add_memory("first")
        id2 = store.add_memory("second")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Search Similar
# ---------------------------------------------------------------------------


class TestSearchSimilar:
    def test_finds_similar_embedding(self, store):
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.9, 0.1, 0.0]  # similar to emb1
        emb3 = [0.0, 0.0, 1.0]  # orthogonal

        store.add_memory("target", embedding=emb1)
        store.add_memory("similar", embedding=emb2)
        store.add_memory("different", embedding=emb3)

        results = store.search_similar(emb1, threshold=0.5)
        contents = [r.memory.content for r in results]
        assert "target" in contents
        assert "similar" in contents
        assert "different" not in contents

    def test_respects_threshold(self, store):
        emb1 = [1.0, 0.0]
        emb2 = [0.7, 0.7]  # ~0.7 similarity

        store.add_memory("a", embedding=emb1)
        store.add_memory("b", embedding=emb2)

        # High threshold should only return exact match
        results = store.search_similar(emb1, threshold=0.95)
        assert len(results) == 1
        assert results[0].memory.content == "a"

    def test_respects_limit(self, store):
        emb = [1.0, 0.0]
        for i in range(10):
            store.add_memory(f"mem-{i}", embedding=emb)

        results = store.search_similar(emb, threshold=0.5, limit=3)
        assert len(results) == 3

    def test_excludes_forgotten_by_default(self, store):
        emb = [1.0, 0.0]
        mid = store.add_memory("forgotten", embedding=emb)
        store.soft_delete(mid)

        results = store.search_similar(emb, threshold=0.5)
        assert len(results) == 0

    def test_includes_forgotten_when_requested(self, store):
        emb = [1.0, 0.0]
        mid = store.add_memory("forgotten", embedding=emb)
        store.soft_delete(mid)

        results = store.search_similar(emb, threshold=0.5, include_forgotten=True)
        assert len(results) == 1

    def test_sorted_by_similarity_descending(self, store):
        query = [1.0, 0.0, 0.0]
        store.add_memory("exact", embedding=[1.0, 0.0, 0.0])
        store.add_memory("close", embedding=[0.9, 0.1, 0.0])
        store.add_memory("far", embedding=[0.5, 0.5, 0.0])

        results = store.search_similar(query, threshold=0.3)
        assert results[0].memory.content == "exact"
        assert results[0].similarity > results[1].similarity


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_content(self, store):
        mid = store.add_memory("original")
        store.update_memory(mid, content="updated")
        mem = store.get_memory(mid)
        assert mem.content == "updated"

    def test_update_importance(self, store):
        mid = store.add_memory("test", importance=0.5)
        store.update_memory(mid, importance=0.9)
        mem = store.get_memory(mid)
        assert mem.importance == pytest.approx(0.9)

    def test_update_nonexistent_returns_false(self, store):
        assert store.update_memory(9999, content="x") is False

    def test_update_refreshes_updated_at(self, store):
        mid = store.add_memory("test")
        mem1 = store.get_memory(mid)
        time.sleep(0.01)
        store.update_memory(mid, content="changed")
        mem2 = store.get_memory(mid)
        assert mem2.updated_at > mem1.updated_at


# ---------------------------------------------------------------------------
# Soft Delete
# ---------------------------------------------------------------------------


class TestSoftDelete:
    def test_soft_delete_marks_forgotten(self, store):
        mid = store.add_memory("forget me")
        store.soft_delete(mid)
        mem = store.get_memory(mid)
        assert mem.forgotten is True

    def test_soft_delete_nonexistent_returns_false(self, store):
        assert store.soft_delete(9999) is False

    def test_soft_delete_by_scope(self, store):
        store.add_memory("keep", scope="/active")
        store.add_memory("forget1", scope="/old/project")
        store.add_memory("forget2", scope="/old/data")

        count = store.soft_delete_by_scope("/old")
        assert count == 2

        active = store.get_all_active()
        assert len(active) == 1
        assert active[0].content == "keep"

    def test_soft_delete_by_scope_with_age(self, store):
        mid = store.add_memory("old", scope="/test")
        # Backdate the created_at
        conn = store._get_conn()
        old_time = time.time() - (60 * 86400)  # 60 days ago
        conn.execute(
            "UPDATE cognitive_memories SET created_at = ? WHERE id = ?",
            (old_time, mid),
        )
        conn.commit()

        store.add_memory("new", scope="/test")

        count = store.soft_delete_by_scope("/test", older_than_days=30)
        assert count == 1

        active = store.get_all_active("/test")
        assert len(active) == 1
        assert active[0].content == "new"


# ---------------------------------------------------------------------------
# Get All Active
# ---------------------------------------------------------------------------


class TestGetAllActive:
    def test_returns_only_active(self, store):
        store.add_memory("active")
        mid = store.add_memory("forgotten")
        store.soft_delete(mid)

        active = store.get_all_active()
        assert len(active) == 1
        assert active[0].content == "active"

    def test_filter_by_scope(self, store):
        store.add_memory("project a", scope="/projects/a")
        store.add_memory("project b", scope="/projects/b")
        store.add_memory("user pref", scope="/user")

        results = store.get_all_active("/projects")
        assert len(results) == 2

    def test_ordered_by_importance(self, store):
        store.add_memory("low", importance=0.1)
        store.add_memory("high", importance=0.9)
        store.add_memory("mid", importance=0.5)

        active = store.get_all_active()
        assert active[0].content == "high"
        assert active[1].content == "mid"
        assert active[2].content == "low"


# ---------------------------------------------------------------------------
# Record Access
# ---------------------------------------------------------------------------


class TestRecordAccess:
    def test_increments_access_count(self, store):
        mid = store.add_memory("test")
        store.record_access(mid)
        store.record_access(mid)
        mem = store.get_memory(mid)
        assert mem.access_count == 2

    def test_updates_last_accessed(self, store):
        mid = store.add_memory("test")
        mem1 = store.get_memory(mid)
        time.sleep(0.01)
        store.record_access(mid)
        mem2 = store.get_memory(mid)
        assert mem2.last_accessed > mem1.last_accessed


# ---------------------------------------------------------------------------
# Decay + Prune
# ---------------------------------------------------------------------------


class TestDecayAndPrune:
    def test_decay_reduces_importance(self, store):
        mid = store.add_memory("old", importance=1.0)
        # Backdate last_accessed to 30 days ago
        conn = store._get_conn()
        old_time = time.time() - (30 * 86400)
        conn.execute(
            "UPDATE cognitive_memories SET last_accessed = ? WHERE id = ?",
            (old_time, mid),
        )
        conn.commit()

        updated = store.decay_importance(half_life_days=30)
        assert updated == 1

        mem = store.get_memory(mid)
        assert mem.importance == pytest.approx(0.5, abs=0.05)

    def test_decay_respects_exempt_scopes(self, store):
        mid = store.add_memory("user pref", importance=1.0, scope="/user")
        conn = store._get_conn()
        old_time = time.time() - (60 * 86400)
        conn.execute(
            "UPDATE cognitive_memories SET last_accessed = ? WHERE id = ?",
            (old_time, mid),
        )
        conn.commit()

        updated = store.decay_importance(exempt_scopes=["/user"])
        assert updated == 0

        mem = store.get_memory(mid)
        assert mem.importance == pytest.approx(1.0)

    def test_prune_removes_low_importance(self, store):
        store.add_memory("important", importance=0.8)
        store.add_memory("weak", importance=0.03)
        store.add_memory("tiny", importance=0.01)

        pruned = store.prune(threshold=0.05)
        assert pruned == 2

        active = store.get_all_active()
        assert len(active) == 1
        assert active[0].content == "important"


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_active(self, store):
        store.add_memory("a")
        mid = store.add_memory("b")
        store.soft_delete(mid)

        assert store.count() == 1

    def test_count_including_forgotten(self, store):
        store.add_memory("a")
        mid = store.add_memory("b")
        store.soft_delete(mid)

        assert store.count(include_forgotten=True) == 2

    def test_count_empty(self, store):
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Scope wildcard safety
# ---------------------------------------------------------------------------


class TestScopeEscape:
    def test_underscore_in_scope_not_wildcard(self, store):
        """Underscore in scope should be literal, not SQL LIKE wildcard."""
        store.add_memory("match", scope="/user_prefs")
        store.add_memory("no match", scope="/user/prefs")

        results = store.get_all_active("/user_prefs")
        assert len(results) == 1
        assert results[0].content == "match"

    def test_percent_in_scope_not_wildcard(self, store):
        """Percent in scope should be literal, not SQL LIKE wildcard."""
        store.add_memory("match", scope="/100%done")
        store.add_memory("no match", scope="/100other")

        results = store.get_all_active("/100%done")
        assert len(results) == 1
        assert results[0].content == "match"

    def test_soft_delete_by_scope_underscore(self, store):
        """soft_delete_by_scope should also escape wildcards."""
        store.add_memory("keep", scope="/user/data")
        store.add_memory("delete", scope="/user_data")

        count = store.soft_delete_by_scope("/user_data")
        assert count == 1

        active = store.get_all_active()
        assert len(active) == 1
        assert active[0].content == "keep"
