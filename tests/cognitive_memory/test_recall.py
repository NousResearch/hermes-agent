"""Tests for cognitive_memory.recall module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from cognitive_memory.recall import (
    RecallConfig,
    RecallEngine,
    composite_score,
    compute_recency,
)
from cognitive_memory.store import CognitiveStore, ScoredMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder(embed_map=None, default_embedding=None):
    """Create a mock embedder that returns predictable embeddings."""
    embedder = MagicMock()
    embedder.dimensions = 3

    def _embed_text(text):
        if embed_map and text in embed_map:
            return embed_map[text]
        if default_embedding:
            return default_embedding
        return [1.0, 0.0, 0.0]

    embedder.embed_text = MagicMock(side_effect=_embed_text)
    return embedder


@pytest.fixture
def store(tmp_path):
    """Create a temporary CognitiveStore for testing."""
    db_path = str(tmp_path / "test_recall.db")
    s = CognitiveStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return _make_embedder()


@pytest.fixture
def engine(store, embedder):
    return RecallEngine(store=store, embedder=embedder)


# ---------------------------------------------------------------------------
# compute_recency
# ---------------------------------------------------------------------------


class TestComputeRecency:
    def test_just_accessed(self):
        now = time.time()
        assert compute_recency(now, now, half_life_days=30.0) == pytest.approx(1.0)

    def test_future_access_returns_one(self):
        now = time.time()
        assert compute_recency(now + 100, now, half_life_days=30.0) == pytest.approx(1.0)

    def test_one_half_life(self):
        now = time.time()
        thirty_days_ago = now - (30 * 86400)
        result = compute_recency(thirty_days_ago, now, half_life_days=30.0)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_two_half_lives(self):
        now = time.time()
        sixty_days_ago = now - (60 * 86400)
        result = compute_recency(sixty_days_ago, now, half_life_days=30.0)
        assert result == pytest.approx(0.25, abs=0.01)

    def test_very_old(self):
        now = time.time()
        year_ago = now - (365 * 86400)
        result = compute_recency(year_ago, now, half_life_days=30.0)
        assert result < 0.001

    def test_custom_half_life(self):
        now = time.time()
        seven_days_ago = now - (7 * 86400)
        result = compute_recency(seven_days_ago, now, half_life_days=7.0)
        assert result == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_all_ones(self):
        config = RecallConfig()
        score = composite_score(1.0, 1.0, 1.0, config)
        assert score == pytest.approx(1.0)

    def test_all_zeros(self):
        config = RecallConfig()
        score = composite_score(0.0, 0.0, 0.0, config)
        assert score == pytest.approx(0.0)

    def test_similarity_dominant(self):
        config = RecallConfig(
            similarity_weight=0.8, recency_weight=0.1, importance_weight=0.1
        )
        # High similarity, low recency and importance
        score = composite_score(1.0, 0.0, 0.0, config)
        assert score == pytest.approx(0.8)

    def test_default_weights(self):
        config = RecallConfig()
        score = composite_score(0.8, 0.6, 0.4, config)
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4
        assert score == pytest.approx(expected)

    def test_custom_weights(self):
        config = RecallConfig(
            similarity_weight=0.4, recency_weight=0.4, importance_weight=0.2
        )
        score = composite_score(0.5, 0.5, 0.5, config)
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RecallConfig
# ---------------------------------------------------------------------------


class TestRecallConfig:
    def test_default_weights_sum_to_one(self):
        config = RecallConfig()
        total = config.similarity_weight + config.recency_weight + config.importance_weight
        assert total == pytest.approx(1.0)

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            RecallConfig(similarity_weight=0.5, recency_weight=0.5, importance_weight=0.5)

    def test_custom_valid_weights(self):
        config = RecallConfig(
            similarity_weight=0.6, recency_weight=0.3, importance_weight=0.1
        )
        assert config.similarity_weight == 0.6


# ---------------------------------------------------------------------------
# RecallEngine.recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_basic_recall(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("target memory", embedding=emb, importance=0.8)

        results = engine.recall("anything")
        assert len(results) == 1
        assert results[0].memory.content == "target memory"
        assert results[0].score > 0

    def test_recall_returns_sorted_by_composite_score(self, store):
        # Memory with high importance but lower similarity
        store.add_memory(
            "important old",
            embedding=[0.7, 0.7, 0.0],
            importance=1.0,
        )
        # Memory with exact similarity match but low importance
        store.add_memory(
            "similar new",
            embedding=[1.0, 0.0, 0.0],
            importance=0.1,
        )

        embedder = _make_embedder(default_embedding=[1.0, 0.0, 0.0])
        engine = RecallEngine(store=store, embedder=embedder)

        results = engine.recall("query", similarity_threshold=0.3)
        assert len(results) == 2
        # "important old" has sim~0.7 + imp=1.0, "similar new" has sim=1.0 + imp=0.1
        # With recency both ~1.0, composite: important_old ~ 0.5*0.7+0.3*1.0+0.2*1.0 = 0.85
        # similar_new ~ 0.5*1.0+0.3*1.0+0.2*0.1 = 0.82
        # So "important old" ranks first due to importance boost
        assert results[0].memory.content == "important old"
        assert results[1].memory.content == "similar new"

    def test_recall_respects_limit(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        for i in range(10):
            store.add_memory(f"memory-{i}", embedding=emb)

        results = engine.recall("query", limit=3)
        assert len(results) == 3

    def test_recall_scope_filter(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("project a", embedding=emb, scope="/projects/a")
        store.add_memory("project b", embedding=emb, scope="/projects/b")
        store.add_memory("user pref", embedding=emb, scope="/user")

        results = engine.recall("query", scope="/projects")
        contents = [r.memory.content for r in results]
        assert "project a" in contents
        assert "project b" in contents
        assert "user pref" not in contents

    def test_recall_category_filter(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("fact", embedding=emb, categories=["fact"])
        store.add_memory("preference", embedding=emb, categories=["preference"])
        store.add_memory("both", embedding=emb, categories=["fact", "preference"])

        results = engine.recall("query", categories=["preference"])
        contents = [r.memory.content for r in results]
        assert "fact" not in contents
        assert "preference" in contents
        assert "both" in contents

    def test_recall_excludes_forgotten_by_default(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        mid = store.add_memory("forgotten", embedding=emb)
        store.soft_delete(mid)
        store.add_memory("active", embedding=emb)

        results = engine.recall("query")
        contents = [r.memory.content for r in results]
        assert "forgotten" not in contents
        assert "active" in contents

    def test_recall_includes_forgotten_when_requested(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        mid = store.add_memory("forgotten", embedding=emb)
        store.soft_delete(mid)

        results = engine.recall("query", include_forgotten=True)
        assert len(results) == 1
        assert results[0].memory.content == "forgotten"

    def test_recall_records_access(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        mid = store.add_memory("test", embedding=emb)

        engine.recall("query")
        mem = store.get_memory(mid)
        assert mem.access_count == 1

    def test_recall_empty_store_returns_empty(self, store, engine):
        results = engine.recall("anything")
        assert results == []

    def test_recall_embedding_failure_returns_empty(self, store):
        embedder = MagicMock()
        embedder.embed_text = MagicMock(side_effect=RuntimeError("API down"))
        engine = RecallEngine(store=store, embedder=embedder)

        store.add_memory("test", embedding=[1.0, 0.0, 0.0])
        results = engine.recall("query")
        assert results == []

    def test_recall_match_reasons(self, store):
        now = time.time()
        emb = [1.0, 0.0, 0.0]
        mid = store.add_memory("important recent", embedding=emb, importance=0.9)

        embedder = _make_embedder(default_embedding=[1.0, 0.0, 0.0])
        engine = RecallEngine(store=store, embedder=embedder)

        results = engine.recall("query")
        assert len(results) == 1
        reasons = results[0].match_reasons
        assert "semantic" in reasons
        assert "recent" in reasons
        assert "important" in reasons

    def test_recall_custom_threshold(self, store, engine):
        store.add_memory("exact", embedding=[1.0, 0.0, 0.0])
        store.add_memory("partial", embedding=[0.7, 0.7, 0.0])

        # Very high threshold - only exact match
        results = engine.recall("query", similarity_threshold=0.95)
        assert len(results) == 1
        assert results[0].memory.content == "exact"


# ---------------------------------------------------------------------------
# RecallEngine.recall_by_embedding
# ---------------------------------------------------------------------------


class TestRecallByEmbedding:
    def test_basic_recall_by_embedding(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("target", embedding=emb)

        results = engine.recall_by_embedding(emb)
        assert len(results) == 1
        assert results[0].memory.content == "target"

    def test_skips_embed_step(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("target", embedding=emb)

        engine.recall_by_embedding(emb)
        engine._embedder.embed_text.assert_not_called()

    def test_scope_filter(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("in scope", embedding=emb, scope="/test")
        store.add_memory("out of scope", embedding=emb, scope="/other")

        results = engine.recall_by_embedding(emb, scope="/test")
        assert len(results) == 1
        assert results[0].memory.content == "in scope"

    def test_category_filter(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("tagged", embedding=emb, categories=["fact"])
        store.add_memory("untagged", embedding=emb, categories=[])

        results = engine.recall_by_embedding(emb, categories=["fact"])
        assert len(results) == 1
        assert results[0].memory.content == "tagged"


# ---------------------------------------------------------------------------
# RecallEngine.add_and_recall
# ---------------------------------------------------------------------------


class TestAddAndRecall:
    def test_adds_memory_and_finds_related(self, store):
        emb = [1.0, 0.0, 0.0]
        embedder = _make_embedder(default_embedding=emb)
        engine = RecallEngine(store=store, embedder=embedder)

        # Pre-existing memory
        store.add_memory("existing related", embedding=emb)

        result = engine.add_and_recall("new memory")
        assert result["memory_id"] > 0
        # Should find the existing memory as related
        assert len(result["related"]) >= 1
        related_contents = [sm.memory.content for sm in result["related"]]
        assert "existing related" in related_contents

    def test_excludes_self_from_related(self, store):
        emb = [1.0, 0.0, 0.0]
        embedder = _make_embedder(default_embedding=emb)
        engine = RecallEngine(store=store, embedder=embedder)

        result = engine.add_and_recall("only memory")
        # Should not include itself in related
        assert len(result["related"]) == 0

    def test_stores_with_metadata(self, store):
        emb = [1.0, 0.0, 0.0]
        embedder = _make_embedder(default_embedding=emb)
        engine = RecallEngine(store=store, embedder=embedder)

        result = engine.add_and_recall(
            "categorized",
            scope="/test",
            importance=0.9,
            categories=["fact"],
        )

        mem = store.get_memory(result["memory_id"])
        assert mem.content == "categorized"
        assert mem.scope == "/test"
        assert mem.importance == pytest.approx(0.9)
        assert mem.categories == ["fact"]

    def test_embedding_failure_stores_without_embedding(self, store):
        embedder = MagicMock()
        embedder.embed_text = MagicMock(side_effect=RuntimeError("API down"))
        engine = RecallEngine(store=store, embedder=embedder)

        result = engine.add_and_recall("no embedding")
        assert result["memory_id"] > 0
        assert result["related"] == []

        mem = store.get_memory(result["memory_id"])
        assert mem.content == "no embedding"
        assert mem.embedding is None

    def test_recall_limit(self, store):
        emb = [1.0, 0.0, 0.0]
        embedder = _make_embedder(default_embedding=emb)
        engine = RecallEngine(store=store, embedder=embedder)

        # Add many existing memories
        for i in range(10):
            store.add_memory(f"existing-{i}", embedding=emb)

        result = engine.add_and_recall("new", recall_limit=3)
        assert len(result["related"]) <= 3


# ---------------------------------------------------------------------------
# Recency boosting integration
# ---------------------------------------------------------------------------


class TestRecencyBoosting:
    def test_recent_memory_ranks_higher(self, store):
        emb = [1.0, 0.0, 0.0]

        # Add old memory with high importance
        old_id = store.add_memory("old important", embedding=emb, importance=0.9)
        # Backdate it
        conn = store._get_conn()
        old_time = time.time() - (60 * 86400)  # 60 days ago
        conn.execute(
            "UPDATE cognitive_memories SET last_accessed = ? WHERE id = ?",
            (old_time, old_id),
        )
        conn.commit()

        # Add recent memory with lower importance
        store.add_memory("recent less important", embedding=emb, importance=0.3)

        embedder = _make_embedder(default_embedding=emb)
        # Boost recency weight
        config = RecallConfig(
            similarity_weight=0.3,
            recency_weight=0.5,
            importance_weight=0.2,
        )
        engine = RecallEngine(store=store, embedder=embedder, config=config)

        results = engine.recall("query")
        assert len(results) == 2
        # Recent memory should rank higher due to recency weight
        assert results[0].memory.content == "recent less important"

    def test_importance_boosting(self, store):
        emb = [1.0, 0.0, 0.0]

        store.add_memory("low importance", embedding=emb, importance=0.1)
        store.add_memory("high importance", embedding=emb, importance=1.0)

        embedder = _make_embedder(default_embedding=emb)
        # Boost importance weight
        config = RecallConfig(
            similarity_weight=0.3,
            recency_weight=0.1,
            importance_weight=0.6,
        )
        engine = RecallEngine(store=store, embedder=embedder, config=config)

        results = engine.recall("query")
        assert len(results) == 2
        assert results[0].memory.content == "high importance"
