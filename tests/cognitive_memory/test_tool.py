"""Tests for tools/cognitive_memory_tool.py."""

import json
from unittest.mock import MagicMock

import pytest

from cognitive_memory.recall import RecallConfig, RecallEngine
from cognitive_memory.store import CognitiveStore
from tools.cognitive_memory_tool import cognitive_memory_tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_embedder(default_embedding=None):
    embedder = MagicMock()
    embedder.dimensions = 3
    embedder.embed_text = MagicMock(
        return_value=default_embedding or [1.0, 0.0, 0.0]
    )
    return embedder


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_tool.db")
    s = CognitiveStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return _make_embedder()


@pytest.fixture
def engine(store, embedder):
    return RecallEngine(store=store, embedder=embedder)


def _call(action, engine=None, store=None, **kwargs):
    result = cognitive_memory_tool(
        action=action,
        engine=engine,
        store=store,
        **kwargs,
    )
    return json.loads(result)


# ---------------------------------------------------------------------------
# No engine/store
# ---------------------------------------------------------------------------


class TestNoEngine:
    def test_returns_error_when_no_engine(self):
        result = _call("recall", engine=None, store=None, query="test")
        assert result["success"] is False
        assert "not available" in result["error"]


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_finds_memories(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("Python is great", embedding=emb, importance=0.8)

        result = _call("recall", engine=engine, store=store, query="Python")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["memories"][0]["content"] == "Python is great"

    def test_recall_empty_query(self, store, engine):
        result = _call("recall", engine=engine, store=store, query=None)
        assert result["success"] is False
        assert "required" in result["error"]

    def test_recall_no_results(self, store, engine):
        result = _call("recall", engine=engine, store=store, query="nothing here")
        assert result["success"] is True
        assert result["count"] == 0

    def test_recall_with_scope(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("in scope", embedding=emb, scope="/project")
        store.add_memory("out scope", embedding=emb, scope="/other")

        result = _call(
            "recall", engine=engine, store=store,
            query="test", scope="/project",
        )
        assert result["success"] is True
        contents = [m["content"] for m in result["memories"]]
        assert "in scope" in contents
        assert "out scope" not in contents

    def test_recall_respects_limit(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        for i in range(10):
            store.add_memory(f"mem-{i}", embedding=emb)

        result = _call(
            "recall", engine=engine, store=store,
            query="test", limit=3,
        )
        assert result["count"] <= 3

    def test_recall_includes_scores(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("test memory", embedding=emb, importance=0.8)

        result = _call("recall", engine=engine, store=store, query="test")
        mem = result["memories"][0]
        assert "score" in mem
        assert "similarity" in mem
        assert "importance" in mem
        assert "match_reasons" in mem


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    def test_store_memory(self, store, engine):
        result = _call(
            "store", engine=engine, store=store,
            content="User prefers dark mode",
        )
        assert result["success"] is True
        assert result["memory_id"] > 0
        assert len(result["categories"]) > 0

    def test_store_no_content(self, store, engine):
        result = _call("store", engine=engine, store=store, content=None)
        assert result["success"] is False
        assert "required" in result["error"]

    def test_store_with_custom_scope(self, store, engine):
        result = _call(
            "store", engine=engine, store=store,
            content="project detail", scope="/projects/myapp",
        )
        assert result["success"] is True

        mem = store.get_memory(result["memory_id"])
        assert mem.scope == "/projects/myapp"

    def test_store_with_custom_importance(self, store, engine):
        result = _call(
            "store", engine=engine, store=store,
            content="critical info", importance=0.95,
        )
        assert result["success"] is True

        mem = store.get_memory(result["memory_id"])
        assert mem.importance == pytest.approx(0.95)

    def test_store_with_custom_categories(self, store, engine):
        result = _call(
            "store", engine=engine, store=store,
            content="some info", categories=["fact", "environment"],
        )
        assert result["success"] is True

        mem = store.get_memory(result["memory_id"])
        assert "fact" in mem.categories
        assert "environment" in mem.categories

    def test_store_returns_related(self, store, engine):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("existing related", embedding=emb)

        result = _call(
            "store", engine=engine, store=store,
            content="new related content",
        )
        assert result["success"] is True
        if "related" in result:
            assert len(result["related"]) >= 1


# ---------------------------------------------------------------------------
# Forget
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_by_scope(self, store, engine):
        store.add_memory("keep", scope="/active")
        store.add_memory("forget1", scope="/old/project")
        store.add_memory("forget2", scope="/old/data")

        result = _call(
            "forget", engine=engine, store=store,
            scope="/old",
        )
        assert result["success"] is True
        assert result["forgotten_count"] == 2

    def test_forget_by_query(self, store, engine):
        store.add_memory("Python is great")
        store.add_memory("Java is fast")

        result = _call(
            "forget", engine=engine, store=store,
            query="Python",
        )
        assert result["success"] is True
        assert result["forgotten_count"] == 1

    def test_forget_no_scope_no_query(self, store, engine):
        result = _call("forget", engine=engine, store=store)
        assert result["success"] is False

    def test_forget_root_scope_requires_query(self, store, engine):
        result = _call("forget", engine=engine, store=store, scope="/")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_empty(self, store, engine):
        result = _call("status", engine=engine, store=store)
        assert result["success"] is True
        assert result["active_memories"] == 0
        assert result["total_memories"] == 0

    def test_status_with_memories(self, store, engine):
        store.add_memory("active")
        mid = store.add_memory("forgotten")
        store.soft_delete(mid)

        result = _call("status", engine=engine, store=store)
        assert result["active_memories"] == 1
        assert result["forgotten_memories"] == 1
        assert result["total_memories"] == 2


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


class TestUnknownAction:
    def test_unknown_action(self, store, engine):
        result = _call("invalid", engine=engine, store=store)
        assert result["success"] is False
        assert "Unknown action" in result["error"]


# ---------------------------------------------------------------------------
# Contradiction detection (store must find and supersede contradictions)
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    def test_store_detects_contradiction(self, store, engine):
        """When storing contradictory info, the old memory should be superseded."""
        emb = [1.0, 0.0, 0.0]
        store.add_memory(
            "The server uses port 3000",
            embedding=emb,
            importance=0.7,
            categories=["fact"],
        )

        result = _call(
            "store", engine=engine, store=store,
            content="Actually the server does not use port 3000, it should be 8080",
        )
        assert result["success"] is True
        # Should detect contradiction
        assert "contradictions" in result
        assert len(result["contradictions"]) >= 1
        assert result["contradictions"][0]["superseded"] is True

    def test_superseded_memory_is_forgotten(self, store, engine):
        """Superseded memory should be soft-deleted."""
        emb = [1.0, 0.0, 0.0]
        old_id = store.add_memory(
            "We use PostgreSQL for the database",
            embedding=emb,
            importance=0.7,
            categories=["fact"],
        )

        _call(
            "store", engine=engine, store=store,
            content="Actually we do not use PostgreSQL, we switched to MySQL",
        )

        old_mem = store.get_memory(old_id)
        assert old_mem.forgotten is True

    def test_no_contradiction_for_similar_content(self, store, engine):
        """Similar but non-contradictory content should not trigger supersede."""
        emb = [1.0, 0.0, 0.0]
        old_id = store.add_memory(
            "Python is a programming language",
            embedding=emb,
            importance=0.5,
            categories=["fact"],
        )

        result = _call(
            "store", engine=engine, store=store,
            content="Python is a great programming language",
        )
        assert result["success"] is True

        old_mem = store.get_memory(old_id)
        assert old_mem.forgotten is False


# ---------------------------------------------------------------------------
# Recall error handling (embedding API failure)
# ---------------------------------------------------------------------------


class TestRecallErrorHandling:
    def test_recall_reports_embedding_failure(self, store):
        """When embedding API fails, recall should return error, not empty results."""
        embedder = MagicMock()
        embedder.dimensions = 3
        embedder.embed_text = MagicMock(side_effect=RuntimeError("API down"))
        engine = RecallEngine(store=store, embedder=embedder)

        store.add_memory("test", embedding=[1.0, 0.0, 0.0])
        result = _call("recall", engine=engine, store=store, query="test")
        assert result["success"] is False
        assert "error" in result
        assert "API" in result["error"] or "failed" in result["error"].lower()


# ---------------------------------------------------------------------------
# Forgetting cycle trigger
# ---------------------------------------------------------------------------


class TestForgettingCycleTrigger:
    def test_store_triggers_forgetting_cycle(self, store, engine):
        """Store action should trigger forgetting cycle when due."""
        from cognitive_memory.extraction import ForgettingManager

        # Add enough memories for cycle to be eligible
        for i in range(6):
            store.add_memory(f"memory-{i}", importance=0.8)

        forgetting = ForgettingManager(store=store)
        assert forgetting._last_cycle_run is None

        result = cognitive_memory_tool(
            action="store",
            content="I prefer dark mode for all my editors",
            engine=engine,
            store=store,
            forgetting=forgetting,
        )
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert forgetting._last_cycle_run is not None

    def test_recall_also_triggers_forgetting_cycle(self, store, engine):
        """Recall action should also trigger forgetting cycle when due."""
        from cognitive_memory.extraction import ForgettingManager

        emb = [1.0, 0.0, 0.0]
        for i in range(6):
            store.add_memory(f"memory-{i}", embedding=emb, importance=0.8)

        forgetting = ForgettingManager(store=store)

        cognitive_memory_tool(
            action="recall",
            query="test",
            engine=engine,
            store=store,
            forgetting=forgetting,
        )
        assert forgetting._last_cycle_run is not None

    def test_forgetting_cycle_exempts_user_scope(self, store, engine):
        """User scope memories should not decay during forgetting cycle."""
        import time as _time
        from cognitive_memory.extraction import ForgettingManager

        # Add user memory with old access time
        mid = store.add_memory("User's name is Gani", importance=0.8, scope="/user")
        conn = store._get_conn()
        old_time = _time.time() - (90 * 86400)  # 90 days ago
        conn.execute(
            "UPDATE cognitive_memories SET last_accessed = ? WHERE id = ?",
            (old_time, mid),
        )
        conn.commit()

        # Add enough non-user memories for cycle to run
        for i in range(5):
            store.add_memory(f"project-fact-{i}", importance=0.8, scope="/project")

        forgetting = ForgettingManager(store=store)

        cognitive_memory_tool(
            action="store",
            content="The server runs on port 3000",
            engine=engine,
            store=store,
            forgetting=forgetting,
        )

        # User memory should still have original importance (exempt from decay)
        user_mem = store.get_memory(mid)
        assert user_mem.importance == pytest.approx(0.8)
        assert user_mem.forgotten is False


# ---------------------------------------------------------------------------
# Scope wildcard safety
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_content_not_stored_twice(self, store, engine):
        """Near-identical content should be detected and skipped."""
        emb = [1.0, 0.0, 0.0]
        store.add_memory(
            "User prefers dark mode",
            embedding=emb,
            importance=0.7,
            categories=["preference"],
        )

        result = _call(
            "store", engine=engine, store=store,
            content="User prefers dark mode",
        )
        assert result["success"] is True
        assert result.get("duplicate") is True
        # Should not create a new memory
        assert store.count() == 1

    def test_similar_but_different_content_stored(self, store, engine):
        """Content that is similar but meaningfully different should be stored."""
        store.add_memory(
            "User prefers dark mode",
            embedding=[1.0, 0.0, 0.0],
            importance=0.7,
        )

        # Use a different embedding for meaningfully different content
        diff_embedder = _make_embedder(default_embedding=[0.7, 0.7, 0.0])
        diff_engine = RecallEngine(store=store, embedder=diff_embedder)

        result = _call(
            "store", engine=diff_engine, store=store,
            content="User prefers light mode in the morning",
        )
        assert result["success"] is True
        assert result.get("duplicate") is not True
        assert store.count() == 2


class TestScopeWildcard:
    def test_scope_with_underscore(self, store, engine):
        """Scope containing underscore should not match as SQL wildcard."""
        emb = [1.0, 0.0, 0.0]
        store.add_memory("match", embedding=emb, scope="/user_prefs")
        store.add_memory("no match", embedding=emb, scope="/user/prefs")

        result = _call(
            "recall", engine=engine, store=store,
            query="test", scope="/user_prefs",
        )
        assert result["success"] is True
        contents = [m["content"] for m in result["memories"]]
        assert "match" in contents
        assert "no match" not in contents
