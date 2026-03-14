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
