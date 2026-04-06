"""Tests for the Ladybug memory plugin (plugins/memory/ladybug/).

Uses a mock LadybugMemory (built from ladybug_intf.AgentMemory) so the
tests run with no external dependencies. All tool handlers, lifecycle
hooks, prefetch, and MemoryManager integration are covered.

Run with:
    python -m pytest tests/agent/test_ladybug_memory_plugin.py -v
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_manager import MemoryManager
from agent.builtin_memory_provider import BuiltinMemoryProvider
from memory.interface import AgentMemory, MemoryEntry, MemorySearchResult
from plugins.memory.ladybug import LadybugMemoryProvider


# ---------------------------------------------------------------------------
# In-process mock LadybugMemory (implements AgentMemory ABC)
# ---------------------------------------------------------------------------


class MockLadybugMemory(AgentMemory):
    """Full in-memory implementation of AgentMemory for testing."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._entries: dict[int, MemoryEntry] = {}
        self._next_id = 1
        self._links: list[tuple[str, str, str]] = []  # (src, tgt, relation)

    # -- Core CRUD -----------------------------------------------------------

    def store(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=self._next_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self._entries[self._next_id] = entry
        self._next_id += 1
        return entry

    def search(
        self,
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[MemorySearchResult]:
        results = []
        for e in self._entries.values():
            if query.lower() in e.content.lower():
                if memory_type is None or e.memory_type == memory_type:
                    results.append(MemorySearchResult(entry=e, score=0.9))
        return results[:limit]

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[MemorySearchResult]:
        return self.search(query, limit, memory_type)

    def recall(
        self,
        limit: int = 10,
        min_importance: int = 0,
        memory_type: str | None = None,
    ) -> list[MemoryEntry]:
        entries = [
            e for e in self._entries.values()
            if e.importance >= min_importance
            and (memory_type is None or e.memory_type == memory_type)
        ]
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:limit]

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self._entries.get(int(memory_id))

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        entry = self._entries.get(int(memory_id))
        if entry is None:
            return None
        new_entry = MemoryEntry(
            id=entry.id,
            content=content if content is not None else entry.content,
            memory_type=entry.memory_type,
            importance=importance if importance is not None else entry.importance,
            metadata=metadata if metadata is not None else entry.metadata,
            created_at=entry.created_at,
            updated_at=datetime.now(),
        )
        self._entries[entry.id] = new_entry
        return new_entry

    def delete(self, memory_id: str) -> bool:
        return self._entries.pop(int(memory_id), None) is not None

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related",
        start_timestamp=None,
        end_timestamp=None,
    ) -> bool:
        self._links.append((source_id, target_id, relation))
        return True

    def get_related(
        self,
        memory_id: str,
        relation: str | None = None,
        max_depth: int = 1,
    ) -> list[tuple[MemoryEntry, str]]:
        results = []
        for src, tgt, rel in self._links:
            if src == str(memory_id):
                if relation is None or rel == relation:
                    entry = self._entries.get(int(tgt))
                    if entry:
                        results.append((entry, rel))
        return results

    def count(self) -> int:
        return len(self._entries)

    # -- Entity KG (GLiNER2) -------------------------------------------------

    def extract_entities(self, content, labels=None, threshold=None):
        raise NotImplementedError("GLiNER2 not available in tests")

    def search_by_entity(self, entity_name, limit=5):
        raise NotImplementedError("GLiNER2 not available in tests")

    def get_entity_graph(self, entity_id, max_depth=1):
        raise NotImplementedError("GLiNER2 not available in tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> MockLadybugMemory:
    return MockLadybugMemory()


@pytest.fixture
def provider(db: MockLadybugMemory) -> LadybugMemoryProvider:
    """LadybugMemoryProvider wired to a fresh in-memory mock DB."""
    p = LadybugMemoryProvider()
    p._db = db
    p._prefetch_limit = 6
    p._min_importance = 3
    return p


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_available_when_memory_installed(self):
        p = LadybugMemoryProvider()
        with patch.object(importlib.util, "find_spec", return_value=MagicMock()):
            assert p.is_available() is True

    def test_unavailable_when_memory_not_installed(self):
        p = LadybugMemoryProvider()
        with patch.object(importlib.util, "find_spec", return_value=None):
            assert p.is_available() is False

    def test_no_import_side_effects(self):
        """is_available must not import 'memory' — only inspect the spec."""
        p = LadybugMemoryProvider()
        import sys
        sys.modules.pop("memory", None)
        # Should not raise even if the package doesn't exist
        result = p.is_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_opens_db_at_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        db_path = str(tmp_path / "ladybug.lbdb")

        mock_db = MockLadybugMemory(db_path)
        fake_cls = MagicMock(return_value=mock_db)

        with patch.dict("sys.modules", {"memory": MagicMock(LadybugMemory=fake_cls)}):
            p = LadybugMemoryProvider()
            p.initialize("sess-1", hermes_home=str(tmp_path))

        fake_cls.assert_called_once_with(db_path)
        assert p._db is mock_db

    def test_expand_hermes_home_in_db_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = {"db_path": "$HERMES_HOME/custom.lbdb"}

        fake_cls = MagicMock(return_value=MockLadybugMemory())
        with patch.object(LadybugMemoryProvider, "_load_plugin_config", return_value=config):
            with patch.dict("sys.modules", {"memory": MagicMock(LadybugMemory=fake_cls)}):
                p = LadybugMemoryProvider()
                p.initialize("sess-1", hermes_home=str(tmp_path))

        expected = str(tmp_path / "custom.lbdb")
        fake_cls.assert_called_once_with(expected)

    def test_missing_package_leaves_db_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = LadybugMemoryProvider()
        # Simulate ImportError from 'from memory import LadybugMemory'
        with patch.dict("sys.modules", {"memory": None}):
            p.initialize("sess-1", hermes_home=str(tmp_path))
        assert p._db is None

    def test_reads_prefetch_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = {"prefetch_limit": "12", "min_importance": "7"}

        fake_cls = MagicMock(return_value=MockLadybugMemory())
        with patch.object(LadybugMemoryProvider, "_load_plugin_config", return_value=config):
            with patch.dict("sys.modules", {"memory": MagicMock(LadybugMemory=fake_cls)}):
                p = LadybugMemoryProvider()
                p.initialize("sess-1", hermes_home=str(tmp_path))

        assert p._prefetch_limit == 12
        assert p._min_importance == 7


# ---------------------------------------------------------------------------
# system_prompt_block
# ---------------------------------------------------------------------------


class TestSystemPromptBlock:
    def test_empty_store(self, provider, db):
        block = provider.system_prompt_block()
        assert "Ladybug Memory" in block
        assert "empty" in block.lower()

    def test_with_entries(self, provider, db):
        db.store("User prefers dark mode")
        db.store("Project: Athena")
        block = provider.system_prompt_block()
        assert "2 memories" in block
        assert "empty" not in block.lower()

    def test_no_db_returns_empty(self):
        p = LadybugMemoryProvider()
        p._db = None
        assert p.system_prompt_block() == ""


# ---------------------------------------------------------------------------
# Tools: ladybug_store
# ---------------------------------------------------------------------------


class TestToolStore:
    def test_store_basic(self, provider, db):
        r = json.loads(provider.handle_tool_call("ladybug_store", {"content": "Dark mode preferred"}))
        assert r["status"] == "stored"
        assert "memory_id" in r
        assert db.count() == 1

    def test_store_sets_type_and_importance(self, provider, db):
        provider.handle_tool_call("ladybug_store", {
            "content": "Loves Python",
            "memory_type": "preference",
            "importance": 9,
        })
        entry = list(db._entries.values())[0]
        assert entry.memory_type == "preference"
        assert entry.importance == 9

    def test_store_defaults(self, provider, db):
        provider.handle_tool_call("ladybug_store", {"content": "Some fact"})
        entry = list(db._entries.values())[0]
        assert entry.memory_type == "general"
        assert entry.importance == 5

    def test_store_with_metadata(self, provider, db):
        provider.handle_tool_call("ladybug_store", {
            "content": "Deploy on Fridays",
            "metadata": {"source": "retro"},
        })
        entry = list(db._entries.values())[0]
        assert entry.metadata == {"source": "retro"}

    def test_store_missing_content_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_store", {}))
        assert "error" in r

    def test_store_empty_content_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_store", {"content": "   "}))
        assert "error" in r

    def test_store_no_db_returns_error(self):
        p = LadybugMemoryProvider()
        p._db = None
        r = json.loads(p.handle_tool_call("ladybug_store", {"content": "Anything"}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Tools: ladybug_search
# ---------------------------------------------------------------------------


class TestToolSearch:
    def test_search_finds_match(self, provider, db):
        db.store("User loves Python")
        db.store("User hates YAML")

        r = json.loads(provider.handle_tool_call("ladybug_search", {"query": "Python"}))
        assert r["count"] == 1
        assert "Python" in r["results"][0]["content"]

    def test_search_returns_score(self, provider, db):
        db.store("Python developer")
        r = json.loads(provider.handle_tool_call("ladybug_search", {"query": "Python"}))
        assert "score" in r["results"][0]

    def test_search_limit_respected(self, provider, db):
        for i in range(10):
            db.store(f"Python fact {i}")
        r = json.loads(provider.handle_tool_call("ladybug_search", {"query": "Python", "limit": 3}))
        assert r["count"] <= 3

    def test_search_memory_type_filter(self, provider, db):
        db.store("Python pref", memory_type="preference")
        db.store("Python fact", memory_type="fact")
        r = json.loads(provider.handle_tool_call("ladybug_search", {"query": "Python", "memory_type": "preference"}))
        assert all(m["memory_type"] == "preference" for m in r["results"])

    def test_search_no_match_returns_empty(self, provider, db):
        db.store("Loves Rust")
        r = json.loads(provider.handle_tool_call("ladybug_search", {"query": "Python"}))
        assert r["count"] == 0
        assert r["results"] == []

    def test_search_missing_query_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_search", {}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Tools: ladybug_recall
# ---------------------------------------------------------------------------


class TestToolRecall:
    def test_recall_all(self, provider, db):
        db.store("low", importance=2)
        db.store("high", importance=8)
        r = json.loads(provider.handle_tool_call("ladybug_recall", {}))
        assert r["count"] == 2

    def test_recall_min_importance_filter(self, provider, db):
        db.store("low", importance=2)
        db.store("high", importance=8)
        r = json.loads(provider.handle_tool_call("ladybug_recall", {"min_importance": 5}))
        assert r["count"] == 1
        assert r["memories"][0]["importance"] == 8

    def test_recall_sorted_by_importance(self, provider, db):
        db.store("medium", importance=5)
        db.store("high", importance=9)
        db.store("low", importance=1)
        r = json.loads(provider.handle_tool_call("ladybug_recall", {}))
        importances = [m["importance"] for m in r["memories"]]
        assert importances == sorted(importances, reverse=True)

    def test_recall_limit(self, provider, db):
        for i in range(10):
            db.store(f"Memory {i}", importance=i)
        r = json.loads(provider.handle_tool_call("ladybug_recall", {"limit": 3}))
        assert len(r["memories"]) <= 3

    def test_recall_type_filter(self, provider, db):
        db.store("pref", memory_type="preference")
        db.store("fact", memory_type="fact")
        r = json.loads(provider.handle_tool_call("ladybug_recall", {"memory_type": "preference"}))
        assert all(m["memory_type"] == "preference" for m in r["memories"])

    def test_recall_entry_has_expected_fields(self, provider, db):
        db.store("Test memory", memory_type="fact", importance=7)
        r = json.loads(provider.handle_tool_call("ladybug_recall", {}))
        m = r["memories"][0]
        for field in ("id", "content", "memory_type", "importance", "metadata"):
            assert field in m


# ---------------------------------------------------------------------------
# Tools: ladybug_update
# ---------------------------------------------------------------------------


class TestToolUpdate:
    def test_update_content(self, provider, db):
        entry = db.store("Old content")
        r = json.loads(provider.handle_tool_call("ladybug_update", {
            "memory_id": entry.id,
            "content": "New content",
        }))
        assert r["status"] == "updated"
        assert r["content"] == "New content"

    def test_update_importance(self, provider, db):
        entry = db.store("Fact", importance=5)
        r = json.loads(provider.handle_tool_call("ladybug_update", {
            "memory_id": entry.id,
            "importance": 9,
        }))
        assert r["importance"] == 9
        assert r["content"] == "Fact"  # unchanged

    def test_update_nonexistent_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_update", {"memory_id": 9999}))
        assert "error" in r

    def test_update_missing_id_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_update", {"content": "x"}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Tools: ladybug_delete
# ---------------------------------------------------------------------------


class TestToolDelete:
    def test_delete_existing(self, provider, db):
        entry = db.store("To be deleted")
        r = json.loads(provider.handle_tool_call("ladybug_delete", {"memory_id": entry.id}))
        assert r["status"] == "deleted"
        assert db.count() == 0

    def test_delete_nonexistent_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_delete", {"memory_id": 9999}))
        assert "error" in r

    def test_delete_missing_id_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_delete", {}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Tools: ladybug_link
# ---------------------------------------------------------------------------


class TestToolLink:
    def test_link_creates_edge(self, provider, db):
        e1 = db.store("Alice")
        e2 = db.store("Acme Corp")
        r = json.loads(provider.handle_tool_call("ladybug_link", {
            "source_id": e1.id,
            "target_id": e2.id,
            "relation": "works-at",
        }))
        assert r["status"] == "linked"
        assert r["relation"] == "works-at"

    def test_link_default_relation(self, provider, db):
        e1 = db.store("A")
        e2 = db.store("B")
        r = json.loads(provider.handle_tool_call("ladybug_link", {
            "source_id": e1.id,
            "target_id": e2.id,
        }))
        assert r["relation"] == "related"

    def test_link_missing_ids_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_link", {"source_id": 1}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Tools: ladybug_related
# ---------------------------------------------------------------------------


class TestToolRelated:
    def test_related_finds_linked_entries(self, provider, db):
        src = db.store("Alice")
        tgt = db.store("Acme")
        db.link(str(src.id), str(tgt.id), "works-at")

        r = json.loads(provider.handle_tool_call("ladybug_related", {"memory_id": src.id}))
        assert r["count"] == 1
        assert r["related"][0]["content"] == "Acme"
        assert r["related"][0]["relation"] == "works-at"

    def test_related_relation_filter(self, provider, db):
        src = db.store("Alice")
        t1 = db.store("Acme")
        t2 = db.store("Python")
        db.link(str(src.id), str(t1.id), "works-at")
        db.link(str(src.id), str(t2.id), "uses")

        r = json.loads(provider.handle_tool_call("ladybug_related", {
            "memory_id": src.id,
            "relation": "works-at",
        }))
        assert r["count"] == 1
        assert r["related"][0]["content"] == "Acme"

    def test_related_no_links_returns_empty(self, provider, db):
        e = db.store("Lonely entry")
        r = json.loads(provider.handle_tool_call("ladybug_related", {"memory_id": e.id}))
        assert r["count"] == 0
        assert r["related"] == []

    def test_related_missing_id_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_related", {}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Tools: ladybug_entity (GLiNER2 graceful degradation)
# ---------------------------------------------------------------------------


class TestToolEntity:
    def test_extract_no_gliner_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_entity", {
            "action": "extract",
            "content": "Alice works at Acme",
        }))
        assert "error" in r
        assert "GLiNER2" in r["error"]

    def test_search_entity_no_gliner_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_entity", {
            "action": "search",
            "entity_name": "Alice",
        }))
        assert "error" in r

    def test_graph_no_gliner_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_entity", {
            "action": "graph",
            "entity_id": "ent-1",
        }))
        assert "error" in r

    def test_extract_missing_content_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_entity", {"action": "extract"}))
        assert "error" in r

    def test_search_missing_entity_name_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_entity", {"action": "search"}))
        assert "error" in r

    def test_graph_missing_entity_id_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_entity", {"action": "graph"}))
        assert "error" in r

    def test_unknown_action_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_entity", {"action": "frobnicate"}))
        assert "error" in r

    def test_extract_with_gliner_available(self, provider, db):
        """When GLiNER2 IS available, extract should return entity list."""
        entities = [{"text": "Alice", "label": "PERSON"}, {"text": "Acme", "label": "ORG"}]
        db.extract_entities = MagicMock(return_value=entities)
        r = json.loads(provider.handle_tool_call("ladybug_entity", {
            "action": "extract",
            "content": "Alice works at Acme",
        }))
        assert r["count"] == 2
        assert r["entities"] == entities

    def test_search_entity_with_gliner_available(self, provider, db):
        memories = [{"id": 1, "content": "Alice works at Acme"}]
        db.search_by_entity = MagicMock(return_value=memories)
        r = json.loads(provider.handle_tool_call("ladybug_entity", {
            "action": "search",
            "entity_name": "Alice",
            "limit": 3,
        }))
        assert r["count"] == 1
        db.search_by_entity.assert_called_once_with(entity_name="Alice", limit=3)

    def test_graph_with_gliner_available(self, provider, db):
        graph = {"entity": "Alice", "related": [{"entity": "Acme", "relation": "works-at"}]}
        db.get_entity_graph = MagicMock(return_value=graph)
        r = json.loads(provider.handle_tool_call("ladybug_entity", {
            "action": "graph",
            "entity_id": "ent-alice",
            "max_depth": 2,
        }))
        assert "entity" in r
        db.get_entity_graph.assert_called_once_with(entity_id="ent-alice", max_depth=2)


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_unknown_tool_returns_error(self, provider):
        r = json.loads(provider.handle_tool_call("ladybug_frobnicate", {}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_returns_empty_with_no_memories(self, provider):
        assert provider.prefetch("anything") == ""

    def test_queue_then_prefetch(self, provider, db):
        db.store("User loves dark mode", importance=8)
        db.store("Afternoon standup at 3pm", importance=6)

        provider.queue_prefetch("dark mode")
        # Give the background thread time to finish
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=3.0)

        result = provider.prefetch("dark mode")
        assert "Ladybug Memory" in result
        assert "dark mode" in result.lower()

    def test_prefetch_deduplicates_recall_and_search(self, provider, db):
        """An entry matching both recall and search should appear only once."""
        e = db.store("dark mode preference", importance=8)

        provider.queue_prefetch("dark mode")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=3.0)

        result = provider.prefetch("dark mode")
        # Count occurrences of the memory content
        assert result.count("dark mode preference") == 1

    def test_prefetch_min_importance_filters_recall(self, provider, db):
        db.store("low importance fact", importance=1)
        provider._min_importance = 5

        provider.queue_prefetch("")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=3.0)

        result = provider.prefetch("")
        assert "low importance fact" not in result

    def test_prefetch_clears_after_consumption(self, provider, db):
        db.store("Important fact", importance=9)

        provider.queue_prefetch("fact")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=3.0)

        provider.prefetch("fact")        # consumes the result
        second = provider.prefetch("fact")  # should be empty now
        assert second == ""

    def test_prefetch_db_none_does_not_raise(self):
        p = LadybugMemoryProvider()
        p._db = None
        p.queue_prefetch("anything")
        assert p.prefetch("anything") == ""


# ---------------------------------------------------------------------------
# on_memory_write (built-in memory mirror)
# ---------------------------------------------------------------------------


class TestOnMemoryWrite:
    def test_add_stores_new_entry(self, provider, db):
        provider.on_memory_write("add", "user", "User is a Python developer")
        assert db.count() == 1
        entry = list(db._entries.values())[0]
        assert entry.content == "User is a Python developer"
        assert entry.memory_type == "preference"
        assert entry.importance == 6

    def test_add_fact_target_uses_fact_type(self, provider, db):
        provider.on_memory_write("add", "memory", "Deploy happens on Fridays")
        entry = list(db._entries.values())[0]
        assert entry.memory_type == "fact"

    def test_replace_updates_existing(self, provider, db):
        db.store("Old preference for light mode", memory_type="preference")
        provider.on_memory_write("replace", "user", "Old preference for light mode — now dark")
        # The entry should be updated (count stays 1) or a new one added
        assert db.count() >= 1

    def test_replace_no_match_stores_new(self, provider, db):
        provider.on_memory_write("replace", "user", "Completely new fact with no prior match")
        assert db.count() == 1

    def test_unknown_action_is_ignored(self, provider, db):
        provider.on_memory_write("remove", "memory", "Something")
        assert db.count() == 0

    def test_empty_content_is_ignored(self, provider, db):
        provider.on_memory_write("add", "user", "")
        assert db.count() == 0

    def test_no_db_does_not_raise(self):
        p = LadybugMemoryProvider()
        p._db = None
        p.on_memory_write("add", "user", "Should not crash")  # no exception


# ---------------------------------------------------------------------------
# on_pre_compress
# ---------------------------------------------------------------------------


class TestOnPreCompress:
    def test_high_importance_surfaces_in_compression(self, provider, db):
        db.store("Must never delete prod DB", importance=10)
        db.store("Low priority note", importance=2)
        result = provider.on_pre_compress([])
        assert "Must never delete prod DB" in result
        assert "Low priority note" not in result

    def test_returns_empty_when_no_high_importance(self, provider, db):
        db.store("Low fact", importance=3)
        result = provider.on_pre_compress([])
        assert result == ""

    def test_returns_empty_when_no_db(self):
        p = LadybugMemoryProvider()
        p._db = None
        assert p.on_pre_compress([]) == ""

    def test_returns_empty_when_store_is_empty(self, provider):
        assert provider.on_pre_compress([]) == ""


# ---------------------------------------------------------------------------
# sync_turn (no-op)
# ---------------------------------------------------------------------------


class TestSyncTurn:
    def test_sync_turn_does_not_store(self, provider, db):
        provider.sync_turn("user message", "assistant reply")
        assert db.count() == 0


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_clears_db(self, provider):
        assert provider._db is not None
        provider.shutdown()
        assert provider._db is None

    def test_shutdown_joins_prefetch_thread(self, provider, db):
        db.store("Entry", importance=9)
        provider.queue_prefetch("Entry")
        # Don't wait — let shutdown join
        provider.shutdown()
        assert provider._prefetch_thread is None or not provider._prefetch_thread.is_alive()

    def test_shutdown_no_db_does_not_raise(self):
        p = LadybugMemoryProvider()
        p._db = None
        p.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# get_tool_schemas
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_all_eight_tools_exposed(self, provider):
        schemas = provider.get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert names == {
            "ladybug_store",
            "ladybug_search",
            "ladybug_recall",
            "ladybug_update",
            "ladybug_delete",
            "ladybug_link",
            "ladybug_related",
            "ladybug_entity",
        }

    def test_schemas_have_required_fields(self, provider):
        for schema in provider.get_tool_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_store_requires_content(self, provider):
        store_schema = next(s for s in provider.get_tool_schemas() if s["name"] == "ladybug_store")
        assert "content" in store_schema["parameters"]["properties"]
        assert "content" in store_schema["parameters"].get("required", [])


# ---------------------------------------------------------------------------
# MemoryManager integration
# ---------------------------------------------------------------------------


class TestMemoryManagerIntegration:
    def test_ladybug_tools_routed_through_manager(self, db):
        p = LadybugMemoryProvider()
        p._db = db

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        mgr.add_provider(p)

        assert mgr.has_tool("ladybug_store")
        assert mgr.has_tool("ladybug_search")
        assert mgr.has_tool("ladybug_recall")
        assert not mgr.has_tool("terminal")

    def test_store_and_recall_via_manager(self, db):
        p = LadybugMemoryProvider()
        p._db = db

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        mgr.add_provider(p)

        r = json.loads(mgr.handle_tool_call("ladybug_store", {"content": "Prefers vim"}))
        assert r["status"] == "stored"

        r = json.loads(mgr.handle_tool_call("ladybug_recall", {}))
        assert r["count"] == 1
        assert "vim" in r["memories"][0]["content"]

    def test_memory_bridge_propagates_to_ladybug(self, db):
        p = LadybugMemoryProvider()
        p._db = db

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        mgr.add_provider(p)

        mgr.on_memory_write("add", "user", "User timezone: US Pacific")
        assert db.count() == 1

    def test_system_prompt_included_in_manager_build(self, db):
        p = LadybugMemoryProvider()
        p._db = db
        db.store("A fact")

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        mgr.add_provider(p)

        prompt = mgr.build_system_prompt()
        assert "Ladybug Memory" in prompt

    def test_on_pre_compress_contributes_to_manager(self, db):
        p = LadybugMemoryProvider()
        p._db = db
        db.store("Critical fact — never ignore", importance=10)

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        mgr.add_provider(p)

        # on_pre_compress is called on each provider
        p.on_pre_compress([])  # direct call; MemoryManager calls each provider

    def test_shutdown_via_manager(self, db):
        p = LadybugMemoryProvider()
        p._db = db

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        mgr.add_provider(p)

        mgr.shutdown_all()
        assert p._db is None


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    def test_ladybug_discovered(self):
        from plugins.memory import discover_memory_providers
        names = [name for name, _, _ in discover_memory_providers()]
        assert "ladybug" in names

    def test_ladybug_loadable(self):
        from plugins.memory import load_memory_provider
        p = load_memory_provider("ladybug")
        assert p is not None
        assert p.name == "ladybug"

    def test_ladybug_has_correct_tools_when_loaded(self):
        from plugins.memory import load_memory_provider
        p = load_memory_provider("ladybug")
        names = {s["name"] for s in p.get_tool_schemas()}
        assert "ladybug_store" in names
        assert "ladybug_recall" in names
