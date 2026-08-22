"""Tests for holographic memory on_memory_write mirroring.

Verifies that add/replace/remove actions on the built-in memory tool are
correctly mirrored to the holographic fact store, so the two stay in sync.
"""
import importlib
import sys
from pathlib import Path

import pytest

# Ensure the hermes-agent source is importable
SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _make_store(tmp_path):
    """Create a real holographic MemoryStore against a temp SQLite DB."""
    from plugins.memory.holographic.store import MemoryStore
    db_path = str(tmp_path / "test_memory.db")
    return MemoryStore(db_path)


def _make_plugin(tmp_path):
    """Create a HolographicMemoryProvider with a real store, no config needed."""
    from plugins.memory.holographic import HolographicMemoryProvider
    plugin = HolographicMemoryProvider.__new__(HolographicMemoryProvider)
    plugin._store = _make_store(tmp_path)
    plugin._retriever = None
    plugin._config = {}
    plugin._min_trust = 0.3
    return plugin


class TestOnMemoryWriteAdd:
    def test_add_mirrors_to_fact_store(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        plugin.on_memory_write("add", "memory", "Test fact about the farm")
        results = plugin._store.search_facts("farm")
        assert len(results) == 1
        assert "Test fact about the farm" in results[0]["content"]

    def test_add_user_target_uses_user_pref_category(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        plugin.on_memory_write("add", "user", "Kevin prefers concise responses")
        results = plugin._store.search_facts("concise", min_trust=0.0)
        assert len(results) == 1
        assert results[0]["category"] == "user_pref"

    def test_add_empty_content_does_nothing(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        plugin.on_memory_write("add", "memory", "")
        facts = plugin._store.list_facts()
        assert len(facts) == 0


class TestOnMemoryWriteReplace:
    def test_replace_updates_matching_fact(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        # Add a fact first
        plugin.on_memory_write("add", "memory", "Old entry about providers")
        # Replace it
        plugin.on_memory_write(
            "replace", "memory", "New entry about providers v2",
            metadata={"old_text": "Old entry about providers"},
        )
        # Old content should be updated
        results = plugin._store.find_facts_by_content("New entry about providers")
        assert len(results) == 1
        # Old content should no longer exist
        old_results = plugin._store.find_facts_by_content("Old entry about")
        assert len(old_results) == 0

    def test_replace_without_old_text_falls_back_to_add(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        plugin.on_memory_write("replace", "memory", "Orphan replace content")
        results = plugin._store.find_facts_by_content("Orphan replace")
        assert len(results) == 1


class TestOnMemoryWriteRemove:
    def test_remove_deletes_matching_fact(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        # Add a fact first
        plugin.on_memory_write("add", "memory", "Stale fact to remove")
        assert len(plugin._store.list_facts()) == 1
        # Remove it
        plugin.on_memory_write(
            "remove", "memory", "",
            metadata={"old_text": "Stale fact to remove"},
        )
        assert len(plugin._store.list_facts()) == 0

    def test_remove_no_match_is_silent(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        plugin.on_memory_write(
            "remove", "memory", "",
            metadata={"old_text": "nonexistent"},
        )
        assert len(plugin._store.list_facts()) == 0


class TestFindFactsByContent:
    def test_finds_by_substring_case_insensitive(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_fact("The Windmill ACs are in the living room")
        results = store.find_facts_by_content("windmill")
        assert len(results) == 1
        assert "Windmill" in results[0]["content"]

    def test_no_match_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_fact("Some fact")
        results = store.find_facts_by_content("totally different")
        assert len(results) == 0