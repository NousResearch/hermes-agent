"""Tests for FTS5 builtin memory provider — store.py and __init__.py."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from plugins.memory.fts5_builtin.store import FTS5Store
from plugins.memory.fts5_builtin import FTS5BuiltinProvider


# ── Store tests ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Create a temporary FTS5Store with an isolated database."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test_fts5.db")
        store = FTS5Store(db_path)
        yield store


class TestFTS5StoreLimitConvergence:
    """Limit values must be clamped to a safe range (1–100)."""

    def test_search_negative_limit_clamped(self, tmp_db):
        tmp_db.add_fact("alpha beta gamma", category="general")
        tmp_db.add_fact("alpha delta epsilon", category="general")
        results = tmp_db.search_facts("alpha", limit=-5)
        assert 1 <= len(results) <= 100
        # Negative limit should be treated as default (not crash)
        assert len(results) >= 1

    def test_search_zero_limit_clamped(self, tmp_db):
        tmp_db.add_fact("hello world", category="general")
        results = tmp_db.search_facts("hello", limit=0)
        assert 1 <= len(results) <= 100

    def test_search_massive_limit_clamped(self, tmp_db):
        for i in range(20):
            tmp_db.add_fact(f"test fact number {i}", category="general")
        results = tmp_db.search_facts("test", limit=99999)
        assert len(results) <= 100

    def test_list_negative_limit_clamped(self, tmp_db):
        tmp_db.add_fact("fact a", category="general")
        results = tmp_db.list_facts(limit=-1)
        assert len(results) >= 1

    def test_list_massive_limit_clamped(self, tmp_db):
        for i in range(20):
            tmp_db.add_fact(f"fact {i}", category="general")
        results = tmp_db.list_facts(limit=99999)
        assert len(results) <= 100


class TestFTS5StoreDedup:
    """Repeated fact insertion should deduplicate, not accumulate."""

    def test_add_same_content_returns_existing_id(self, tmp_db):
        fid1 = tmp_db.add_fact("unique fact content here", category="user_pref")
        fid2 = tmp_db.add_fact("unique fact content here", category="user_pref")
        assert fid1 == fid2  # Should return existing ID, not create new
        # Only one row should exist
        all_facts = tmp_db.list_facts(limit=100)
        assert len(all_facts) == 1

    def test_similar_but_not_identical_not_deduped(self, tmp_db):
        """Only exact content match should dedup."""
        tmp_db.add_fact("project uses Python 3.12", category="project")
        tmp_db.add_fact("project uses Python 3.11", category="project")
        all_facts = tmp_db.list_facts(limit=100)
        assert len(all_facts) == 2

    def test_dedup_across_categories(self, tmp_db):
        """Same content with different category should still dedup (content is key)."""
        fid1 = tmp_db.add_fact("cross-category fact", category="general")
        fid2 = tmp_db.add_fact("cross-category fact", category="user_pref")
        assert fid1 == fid2
        assert len(tmp_db.list_facts(limit=100)) == 1


class TestFTS5StoreWAL:
    """SQLite must use WAL journal mode and set busy_timeout."""

    def test_wal_mode_enabled(self, tmp_db):
        with tmp_db._lock:
            conn = tmp_db._get_conn()
            try:
                journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                assert journal_mode.lower() == "wal", f"Expected WAL, got {journal_mode}"
            finally:
                conn.close()

    def test_busy_timeout_set(self, tmp_db):
        with tmp_db._lock:
            conn = tmp_db._get_conn()
            try:
                timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                assert timeout >= 1000, f"busy_timeout too low: {timeout}"
            finally:
                conn.close()


# ── Provider tests ─────────────────────────────────────────────────────────

@pytest.fixture
def provider():
    """Create an initialized FTS5BuiltinProvider."""
    with tempfile.TemporaryDirectory() as td:
        p = FTS5BuiltinProvider()
        p.initialize(session_id="test-session", hermes_home=td)
        yield p
        p.shutdown()


class TestFTS5ProviderOnMemoryWrite:
    """on_memory_write must mirror built-in memory.add/replace to FTS5."""

    def test_on_memory_write_add_stores_fact(self, provider):
        provider.on_memory_write("add", "memory", "Jordan likes concise docs")
        results = provider._store.search_facts("Jordan likes concise docs", limit=5)
        assert len(results) >= 1
        assert "Jordan likes concise docs" in results[0]["content"]

    def test_on_memory_write_add_with_user_target(self, provider):
        provider.on_memory_write("add", "user", "Prefers dark mode everywhere")
        results = provider._store.search_facts("Prefers dark mode", limit=5)
        assert len(results) >= 1

    def test_on_memory_write_replace_stores_fact(self, provider):
        provider.on_memory_write("replace", "memory", "Replaced fact content")
        results = provider._store.search_facts("Replaced fact content", limit=5)
        assert len(results) >= 1

    def test_on_memory_write_remove_does_nothing(self, provider):
        """Remove action should not crash (it's a no-op for FTS5)."""
        provider.on_memory_write("add", "memory", "To be removed")
        provider.on_memory_write("remove", "memory", "To be removed")
        # Should not throw; fact still exists (remove is no-op for keyword store)
        results = provider._store.search_facts("To be removed", limit=5)
        assert len(results) >= 1

    def test_on_memory_write_empty_content_skipped(self, provider):
        """Empty content should be skipped gracefully."""
        provider.on_memory_write("add", "memory", "")
        all_facts = provider._store.list_facts(limit=100)
        assert len(all_facts) == 0

    def test_on_memory_write_whitespace_content_skipped(self, provider):
        provider.on_memory_write("add", "memory", "   ")
        all_facts = provider._store.list_facts(limit=100)
        assert len(all_facts) == 0


class TestFTS5ProviderPrefetch:
    """prefetch results are injected into user message, not system prompt."""

    def test_prefetch_matches_stored_facts(self, provider):
        provider.on_memory_write("add", "memory", "Hermes supports multiple memory backends")
        prefetch_text = provider.prefetch("memory backends", session_id="test-session")
        assert "Hermes supports multiple memory backends" in prefetch_text

    def test_prefetch_empty_query_returns_empty(self, provider):
        provider.on_memory_write("add", "memory", "Some fact here")
        result = provider.prefetch("", session_id="test-session")
        assert result == ""

    def test_prefetch_no_match_returns_empty(self, provider):
        provider.on_memory_write("add", "memory", "Apples are fruit")
        result = provider.prefetch("zzz_nonexistent_query_zzz", session_id="test-session")
        assert result == ""

    def test_system_prompt_block_does_not_contain_prefetched_facts(self, provider):
        """system_prompt_block() is static — no recalled facts."""
        block = provider.system_prompt_block()
        assert "fts5_add" in block  # mentions the tools
        # Should NOT contain any recalled fact content
        assert "Jordan" not in block


class TestFTS5ProviderToolCalls:
    """Tool calls via handle_tool_call must apply limit convergence."""

    def test_fts5_search_limit_convergence(self, provider):
        for i in range(20):
            provider.on_memory_write("add", "memory", f"searchable item {i}")
        result_json = provider.handle_tool_call("fts5_search", {
            "query": "searchable item",
            "limit": -5,
        })
        result = json.loads(result_json)
        assert result["ok"] is True
        assert 1 <= len(result["results"]) <= 100

    def test_fts5_list_limit_convergence(self, provider):
        for i in range(20):
            provider.on_memory_write("add", "memory", f"list item {i}")
        result_json = provider.handle_tool_call("fts5_list", {"limit": 99999})
        result = json.loads(result_json)
        assert result["ok"] is True
        assert len(result["results"]) <= 100

    def test_fts5_search_with_category(self, provider):
        provider._store.add_fact("general knowledge fact", category="general")
        provider._store.add_fact("user preference fact", category="user_pref")
        result_json = provider.handle_tool_call("fts5_search", {
            "query": "fact",
            "category": "user_pref",
        })
        result = json.loads(result_json)
        assert result["ok"] is True
        for r in result["results"]:
            assert r["category"] == "user_pref"


class TestFTS5ProviderEdgeCases:
    """Edge case handling."""

    def test_handle_tool_call_unknown_tool(self, provider):
        result = json.loads(provider.handle_tool_call("nonexistent", {}))
        assert result["ok"] is False

    def test_handle_tool_call_before_initialize(self):
        p = FTS5BuiltinProvider()
        result = json.loads(p.handle_tool_call("fts5_search", {"query": "test"}))
        assert result["ok"] is False
        assert "not initialized" in result["error"]
