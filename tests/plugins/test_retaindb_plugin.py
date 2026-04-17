"""Tests for the RetainDB memory plugin.

Covers: _Client HTTP client, _WriteQueue SQLite queue, _build_overlay formatter,
RetainDBMemoryProvider lifecycle/tools/prefetch, thread management, connection pooling.
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Imports — guarded since plugins/memory lives outside the standard test path
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Ensure HERMES_HOME and RETAINDB vars are isolated."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("RETAINDB_API_KEY", raising=False)
    monkeypatch.delenv("RETAINDB_BASE_URL", raising=False)
    monkeypatch.delenv("RETAINDB_PROJECT", raising=False)


# We need the repo root on sys.path so the plugin can import agent.memory_provider
import sys
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from plugins.memory.retaindb import (
    _Client,
    _WriteQueue,
    _build_overlay,
    RetainDBMemoryProvider,
    _ASYNC_SHUTDOWN,
    _DEFAULT_BASE_URL,
)


# ===========================================================================
# _Client tests
# ===========================================================================

class TestClient:
    """Test the HTTP client with mocked requests."""

    def _make_client(self, api_key="rdb-test-key", base_url="https://api.retaindb.com", project="test"):
        return _Client(api_key, base_url, project)

    def test_base_url_trailing_slash_stripped(self):
        c = self._make_client(base_url="https://api.retaindb.com///")
        assert c.base_url == "https://api.retaindb.com"

    def test_headers_include_auth(self):
        c = self._make_client()
        h = c._headers("/v1/files")
        assert h["Authorization"] == "Bearer rdb-test-key"
        assert "X-API-Key" not in h

    def test_headers_include_api_key_for_memory_path(self):
        c = self._make_client()
        h = c._headers("/v1/memory/search")
        assert h["X-API-Key"] == "rdb-test-key"

    def test_headers_include_api_key_for_context_path(self):
        c = self._make_client()
        h = c._headers("/v1/context/query")
        assert h["X-API-Key"] == "rdb-test-key"

    def test_headers_strip_bearer_prefix(self):
        c = self._make_client(api_key="Bearer rdb-test-key")
        h = c._headers("/v1/memory/search")
        assert h["Authorization"] == "Bearer rdb-test-key"
        assert h["X-API-Key"] == "rdb-test-key"

    def test_add_memory_tries_fallback(self):
        c = self._make_client()
        call_count = 0
        def fake_request(method, path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("404")
            return {"id": "mem-1"}

        with patch.object(c, "request", side_effect=fake_request):
            result = c.add_memory("u1", "s1", "test fact")
            assert result == {"id": "mem-1"}
            assert call_count == 2

    def test_delete_memory_tries_fallback(self):
        c = self._make_client()
        call_count = 0
        def fake_request(method, path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("404")
            return {"deleted": True}

        with patch.object(c, "request", side_effect=fake_request):
            result = c.delete_memory("mem-123")
            assert result == {"deleted": True}
            assert call_count == 2

# ===========================================================================
# _WriteQueue tests
# ===========================================================================

class TestWriteQueue:
    """Test the SQLite-backed write queue with real SQLite."""

    def _make_queue(self, tmp_path, client=None):
        if client is None:
            client = MagicMock()
            client.ingest_session = MagicMock(return_value={"status": "ok"})
        db_path = tmp_path / "test_queue.db"
        return _WriteQueue(client, db_path), client, db_path

    def test_enqueue_creates_row(self, tmp_path):
        q, client, db_path = self._make_queue(tmp_path)
        q.enqueue("user1", "sess1", [{"role": "user", "content": "hi"}])
        # Give the writer thread a moment to process
        time.sleep(1)
        q.shutdown()
        # If ingest succeeded, the row should be deleted
        client.ingest_session.assert_called_once()

    def test_enqueue_persists_to_sqlite(self, tmp_path):
        client = MagicMock()
        # Make ingest hang so the row stays in SQLite
        client.ingest_session = MagicMock(side_effect=lambda *a, **kw: time.sleep(5))
        db_path = tmp_path / "test_queue.db"
        q = _WriteQueue(client, db_path)
        q.enqueue("user1", "sess1", [{"role": "user", "content": "test"}])
        # Check SQLite directly — row should exist since flush is slow
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT user_id, session_id FROM pending").fetchall()
        conn.close()
        assert len(rows) >= 1
        assert rows[0][0] == "user1"
        q.shutdown()

    def test_flush_deletes_row_on_success(self, tmp_path):
        q, client, db_path = self._make_queue(tmp_path)
        q.enqueue("user1", "sess1", [{"role": "user", "content": "hi"}])
        time.sleep(1)
        q.shutdown()
        # Row should be gone
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        conn.close()
        assert rows == 0

    def test_flush_records_error_on_failure(self, tmp_path):
        client = MagicMock()
        client.ingest_session = MagicMock(side_effect=RuntimeError("API down"))
        db_path = tmp_path / "test_queue.db"
        q = _WriteQueue(client, db_path)
        q.enqueue("user1", "sess1", [{"role": "user", "content": "hi"}])
        time.sleep(3)  # Allow retry + sleep(2) in _flush_row
        q.shutdown()
        # Row should still exist with error recorded
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT last_error FROM pending").fetchone()
        conn.close()
        assert row is not None
        assert "API down" in row[0]

    def test_thread_local_connection_reuse(self, tmp_path):
        q, _, _ = self._make_queue(tmp_path)
        # Same thread should get same connection
        conn1 = q._get_conn()
        conn2 = q._get_conn()
        assert conn1 is conn2
        q.shutdown()

    def test_crash_recovery_replays_pending(self, tmp_path):
        """Simulate crash: create rows, then new queue should replay them."""
        db_path = tmp_path / "recovery_test.db"
        # First: create a queue and insert rows, but don't let them flush
        client1 = MagicMock()
        client1.ingest_session = MagicMock(side_effect=RuntimeError("fail"))
        q1 = _WriteQueue(client1, db_path)
        q1.enqueue("user1", "sess1", [{"role": "user", "content": "lost turn"}])
        time.sleep(3)
        q1.shutdown()

        # Now create a new queue — it should replay the pending rows
        client2 = MagicMock()
        client2.ingest_session = MagicMock(return_value={"status": "ok"})
        q2 = _WriteQueue(client2, db_path)
        time.sleep(2)
        q2.shutdown()

        # The replayed row should have been ingested via client2
        client2.ingest_session.assert_called_once()
        call_args = client2.ingest_session.call_args
        assert call_args[0][0] == "user1"  # user_id


# ===========================================================================
# _build_overlay tests
# ===========================================================================

class TestBuildOverlay:
    """Test the overlay formatter (pure function)."""

    def test_empty_inputs_returns_empty(self):
        assert _build_overlay({}, {}) == ""

    def test_empty_memories_returns_empty(self):
        assert _build_overlay({"memories": []}, {"results": []}) == ""

    def test_profile_items_included(self):
        profile = {"memories": [{"content": "User likes Python"}]}
        result = _build_overlay(profile, {})
        assert "User likes Python" in result
        assert "[RetainDB Context]" in result

    def test_query_results_included(self):
        query_result = {"results": [{"content": "Previous discussion about Rust"}]}
        result = _build_overlay({}, query_result)
        assert "Previous discussion about Rust" in result

    def test_deduplication_removes_duplicates(self):
        profile = {"memories": [{"content": "User likes Python"}]}
        query_result = {"results": [{"content": "User likes Python"}]}
        result = _build_overlay(profile, query_result)
        assert result.count("User likes Python") == 1

    def test_local_entries_filter(self):
        profile = {"memories": [{"content": "Already known fact"}]}
        result = _build_overlay(profile, {}, local_entries=["Already known fact"])
        # The profile item matches a local entry, should be filtered
        assert result == ""

    def test_max_five_items_per_section(self):
        profile = {"memories": [{"content": f"Fact {i}"} for i in range(10)]}
        result = _build_overlay(profile, {})
        # Should only include first 5
        assert "Fact 0" in result
        assert "Fact 4" in result
        assert "Fact 5" not in result

    def test_none_content_handled(self):
        profile = {"memories": [{"content": None}, {"content": "Real fact"}]}
        result = _build_overlay(profile, {})
        assert "Real fact" in result

    def test_truncation_at_320_chars(self):
        long_content = "x" * 500
        profile = {"memories": [{"content": long_content}]}
        result = _build_overlay(profile, {})
        # Each item is compacted to 320 chars max
        for line in result.split("\n"):
            if line.startswith("- "):
                assert len(line) <= 322  # "- " + 320


# ===========================================================================
# RetainDBMemoryProvider tests
# ===========================================================================

class TestRetainDBMemoryProvider:
    """Test the main plugin class."""

    def _make_provider(self, tmp_path, monkeypatch, api_key="rdb-test-key"):
        monkeypatch.setenv("RETAINDB_API_KEY", api_key)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir(exist_ok=True)
        provider = RetainDBMemoryProvider()
        return provider

    def test_name(self):
        p = RetainDBMemoryProvider()
        assert p.name == "retaindb"

    def test_is_available_without_key(self):
        p = RetainDBMemoryProvider()
        assert p.is_available() is False

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test")
        p = RetainDBMemoryProvider()
        assert p.is_available() is True

    def test_config_schema(self):
        p = RetainDBMemoryProvider()
        schema = p.get_config_schema()
        assert len(schema) == 3
        keys = [s["key"] for s in schema]
        assert "api_key" in keys
        assert "base_url" in keys
        assert "project" in keys

    def test_initialize_creates_client_and_queue(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        assert p._client is not None
        assert p._queue is not None
        assert p._session_id == "test-session"
        p.shutdown()

    def test_initialize_default_project(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        assert p._client.project == "default"
        p.shutdown()

    def test_initialize_explicit_project(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_PROJECT", "my-project")
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        assert p._client.project == "my-project"
        p.shutdown()

    def test_initialize_profile_project(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        profile_home = str(tmp_path / "profiles" / "coder")
        p.initialize("test-session", hermes_home=profile_home)
        assert p._client.project == "hermes-coder"
        p.shutdown()

    def test_initialize_seeds_soul_md(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        soul_path = tmp_path / ".hermes" / "SOUL.md"
        soul_path.write_text("I am a helpful agent.")
        with patch.object(RetainDBMemoryProvider, "_seed_soul") as mock_seed:
            p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
            # Give thread time to start
            time.sleep(0.5)
            mock_seed.assert_called_once_with("I am a helpful agent.")
        p.shutdown()

    def test_system_prompt_block(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        block = p.system_prompt_block()
        assert "RetainDB Memory" in block
        assert "Active" in block
        p.shutdown()

    def test_handle_tool_call_not_initialized(self):
        p = RetainDBMemoryProvider()
        result = json.loads(p.handle_tool_call("retaindb_profile", {}))
        assert "error" in result
        assert "not initialized" in result["error"]

    def test_handle_tool_call_unknown_tool(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_nonexistent", {}))
        assert result == {"error": "Unknown tool: retaindb_nonexistent"}
        p.shutdown()

    def test_dispatch_profile(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "get_profile", return_value={"memories": []}):
            result = json.loads(p.handle_tool_call("retaindb_profile", {}))
            assert "memories" in result
        p.shutdown()

    def test_dispatch_search_requires_query(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_search", {}))
        assert result == {"error": "query is required"}
        p.shutdown()

    def test_dispatch_search(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "search", return_value={"results": [{"content": "found"}]}):
            result = json.loads(p.handle_tool_call("retaindb_search", {"query": "test"}))
            assert "results" in result
        p.shutdown()

    def test_dispatch_search_top_k_capped(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "search") as mock_search:
            mock_search.return_value = {"results": []}
            p.handle_tool_call("retaindb_search", {"query": "test", "top_k": 100})
            # top_k should be capped at 20
            assert mock_search.call_args[1]["top_k"] == 20
        p.shutdown()

    def test_dispatch_remember(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "add_memory", return_value={"id": "mem-1"}):
            result = json.loads(p.handle_tool_call("retaindb_remember", {"content": "test fact"}))
            assert result["id"] == "mem-1"
        p.shutdown()

    def test_dispatch_remember_requires_content(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_remember", {}))
        assert result == {"error": "content is required"}
        p.shutdown()

    def test_dispatch_forget(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "delete_memory", return_value={"deleted": True}):
            result = json.loads(p.handle_tool_call("retaindb_forget", {"memory_id": "mem-1"}))
            assert result["deleted"] is True
        p.shutdown()

    def test_dispatch_forget_requires_id(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_forget", {}))
        assert result == {"error": "memory_id is required"}
        p.shutdown()

    def test_dispatch_context(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "query_context", return_value={"results": [{"content": "relevant"}]}), \
             patch.object(p._client, "get_profile", return_value={"memories": []}):
            result = json.loads(p.handle_tool_call("retaindb_context", {"query": "current task"}))
            assert "context" in result
            assert "raw" in result
        p.shutdown()

    def test_dispatch_file_list(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "list_files", return_value={"files": []}):
            result = json.loads(p.handle_tool_call("retaindb_list_files", {}))
            assert "files" in result
        p.shutdown()

    def test_dispatch_file_upload_missing_path(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_upload_file", {}))
        assert "error" in result

    def test_dispatch_file_upload_not_found(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_upload_file", {"local_path": "/nonexistent/file.txt"}))
        assert "File not found" in result["error"]
        p.shutdown()

    def test_dispatch_file_read_requires_id(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_read_file", {}))
        assert result == {"error": "file_id is required"}
        p.shutdown()

    def test_dispatch_file_ingest_requires_id(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_ingest_file", {}))
        assert result == {"error": "file_id is required"}
        p.shutdown()

    def test_dispatch_file_delete_requires_id(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        result = json.loads(p.handle_tool_call("retaindb_delete_file", {}))
        assert result == {"error": "file_id is required"}
        p.shutdown()

    def test_handle_tool_call_wraps_exception(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "get_profile", side_effect=RuntimeError("API exploded")):
            result = json.loads(p.handle_tool_call("retaindb_profile", {}))
            assert "API exploded" in result["error"]
        p.shutdown()


# ===========================================================================
# Prefetch and thread management tests
# ===========================================================================

class TestPrefetch:
    """Test background prefetch and thread accumulation prevention."""

    def _make_initialized_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        p = RetainDBMemoryProvider()
        p.initialize("test-session", hermes_home=str(hermes_home))
        return p

    def test_queue_prefetch_skips_without_client(self):
        p = RetainDBMemoryProvider()
        p.queue_prefetch("test")  # Should not raise

    def test_prefetch_returns_empty_when_nothing_cached(self, tmp_path, monkeypatch):
        p = self._make_initialized_provider(tmp_path, monkeypatch)
        result = p.prefetch("test")
        assert result == ""
        p.shutdown()

    def test_prefetch_consumes_context_result(self, tmp_path, monkeypatch):
        p = self._make_initialized_provider(tmp_path, monkeypatch)
        # Manually set the cached result
        with p._lock:
            p._context_result = "[RetainDB Context]\nProfile:\n- User likes tests"
        result = p.prefetch("test")
        assert "User likes tests" in result
        # Should be consumed
        assert p.prefetch("test") == ""
        p.shutdown()

    def test_prefetch_consumes_dialectic_result(self, tmp_path, monkeypatch):
        p = self._make_initialized_provider(tmp_path, monkeypatch)
        with p._lock:
            p._dialectic_result = "User is a software engineer who prefers Python."
        result = p.prefetch("test")
        assert "[RetainDB User Synthesis]" in result
        assert "software engineer" in result
        p.shutdown()

    def test_prefetch_consumes_agent_model(self, tmp_path, monkeypatch):
        p = self._make_initialized_provider(tmp_path, monkeypatch)
        with p._lock:
            p._agent_model = {
                "memory_count": 5,
                "persona": "Helpful coding assistant",
                "persistent_instructions": ["Be concise", "Use Python"],
                "working_style": "Direct and efficient",
            }
        result = p.prefetch("test")
        assert "[RetainDB Agent Self-Model]" in result
        assert "Helpful coding assistant" in result
        assert "Be concise" in result
        assert "Direct and efficient" in result
        p.shutdown()

    def test_prefetch_skips_empty_agent_model(self, tmp_path, monkeypatch):
        p = self._make_initialized_provider(tmp_path, monkeypatch)
        with p._lock:
            p._agent_model = {"memory_count": 0}
        result = p.prefetch("test")
        assert "Agent Self-Model" not in result
        p.shutdown()

    def test_thread_accumulation_guard(self, tmp_path, monkeypatch):
        """Verify old prefetch threads are joined before new ones spawn."""
        p = self._make_initialized_provider(tmp_path, monkeypatch)
        # Mock the prefetch methods to be slow
        with patch.object(p, "_prefetch_context", side_effect=lambda q: time.sleep(0.5)), \
             patch.object(p, "_prefetch_dialectic", side_effect=lambda q: time.sleep(0.5)), \
             patch.object(p, "_prefetch_agent_model", side_effect=lambda: time.sleep(0.5)):
            p.queue_prefetch("query 1")
            first_threads = list(p._prefetch_threads)
            assert len(first_threads) == 3

            # Call again — should join first batch before spawning new
            p.queue_prefetch("query 2")
            second_threads = list(p._prefetch_threads)
            assert len(second_threads) == 3
            # Should be different thread objects
            for t in second_threads:
                assert t not in first_threads
        p.shutdown()

    def test_reasoning_level_short(self):
        assert RetainDBMemoryProvider._reasoning_level("hi") == "low"

    def test_reasoning_level_medium(self):
        assert RetainDBMemoryProvider._reasoning_level("x" * 200) == "medium"

    def test_reasoning_level_long(self):
        assert RetainDBMemoryProvider._reasoning_level("x" * 500) == "high"


# ===========================================================================
# sync_turn tests
# ===========================================================================

class TestSyncTurn:
    """Test turn synchronization via the write queue."""

    def test_sync_turn_enqueues(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        p = RetainDBMemoryProvider()
        p.initialize("test-session", hermes_home=str(hermes_home))
        with patch.object(p._queue, "enqueue") as mock_enqueue:
            p.sync_turn("user msg", "assistant msg")
            mock_enqueue.assert_called_once()
            args = mock_enqueue.call_args[0]
            assert args[0] == "default"  # user_id
            assert args[1] == "test-session"  # session_id
            msgs = args[2]
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"
        p.shutdown()

    def test_sync_turn_skips_empty_user_content(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        p = RetainDBMemoryProvider()
        p.initialize("test-session", hermes_home=str(hermes_home))
        with patch.object(p._queue, "enqueue") as mock_enqueue:
            p.sync_turn("", "assistant msg")
            mock_enqueue.assert_not_called()
        p.shutdown()


# ===========================================================================
# on_memory_write hook tests
# ===========================================================================

class TestOnMemoryWrite:
    """Test the built-in memory mirror hook."""

    def test_mirrors_add_action(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        p = RetainDBMemoryProvider()
        p.initialize("test-session", hermes_home=str(hermes_home))
        with patch.object(p._client, "add_memory", return_value={"id": "mem-1"}) as mock_add:
            p.on_memory_write("add", "user", "User prefers dark mode")
            mock_add.assert_called_once()
            assert mock_add.call_args[1]["memory_type"] == "preference"
        p.shutdown()

    def test_skips_non_add_action(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        p = RetainDBMemoryProvider()
        p.initialize("test-session", hermes_home=str(hermes_home))
        with patch.object(p._client, "add_memory") as mock_add:
            p.on_memory_write("remove", "user", "something")
            mock_add.assert_not_called()
        p.shutdown()

    def test_skips_empty_content(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        p = RetainDBMemoryProvider()
        p.initialize("test-session", hermes_home=str(hermes_home))
        with patch.object(p._client, "add_memory") as mock_add:
            p.on_memory_write("add", "user", "")
            mock_add.assert_not_called()
        p.shutdown()

    def test_memory_target_maps_to_type(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "rdb-test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        p = RetainDBMemoryProvider()
        p.initialize("test-session", hermes_home=str(hermes_home))
        with patch.object(p._client, "add_memory", return_value={"id": "mem-1"}) as mock_add:
            p.on_memory_write("add", "memory", "Some env fact")
            assert mock_add.call_args[1]["memory_type"] == "factual"
        p.shutdown()


# ===========================================================================
# register() test
# ===========================================================================

class TestRegister:
    def test_register_calls_register_memory_provider(self):
        from plugins.memory.retaindb import register
        ctx = MagicMock()
        register(ctx)
        ctx.register_memory_provider.assert_called_once()
        arg = ctx.register_memory_provider.call_args[0][0]
        assert isinstance(arg, RetainDBMemoryProvider)
