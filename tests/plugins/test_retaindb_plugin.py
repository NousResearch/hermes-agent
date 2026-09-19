"""Tests for the RetainDB memory plugin.

Covers: _Client HTTP client, _WriteQueue SQLite queue, _build_overlay formatter,
RetainDBMemoryProvider lifecycle/tools/prefetch, thread management, connection pooling.
"""

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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


@pytest.fixture(autouse=True)
def _cap_retaindb_sleeps(monkeypatch):
    """Cap production-code sleeps so background-thread tests run fast.

    The retaindb ``_WriteQueue._flush_row`` does ``time.sleep(2)`` after
    errors. Across multiple tests that trigger the retry path, that adds
    up. Cap the module's bound ``time.sleep`` to 0.05s — tests don't care
    about the exact retry delay, only that it happens. The test file's
    own ``time.sleep`` stays real since it uses a different reference.
    """
    try:
        from plugins.memory import retaindb as _retaindb
    except ImportError:
        return

    real_sleep = _retaindb.time.sleep

    def _capped_sleep(seconds):
        return real_sleep(min(float(seconds), 0.05))

    import types as _types
    fake_time = _types.SimpleNamespace(sleep=_capped_sleep, time=_retaindb.time.time)
    monkeypatch.setattr(_retaindb, "time", fake_time)


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
        # shutdown() blocks until the writer thread drains the queue — no need
        # to pre-sleep (the old 1s sleep was a just-in-case wait, but shutdown
        # does the right thing).
        q.shutdown()
        # If ingest succeeded, the row should be deleted
        client.ingest_session.assert_called_once()


    def test_flush_deletes_row_on_success(self, tmp_path):
        q, client, db_path = self._make_queue(tmp_path)
        q.enqueue("user1", "sess1", [{"role": "user", "content": "hi"}])
        q.shutdown()  # blocks until drain
        # Row should be gone
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        conn.close()
        assert rows == 0


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
        # Wait until the error is recorded (poll with short interval).
        deadline = time.time() + 2.0
        while time.time() < deadline:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT last_error FROM pending").fetchone()
            conn.close()
            if row and row[0]:
                break
            time.sleep(0.05)
        q1.shutdown()

        # Now create a new queue — it should replay the pending rows
        client2 = MagicMock()
        client2.ingest_session = MagicMock(return_value={"status": "ok"})
        q2 = _WriteQueue(client2, db_path)
        # Poll for the replay to happen.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if client2.ingest_session.called:
                break
            time.sleep(0.05)
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


    def test_dispatch_profile(self, tmp_path, monkeypatch):
        p = self._make_provider(tmp_path, monkeypatch)
        p.initialize("test-session", hermes_home=str(tmp_path / ".hermes"))
        with patch.object(p._client, "get_profile", return_value={"memories": []}):
            result = json.loads(p.handle_tool_call("retaindb_profile", {}))
            assert "memories" in result
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


    def test_reasoning_level_short(self):
        assert RetainDBMemoryProvider._reasoning_level("hi") == "low"


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
