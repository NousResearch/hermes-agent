"""Tests for mem0_integration/manager.py — Mem0 memory manager."""

import time
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from mem0_integration.client import Mem0ClientConfig
from mem0_integration.manager import Mem0MemoryManager


@pytest.fixture
def config():
    return Mem0ClientConfig(
        api_key="m0-test",
        enabled=True,
        user_id="testuser",
        agent_id="hermes",
        rerank=True,
        keyword_search=True,
        custom_instructions=None,
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.add.return_value = {"results": [{"status": "PENDING"}]}
    client.search.return_value = {
        "results": [
            {"id": "m1", "memory": "Likes Python", "score": 0.9, "categories": ["technology"]},
            {"id": "m2", "memory": "Works at Acme", "score": 0.7, "categories": ["professional_details"]},
        ]
    }
    client.get_all.return_value = {
        "results": [
            {"id": "m1", "memory": "Likes Python"},
            {"id": "m2", "memory": "Works at Acme"},
        ]
    }
    return client


@pytest.fixture
def manager(mock_client, config):
    return Mem0MemoryManager(client=mock_client, config=config)


class TestAdd:
    def test_add_sends_messages_to_mem0(self, manager, mock_client):
        messages = [
            {"role": "user", "content": "I like Python"},
            {"role": "assistant", "content": "Noted!"},
        ]
        manager.add(messages, user_id="testuser", run_id="session-1")
        mock_client.add.assert_called_once_with(
            messages,
            user_id="testuser",
            run_id="session-1",
            custom_instructions=None,
        )

    def test_add_with_custom_instructions(self, mock_client):
        config = Mem0ClientConfig(
            api_key="key", enabled=True, user_id="u",
            custom_instructions="Only tech prefs",
        )
        mgr = Mem0MemoryManager(client=mock_client, config=config)
        mgr.add([{"role": "user", "content": "hi"}], user_id="u")
        _, kwargs = mock_client.add.call_args
        assert kwargs["custom_instructions"] == "Only tech prefs"

    def test_add_handles_exception_gracefully(self, manager, mock_client):
        mock_client.add.side_effect = Exception("network error")
        # Should not raise
        manager.add([{"role": "user", "content": "test"}], user_id="u")


class TestSearch:
    def test_search_default_mode(self, manager, mock_client):
        results = manager.search("programming", user_id="testuser")
        mock_client.search.assert_called_once()
        _, kwargs = mock_client.search.call_args
        assert kwargs["version"] == "v2"
        assert kwargs["filters"] == {"OR": [{"user_id": "testuser"}, {"AND": [{"user_id": "testuser"}, {"run_id": "*"}]}]}
        assert kwargs["keyword_search"] is True
        assert "rerank" not in kwargs  # rerank off by default
        assert len(results) == 2

    def test_search_with_rerank(self, manager, mock_client):
        manager.search("programming", user_id="testuser", rerank=True)
        _, kwargs = mock_client.search.call_args
        assert kwargs["rerank"] is True

    def test_search_uses_v2_filters(self, manager, mock_client):
        """v2 API passes user_id inside filters with OR operator."""
        manager.search("query", user_id="testuser")
        _, kwargs = mock_client.search.call_args
        assert kwargs["version"] == "v2"
        assert kwargs["filters"] == {"OR": [{"user_id": "testuser"}, {"AND": [{"user_id": "testuser"}, {"run_id": "*"}]}]}

    def test_search_handles_exception(self, manager, mock_client):
        mock_client.search.side_effect = Exception("fail")
        results = manager.search("query", user_id="testuser")
        assert results == []


class TestGetProfile:
    def test_returns_memories(self, manager, mock_client):
        result = manager.get_profile(user_id="testuser")
        mock_client.get_all.assert_called_once_with(
            version="v2",
            filters={"OR": [{"user_id": "testuser"}, {"AND": [{"user_id": "testuser"}, {"run_id": "*"}]}]},
            page_size=20,
        )
        assert len(result) == 2

    def test_handles_exception(self, manager, mock_client):
        mock_client.get_all.side_effect = Exception("fail")
        result = manager.get_profile(user_id="testuser")
        assert result == []


class TestStoreFact:
    def test_stores_with_infer_false(self, manager, mock_client):
        manager.store_fact("Prefers dark mode", user_id="testuser")
        mock_client.add.assert_called_once()
        args, kwargs = mock_client.add.call_args
        assert args[0] == [{"role": "user", "content": "Prefers dark mode"}]
        assert kwargs["infer"] is False
        assert kwargs["metadata"] == {"source": "hermes_conclude"}

    def test_handles_exception(self, manager, mock_client):
        mock_client.add.side_effect = Exception("fail")
        result = manager.store_fact("test", user_id="u")
        assert "error" in result


class TestPrefetch:
    def _wait_for_prefetch(self, manager, user_id, run_id=None, timeout=5):
        """Wait for prefetch thread to complete (deterministic, no sleep)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with manager._prefetch_lock:
                if (user_id, run_id) in manager._prefetch_cache:
                    return
            time.sleep(0.01)

    def test_prefetch_caches_result(self, manager, mock_client):
        manager.prefetch(user_id="testuser", query="what do you know?")
        self._wait_for_prefetch(manager, "testuser")
        result = manager.pop_prefetch(user_id="testuser")
        assert result is not None
        assert "Likes Python" in result
        assert "Works at Acme" in result

    def test_pop_prefetch_returns_none_when_empty(self, manager):
        result = manager.pop_prefetch(user_id="nobody")
        assert result is None

    def test_pop_prefetch_clears_cache(self, manager, mock_client):
        manager.prefetch(user_id="testuser", query="test")
        self._wait_for_prefetch(manager, "testuser")
        first = manager.pop_prefetch(user_id="testuser")
        second = manager.pop_prefetch(user_id="testuser")
        assert first is not None
        assert second is None


class TestShutdown:
    def test_shutdown_is_noop(self, manager):
        # Should not raise
        manager.shutdown()
