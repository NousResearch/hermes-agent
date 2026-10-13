"""
Tests for memory_tencentdb_v2 provider.
"""
import pytest
from unittest.mock import MagicMock, patch
import os

# Mock the SDK before importing provider
mock_client_cls = MagicMock()
mock_async_client_cls = MagicMock()
mock_error_cls = type("TDAMError", (Exception,), {"code": 0, "message": ""})


@pytest.fixture(autouse=True)
def patch_sdk(monkeypatch):
    """Patch the SDK import for all tests."""
    import memory_tencentdb_v2 as mod
    monkeypatch.setattr(mod, "_sdk_available", True)
    monkeypatch.setattr(mod, "_MemoryClient", mock_client_cls)
    monkeypatch.setattr(mod, "_TDAMError", mock_error_cls)


@pytest.fixture
def provider():
    from memory_tencentdb_v2 import MemoryTencentdbV2Provider
    p = MemoryTencentdbV2Provider()
    return p


@pytest.fixture
def initialized_provider(provider, monkeypatch):
    """Provider with a mock client."""
    monkeypatch.setenv("TDAI_MEMORY_ENDPOINT", "http://localhost:3100")
    monkeypatch.setenv("TDAI_MEMORY_API_KEY", "test-key")
    monkeypatch.setenv("TDAI_MEMORY_SERVICE_ID", "space-001")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    provider.initialize("test-session-001")
    provider._client = mock_client
    return provider, mock_client


class TestLifecycle:
    def test_is_available_with_sdk(self, provider):
        assert provider.is_available() is True

    def test_initialize_creates_client(self, initialized_provider):
        provider, mock_client = initialized_provider
        assert provider._available is True
        assert provider._session_id == "test-session-001"

    def test_shutdown_clears_state(self, initialized_provider):
        provider, _ = initialized_provider
        provider.shutdown()
        assert provider._client is None
        assert provider._available is False


class TestPrefetch:
    def test_prefetch_returns_memories_and_core(self, initialized_provider):
        provider, mock_client = initialized_provider

        mock_client.search_atomic.return_value = {
            "results": [
                {"content": "User likes Python", "type": "persona", "score": 0.9},
            ]
        }
        mock_client.read_core.return_value = {"content": "# User Profile\nSoftware engineer."}
        mock_client.list_scenarios.return_value = {"entries": []}

        result = provider.prefetch("What does the user like?")

        assert result is not None
        assert "Python" in result["prepend_context"]
        assert "Software engineer" in result["append_system_context"]

    def test_prefetch_empty_memories(self, initialized_provider):
        provider, mock_client = initialized_provider
        mock_client.search_atomic.return_value = {"results": []}
        mock_client.read_core.return_value = {"content": ""}
        mock_client.list_scenarios.return_value = {"entries": []}

        result = provider.prefetch("anything")
        assert result is not None
        assert result["prepend_context"] == ""

    # Regression: the real SDK returns {"items": [...]} for search_atomic and
    # {"messages": [...]} for search_conversation, not {"results": [...]}. The
    # adapter must read both shapes or recall silently returns nothing.
    def test_prefetch_reads_sdk_items_key(self, initialized_provider):
        provider, mock_client = initialized_provider
        mock_client.search_atomic.return_value = {
            "items": [{"content": "User likes Go", "type": "persona", "score": 0.9}]
        }
        mock_client.read_core.return_value = {"content": ""}
        mock_client.list_scenarios.return_value = {"entries": []}

        result = provider.prefetch("language")
        assert "Go" in result["prepend_context"]


class TestSyncTurn:
    def test_sync_turn_calls_add_conversation(self, initialized_provider):
        provider, mock_client = initialized_provider
        mock_client.add_conversation.return_value = {"accepted_ids": ["m1", "m2"], "total_count": 2}

        provider.sync_turn("Hello", "Hi there!", session_id="sess-1")

        mock_client.add_conversation.assert_called_once()
        args = mock_client.add_conversation.call_args
        assert args.kwargs["session_id"] == "sess-1"
        assert len(args.kwargs["messages"]) == 2


class TestTools:
    def test_get_tool_schemas(self, initialized_provider):
        provider, _ = initialized_provider
        schemas = provider.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "tdai_memory_search" in names
        assert "tdai_conversation_search" in names
        assert "tdai_read_scene" in names

    def test_handle_memory_search(self, initialized_provider):
        provider, mock_client = initialized_provider
        mock_client.search_atomic.return_value = {
            "results": [{"content": "Likes coffee", "type": "persona"}]
        }

        result = provider.handle_tool_call("tdai_memory_search", {"query": "coffee"})
        assert "coffee" in result.lower()

    def test_handle_read_scene(self, initialized_provider):
        provider, mock_client = initialized_provider
        mock_client.read_scenario.return_value = {"content": "# Travel\nGoing to Japan."}

        result = provider.handle_tool_call("tdai_read_scene", {"scene_id": "travel-plan"})
        assert "Japan" in result

    def test_handle_unknown_tool(self, initialized_provider):
        provider, _ = initialized_provider
        assert provider.handle_tool_call("unknown_tool", {}) is None


class TestCircuitBreaker:
    def test_circuit_opens_after_max_failures(self, initialized_provider):
        provider, mock_client = initialized_provider
        mock_client.search_atomic.side_effect = Exception("connection refused")

        # Trigger max_failures times
        for _ in range(5):
            provider.prefetch("test")

        assert provider._consecutive_failures >= 5

        # Next call should be skipped (circuit open)
        result = provider.prefetch("test")
        assert result is None

    def test_circuit_resets_on_success(self, initialized_provider):
        provider, mock_client = initialized_provider
        provider._consecutive_failures = 4

        mock_client.search_atomic.return_value = {"results": []}
        mock_client.read_core.return_value = {"content": ""}
        mock_client.list_scenarios.return_value = {"entries": []}

        provider.prefetch("test")
        assert provider._consecutive_failures == 0
