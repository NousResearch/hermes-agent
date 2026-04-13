"""Tests for MemOS Platform memory provider."""

import json
import pytest
import threading

from plugins.memory.memos import MemosMemoryProvider

class FakeMemosClient:
    """Fake MemOS client for testing."""
    def __init__(self, search_results=None, add_result=None):
        self._search_results = search_results or {"code": 0, "data": {"memory_detail_list": []}}
        self._add_result = add_result or {"code": 0, "message": "success"}
        self.captured_search = []
        self.captured_add = []

    def search_memory(self, **kwargs):
        self.captured_search.append(kwargs)
        return self._search_results

    def add_message(self, **kwargs):
        self.captured_add.append(kwargs)
        return self._add_result

def test_memos_is_available(monkeypatch):
    provider = MemosMemoryProvider()
    monkeypatch.setenv("MEMOS_API_KEY", "test_key")
    assert provider.is_available() is True
    monkeypatch.delenv("MEMOS_API_KEY", raising=False)

def test_memos_prefetch_success(monkeypatch):
    client = FakeMemosClient(search_results={
        "code": 0,
        "data": {
            "memory_detail_list": [
                {"memory_value": "User likes Python"}
            ],
            "preference_detail_list": [
                {"preference": "Dark mode"}
            ]
        }
    })
    
    provider = MemosMemoryProvider()
    provider.initialize("test-session")
    provider._user_id = "test-user"
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    result = provider.prefetch("What does user like?")
    
    assert "User likes Python" in result
    assert "Dark mode" in result
    assert len(client.captured_search) == 1
    assert client.captured_search[0]["query"] == "What does user like?"
    assert client.captured_search[0]["user_id"] == "test-user"

def test_memos_sync_turn(monkeypatch):
    client = FakeMemosClient()
    
    provider = MemosMemoryProvider()
    provider.initialize("test-session")
    provider._user_id = "test-user"
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    provider.sync_turn("hello", "hi there", session_id="test-session")
    provider._sync_thread.join(timeout=2)

    assert len(client.captured_add) == 1
    assert client.captured_add[0]["user_id"] == "test-user"
    assert client.captured_add[0]["conversation_id"] == "test-session"
    assert len(client.captured_add[0]["messages"]) == 2

def test_memos_search_tool(monkeypatch):
    client = FakeMemosClient(search_results={
        "code": 0,
        "data": {
            "memory_detail_list": [
                {"memory_value": "User uses macOS"}
            ]
        }
    })
    
    provider = MemosMemoryProvider()
    provider.initialize("test-session")
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    result_json = provider.handle_tool_call("memos_search", {"query": "macOS"})
    result = json.loads(result_json)
    
    assert result["count"] == 1
    assert "User uses macOS" in result["results"]

def test_memos_add_message_tool(monkeypatch):
    client = FakeMemosClient()
    
    provider = MemosMemoryProvider()
    provider.initialize("test-session")
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    result_json = provider.handle_tool_call("memos_add_message", {"content": "Save this fact"})
    result = json.loads(result_json)
    
    assert "result" in result
    assert len(client.captured_add) == 1
    assert client.captured_add[0]["messages"][0]["content"] == "Save this fact"

def test_multi_agent_isolation(monkeypatch):
    client = FakeMemosClient()
    
    provider = MemosMemoryProvider()
    # Mock config to enable multi_agent_mode
    monkeypatch.setattr("plugins.memory.memos._load_config", lambda: {"multiAgentMode": True})
    provider.initialize("test-session", agent_id="agent-123")
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    provider.prefetch("query")
    
    assert len(client.captured_search) == 1
    filters = client.captured_search[0].get("filter")
    assert filters == {"user": {"and": [{"agent_id": "agent-123"}]}}
