"""Tests for Mem0 API v2 compatibility — filters param and dict response unwrapping.

Salvaged from PRs #5301 (qaqcvc) and #5117 (vvvanguards).
"""

import json
import os
import stat
import time

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider


class FakeClientV2:
    """Fake Mem0 client that returns v2-style dict responses and captures call kwargs."""

    def __init__(self, search_results=None, all_results=None):
        self._search_results = search_results or {"results": []}
        self._all_results = all_results or {"results": []}
        self.captured_search = {}
        self.captured_get_all = {}
        self.captured_add = []

    def search(self, **kwargs):
        self.captured_search = kwargs
        return self._search_results

    def get_all(self, **kwargs):
        self.captured_get_all = kwargs
        return self._all_results

    def add(self, messages, **kwargs):
        self.captured_add.append({"messages": messages, **kwargs})

    def delete(self, **kwargs):
        self.captured_delete = kwargs
        return {"message": "Memory deleted successfully."}

    def delete_all(self, **kwargs):
        self.captured_delete_all = kwargs
        return {"message": "All memories deleted successfully."}

    def batch_delete(self, **kwargs):
        self.captured_batch_delete = kwargs
        return {"message": "Memories deleted successfully."}


# ---------------------------------------------------------------------------
# Filter migration: bare user_id= -> filters={}
# ---------------------------------------------------------------------------


class TestMem0FiltersV2:
    """All API calls must use filters={} instead of bare user_id= kwargs."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_search_uses_filters(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        provider.handle_tool_call("mem0_search", {"query": "hello", "top_k": 3, "rerank": False})

        assert client.captured_search["query"] == "hello"
        assert client.captured_search["top_k"] == 3
        assert client.captured_search["rerank"] is False
        assert client.captured_search["filters"] == {"user_id": "u123"}
        # Must NOT have bare user_id kwarg
        assert "user_id" not in {k for k in client.captured_search if k != "filters"}

    def test_profile_uses_filters(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        provider.handle_tool_call("mem0_profile", {})

        assert client.captured_get_all["filters"] == {"user_id": "u123"}
        assert "user_id" not in {k for k in client.captured_get_all if k != "filters"}

    def test_prefetch_uses_filters(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        provider.queue_prefetch("hello")
        provider._prefetch_thread.join(timeout=2)

        assert client.captured_search["query"] == "hello"
        assert client.captured_search["filters"] == {"user_id": "u123"}
        assert "user_id" not in {k for k in client.captured_search if k != "filters"}

    def test_sync_turn_uses_write_filters(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        provider.sync_turn("user said this", "assistant replied", session_id="s1")
        provider._sync_thread.join(timeout=2)

        assert len(client.captured_add) == 1
        call = client.captured_add[0]
        assert call["user_id"] == "u123"
        assert call["agent_id"] == "hermes"

    def test_conclude_uses_write_filters(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        provider.handle_tool_call("mem0_conclude", {"conclusion": "user likes dark mode"})

        assert len(client.captured_add) == 1
        call = client.captured_add[0]
        assert call["user_id"] == "u123"
        assert call["agent_id"] == "hermes"
        assert call["infer"] is False

    def test_read_filters_no_agent_id(self):
        """Read filters should use user_id only — cross-session recall across agents."""
        provider = Mem0MemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        assert provider._read_filters() == {"user_id": "u123"}

    def test_write_filters_include_agent_id(self):
        """Write filters should include agent_id for attribution."""
        provider = Mem0MemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        assert provider._write_filters() == {"user_id": "u123", "agent_id": "hermes"}


# ---------------------------------------------------------------------------
# Dict response unwrapping (API v2 wraps in {"results": [...]})
# ---------------------------------------------------------------------------


class TestMem0ResponseUnwrapping:
    """API v2 returns {"results": [...]} dicts; we must extract the list."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_profile_dict_response(self, monkeypatch):
        client = FakeClientV2(all_results={"results": [{"memory": "alpha"}, {"memory": "beta"}]})
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_profile", {}))

        assert result["count"] == 2
        assert "alpha" in result["result"]
        assert "beta" in result["result"]

    def test_profile_list_response_backward_compat(self, monkeypatch):
        """Old API returned bare lists — still works."""
        client = FakeClientV2(all_results=[{"memory": "gamma"}])
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert result["count"] == 1
        assert "gamma" in result["result"]

    def test_search_dict_response(self, monkeypatch):
        client = FakeClientV2(search_results={
            "results": [{"memory": "foo", "score": 0.9}, {"memory": "bar", "score": 0.7}]
        })
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call(
            "mem0_search", {"query": "test", "top_k": 5}
        ))

        assert result["count"] == 2
        assert result["results"][0]["memory"] == "foo"

    def test_search_list_response_backward_compat(self, monkeypatch):
        """Old API returned bare lists — still works."""
        client = FakeClientV2(search_results=[{"memory": "baz", "score": 0.8}])
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call(
            "mem0_search", {"query": "test"}
        ))
        assert result["count"] == 1

    def test_unwrap_results_edge_cases(self):
        """_unwrap_results handles all shapes gracefully."""
        assert Mem0MemoryProvider._unwrap_results({"results": [1, 2]}) == [1, 2]
        assert Mem0MemoryProvider._unwrap_results([3, 4]) == [3, 4]
        assert Mem0MemoryProvider._unwrap_results({}) == []
        assert Mem0MemoryProvider._unwrap_results(None) == []
        assert Mem0MemoryProvider._unwrap_results("unexpected") == []

    def test_prefetch_dict_response(self, monkeypatch):
        client = FakeClientV2(search_results={
            "results": [{"memory": "user prefers dark mode"}]
        })
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        monkeypatch.setattr(provider, "_get_client", lambda: client)

        provider.queue_prefetch("preferences")
        provider._prefetch_thread.join(timeout=2)
        result = provider.prefetch("preferences")

        assert "dark mode" in result


# ---------------------------------------------------------------------------
# Default preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    """mem0.json must be written with 0o600 so API key is not world-readable."""
    provider = Mem0MemoryProvider()
    provider.save_config({"api_key": "m0-test-key"}, str(tmp_path))
    config_file = tmp_path / "mem0.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"


class TestMem0Defaults:
    """Ensure we don't break existing users' defaults."""

    def test_default_user_id_hermes_user(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize("test")

        assert provider._user_id == "hermes-user"

    def test_default_agent_id_hermes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize("test")

        assert provider._agent_id == "hermes"


# ---------------------------------------------------------------------------
# Delete tools: filters, safety, and response handling
# ---------------------------------------------------------------------------


class TestMem0DeleteFiltersV2:
    """Delete API calls must use correct filters and capture kwargs."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_delete_uses_memory_id(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_delete", {"memory_id": "mem-abc"}))

        assert client.captured_delete["memory_id"] == "mem-abc"
        assert result["result"] == "Memory deleted."
        assert result["memory_id"] == "mem-abc"

    def test_delete_all_uses_write_filters(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_delete_all", {"confirm": True}))

        assert client.captured_delete_all["user_id"] == "u123"
        assert client.captured_delete_all["agent_id"] == "hermes"
        assert result["result"] == "All memories deleted successfully."

    def test_batch_delete_uses_memory_ids(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_batch_delete", {"memory_ids": ["m1", "m2", "m3"]}))

        assert len(client.captured_batch_delete["memories"]) == 3
        assert client.captured_batch_delete["memories"][0] == {"memory_id": "m1"}
        assert result["result"] == "Memories deleted successfully."
        assert result["count"] == 3


class TestMem0DeleteSafety:
    """Safety guardrails for destructive operations."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_delete_missing_memory_id(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_delete", {}))

        assert "error" in result
        assert "memory_id" in result["error"].lower()

    def test_delete_all_without_confirm(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_delete_all", {}))

        assert "error" in result
        assert "confirm" in result["error"].lower()
        # Should NOT have called the SDK
        assert not hasattr(client, "captured_delete_all")

    def test_delete_all_with_confirm_false(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_delete_all", {"confirm": False}))

        assert "error" in result
        assert "confirm" in result["error"].lower()
        assert not hasattr(client, "captured_delete_all")

    def test_delete_all_with_confirm_string_true(self, monkeypatch):
        """String 'true' must be rejected — strict bool check."""
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_delete_all", {"confirm": "true"}))

        assert "error" in result
        assert not hasattr(client, "captured_delete_all")

    def test_batch_delete_empty_list(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_batch_delete", {"memory_ids": []}))

        assert "error" in result
        assert "memory_ids" in result["error"].lower()

    def test_batch_delete_over_limit(self, monkeypatch):
        client = FakeClientV2()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_batch_delete", {"memory_ids": ["m"] * 51}))

        assert "error" in result
        assert "limit" in result["error"].lower()


class TestMem0DeleteResponses:
    """Response unwrapping and error handling for delete operations."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_delete_api_error(self, monkeypatch):
        class BrokenClient:
            def delete(self, **kwargs):
                raise RuntimeError("API down")

        provider = self._make_provider(monkeypatch, BrokenClient())
        result = json.loads(provider.handle_tool_call("mem0_delete", {"memory_id": "m1"}))

        assert "error" in result
        assert "API down" in result["error"]

    def test_delete_all_api_error(self, monkeypatch):
        class BrokenClient:
            def delete_all(self, **kwargs):
                raise RuntimeError("service unavailable")

        provider = self._make_provider(monkeypatch, BrokenClient())
        result = json.loads(provider.handle_tool_call("mem0_delete_all", {"confirm": True}))

        assert "error" in result
        assert "service unavailable" in result["error"]

    def test_batch_delete_api_error(self, monkeypatch):
        class BrokenClient:
            def batch_delete(self, **kwargs):
                raise RuntimeError("batch failed")

        provider = self._make_provider(monkeypatch, BrokenClient())
        result = json.loads(provider.handle_tool_call("mem0_batch_delete", {"memory_ids": ["m1", "m2"]}))

        assert "error" in result
        assert "batch failed" in result["error"]

    def test_circuit_breaker_blocks_delete(self, monkeypatch):
        """When breaker is open, all delete tools return the breaker message."""
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._consecutive_failures = 999
        provider._breaker_open_until = time.monotonic() + 9999

        result = json.loads(provider.handle_tool_call("mem0_delete", {"memory_id": "m1"}))
        assert "unavailable" in result["error"].lower()

        result = json.loads(provider.handle_tool_call("mem0_delete_all", {"confirm": True}))
        assert "unavailable" in result["error"].lower()

        result = json.loads(provider.handle_tool_call("mem0_batch_delete", {"memory_ids": ["m1"]}))
        assert "unavailable" in result["error"].lower()
