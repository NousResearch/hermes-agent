import json
from typing import Any, cast

import pytest

from plugins.memory.gbrain import (
    GBrainMemoryProvider,
    _GBrainMCPClient,
    _coerce_config,
    _load_gbrain_config,
    _save_gbrain_config,
)


class FakeGBrainClient:
    def __init__(self, endpoint="http://example.test/mcp", timeout=5.0):
        self.endpoint = endpoint
        self.timeout = timeout
        self.calls = []
        self.fail = False
        self.results: dict[str, Any] = {
            "query": {
                "content": [
                    {"type": "text", "text": "Garry prefers English-only replies."}
                ]
            }
        }

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise RuntimeError("connection refused")
        if name not in self.results:
            raise RuntimeError(f"unknown tool {name}")
        return self.results[name]


@pytest.fixture
def provider(tmp_path):
    p = GBrainMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_identity="default")
    fake = FakeGBrainClient()
    p._client = cast(Any, fake)
    return p


def test_coerce_config_defaults_to_read_only():
    cfg = _coerce_config({})
    assert cfg["endpoint"] == "http://127.0.0.1:3132/mcp"
    assert cfg["mode"] == "read-only"
    assert cfg["max_results"] == 6


def test_save_and_load_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _save_gbrain_config(
        {
            "endpoint": "http://127.0.0.1:3132/mcp",
            "mode": "read-write",
            "source_id": "default",
            "max_results": 99,
        },
        str(tmp_path),
    )
    cfg = _load_gbrain_config(str(tmp_path))
    assert cfg["mode"] == "read-write"
    assert cfg["source_id"] == "default"
    assert cfg["max_results"] == 20  # clamped


def test_is_available_checks_endpoint_only_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GBRAIN_MCP_ENDPOINT", "")
    assert GBrainMemoryProvider().is_available() is False

    monkeypatch.setenv("GBRAIN_MCP_ENDPOINT", "http://127.0.0.1:3132/mcp")
    assert GBrainMemoryProvider().is_available() is True


def test_prefetch_formats_mcp_text_result(provider):
    result = provider.prefetch("language preference")
    assert "<gbrain-memory-context>" in result
    assert "Garry prefers English-only replies" in result
    assert provider._client.calls[0][0] == "query"
    assert provider._client.calls[0][1]["adaptive_return"] is True
    assert provider._client.calls[0][1]["source_id"] == "__all__"


def test_search_tool_returns_structured_context(provider):
    payload = json.loads(
        provider.handle_tool_call("gbrain_memory_search", {"query": "language", "limit": 3})
    )
    assert payload["ok"] is True
    assert "GBrain recalled context" in payload["context"]
    assert provider._client.calls[0][1]["limit"] == 3


def test_search_falls_back_to_alternate_query_tool(provider):
    provider._client.results = {
        "gbrain_query": {"content": [{"type": "text", "text": "fallback result"}]}
    }
    payload = json.loads(provider.handle_tool_call("gbrain_memory_search", {"query": "x"}))
    assert payload["ok"] is True
    assert "fallback result" in payload["context"]
    assert [name for name, _ in provider._client.calls] == ["query", "gbrain_query"]


def test_unavailable_mcp_returns_error_for_tool_and_empty_prefetch(provider):
    provider._client.fail = True
    assert provider.prefetch("anything") == ""
    payload = json.loads(provider.handle_tool_call("gbrain_memory_search", {"query": "anything"}))
    assert payload["ok"] is False
    assert "connection refused" in payload["error"]


def test_read_only_store_candidate_is_blocked(provider):
    payload = json.loads(
        provider.handle_tool_call(
            "gbrain_memory_store_candidate",
            {"content": "User prefers concise responses", "target": "user"},
        )
    )
    assert payload["blocked"] is True
    assert payload["mode"] == "read-only"
    assert provider._client.calls == []


def test_read_only_sync_and_memory_write_do_not_call_client(provider):
    provider.sync_turn("remember x", "ok", session_id="session-1")
    provider.on_memory_write("add", "memory", "User prefers concise responses")
    assert provider._client.calls == []


def test_read_write_store_candidate_calls_configured_write_tool(tmp_path):
    _save_gbrain_config(
        {"mode": "read-write", "write_tool": "create_page", "endpoint": "http://example.test/mcp"},
        str(tmp_path),
    )
    p = GBrainMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_identity="athena")
    fake = FakeGBrainClient()
    fake.results["create_page"] = {"ok": True, "slug": "memory/user-prefers-concise"}
    p._client = cast(Any, fake)

    payload = json.loads(
        p.handle_tool_call(
            "gbrain_memory_store_candidate",
            {"content": "User prefers concise responses", "target": "user"},
        )
    )

    assert payload["ok"] is True
    assert fake.calls[0][0] == "create_page"
    assert fake.calls[0][1]["metadata"]["target"] == "user"


def test_read_write_on_memory_write_mirrors_explicit_add(tmp_path):
    _save_gbrain_config(
        {"mode": "read-write", "write_tool": "create_page", "endpoint": "http://example.test/mcp"},
        str(tmp_path),
    )
    p = GBrainMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_identity="athena")
    fake = FakeGBrainClient()
    fake.results["create_page"] = {"ok": True}
    p._client = cast(Any, fake)

    p.on_memory_write("add", "memory", "Durable preference", {"write_origin": "tool"})

    assert fake.calls[0][0] == "create_page"
    assert fake.calls[0][1]["content"] == "Durable preference"
    assert fake.calls[0][1]["metadata"]["agent_identity"] == "athena"
    assert fake.calls[0][1]["metadata"]["write_origin"] == "tool"


def test_post_setup_activates_read_only_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import load_config

    config = {"memory": {"gbrain": {"mode": "read-write", "endpoint": "http://writer/mcp"}}}
    p = GBrainMemoryProvider()
    p.post_setup(str(tmp_path), config)

    saved = load_config()
    assert saved["memory"]["provider"] == "gbrain"
    assert saved["memory"]["gbrain"]["mode"] == "read-only"
    assert saved["memory"]["gbrain"]["endpoint"] == "http://writer/mcp"
    assert _load_gbrain_config(str(tmp_path))["mode"] == "read-only"


def test_sse_decode_response_extracts_data_event():
    response = _GBrainMCPClient._decode_response(
        'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n',
        "text/event-stream",
    )
    assert response["result"] == {"ok": True}
