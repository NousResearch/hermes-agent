"""Tests for the Mem0 HTTP memory plugin."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.delenv("MEM0_BASE_URL", raising=False)
    monkeypatch.delenv("MEM0_USER_ID", raising=False)
    monkeypatch.delenv("MEM0_AGENT_ID", raising=False)


_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from plugins.memory.mem0_http import (  # noqa: E402
    _Client,
    Mem0HttpMemoryProvider,
    _DEFAULT_BASE_URL,
    _load_config,
    register,
)


class TestClient:
    def test_headers_include_token_auth(self):
        client = _Client("test-key", _DEFAULT_BASE_URL)
        assert client._headers()["Authorization"] == "Token test-key"

    def test_search_payload_uses_and_filter(self):
        client = _Client("test-key", _DEFAULT_BASE_URL)
        with patch.object(client, "request", return_value={"results": []}) as mock_request:
            client.search(
                "deploy workflow",
                filters={"user_id": "hermes-agent", "agent_id": "hermes-agent"},
                rerank=True,
                top_k=5,
            )
        mock_request.assert_called_once_with(
            "POST",
            "/v2/memories/search/",
            json_body={
                "query": "deploy workflow",
                "filters": {"AND": [{"user_id": "hermes-agent"}, {"agent_id": "hermes-agent"}]},
                "rerank": True,
                "top_k": 5,
            },
            timeout=8.0,
        )

    def test_add_messages_payload(self):
        client = _Client("test-key", _DEFAULT_BASE_URL)
        with patch.object(client, "request", return_value={"results": []}) as mock_request:
            client.add_messages(
                [{"role": "user", "content": "fact"}],
                user_id="hermes-agent",
                agent_id="hermes-agent",
                version="v2",
                infer=False,
            )
        mock_request.assert_called_once_with(
            "POST",
            "/v1/memories/",
            json_body={
                "messages": [{"role": "user", "content": "fact"}],
                "user_id": "hermes-agent",
                "agent_id": "hermes-agent",
                "version": "v2",
                "infer": False,
            },
            timeout=10.0,
        )

    def test_get_all_tries_trailing_slash_first(self):
        client = _Client("test-key", _DEFAULT_BASE_URL)

        def fake_request(method, path, **kwargs):
            if path == "/v1/memories/":
                raise RuntimeError("404")
            return {"results": []}

        with patch.object(client, "request", side_effect=fake_request) as mock_request:
            result = client.get_all(
                filters={"user_id": "hermes-agent", "agent_id": "hermes-agent"},
                version="v2",
            )
        assert result == {"results": []}
        assert mock_request.call_count == 2


class TestConfig:
    def test_load_config_defaults(self):
        cfg = _load_config()
        assert cfg["base_url"] == _DEFAULT_BASE_URL
        assert cfg["user_id"] == "hermes-user"
        assert cfg["agent_id"] == "hermes"

    def test_load_config_from_env(self, monkeypatch):
        monkeypatch.setenv("MEM0_API_KEY", "key")
        monkeypatch.setenv("MEM0_BASE_URL", "https://mem0.example")
        monkeypatch.setenv("MEM0_USER_ID", "hermes-agent")
        monkeypatch.setenv("MEM0_AGENT_ID", "hermes-agent")
        cfg = _load_config()
        assert cfg["api_key"] == "key"
        assert cfg["base_url"] == "https://mem0.example"
        assert cfg["user_id"] == "hermes-agent"
        assert cfg["agent_id"] == "hermes-agent"


@pytest.fixture()
def provider(monkeypatch):
    monkeypatch.setenv("MEM0_API_KEY", "key")
    monkeypatch.setenv("MEM0_USER_ID", "hermes-agent")
    monkeypatch.setenv("MEM0_AGENT_ID", "hermes-agent")
    p = Mem0HttpMemoryProvider()
    p.initialize(session_id="sess-1")
    p._client = MagicMock()
    return p


class TestProvider:
    def test_is_available_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        assert Mem0HttpMemoryProvider().is_available() is False
        monkeypatch.setenv("MEM0_API_KEY", "key")
        assert Mem0HttpMemoryProvider().is_available() is True

    def test_system_prompt_includes_scope(self, provider):
        block = provider.system_prompt_block()
        assert "hermes-agent" in block
        assert "Mem0 HTTP Memory" in block

    def test_profile_uses_scoped_get_all(self, provider):
        provider._client.get_all.return_value = {
            "results": [{"memory": "Hermes uses isolated memory."}]
        }
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert result["count"] == 1
        provider._client.get_all.assert_called_once_with(
            filters={"user_id": "hermes-agent", "agent_id": "hermes-agent"},
            version="v2",
        )

    def test_search_uses_scoped_filters(self, provider):
        provider._client.search.return_value = {
            "results": [{"memory": "Deploy workflow", "score": 0.9}]
        }
        result = json.loads(
            provider.handle_tool_call("mem0_search", {"query": "deploy", "top_k": 3})
        )
        assert result["count"] == 1
        provider._client.search.assert_called_once_with(
            query="deploy",
            filters={"user_id": "hermes-agent", "agent_id": "hermes-agent"},
            rerank=False,
            top_k=3,
        )

    def test_conclude_stores_verbatim_fact(self, provider):
        result = json.loads(
            provider.handle_tool_call("mem0_conclude", {"conclusion": "Hermes uses HTTP Mem0."})
        )
        assert result["result"] == "Fact stored."
        provider._client.add_messages.assert_called_once_with(
            [{"role": "user", "content": "Hermes uses HTTP Mem0."}],
            user_id="hermes-agent",
            agent_id="hermes-agent",
            version="v2",
            infer=False,
        )

    def test_sync_turn_writes_messages(self, provider):
        provider.sync_turn("hi", "hello")
        provider.shutdown()
        provider._client.add_messages.assert_called_once_with(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            user_id="hermes-agent",
            agent_id="hermes-agent",
            version="v2",
        )

    def test_register_calls_register_memory_provider(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_memory_provider.assert_called_once()
        provider = ctx.register_memory_provider.call_args[0][0]
        assert isinstance(provider, Mem0HttpMemoryProvider)
