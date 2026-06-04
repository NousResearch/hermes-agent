"""Structured-output (response_format / json_schema) support on the API server.

Covers the fix for #33864: the gateway must *forward* a requested structured
output schema to the backend model — on both ``/v1/chat/completions``
(``response_format``) and ``/v1/responses`` (``text.format``) — and reject
up-front when the configured backend cannot honour it, rather than silently
returning plain text.

Three layers are exercised:
  1. The pure normalizers that translate each endpoint's request shape into the
     shared ``response_format`` payload.
  2. The HTTP handlers, which must forward the payload to ``_run_agent`` and
     return a clear 400 for unsupported backends / malformed input.
  3. ``build_api_kwargs``, which must attach ``response_format`` to the
     chat.completions call when the agent carries a gateway schema.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _normalize_response_format,
    _response_format_from_text_format,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Pure-function tests — _normalize_response_format (Chat Completions shape)
# ---------------------------------------------------------------------------


class TestNormalizeResponseFormat:
    def test_none_returns_no_format(self):
        assert _normalize_response_format(None) == (None, None)

    def test_text_type_is_plain(self):
        assert _normalize_response_format({"type": "text"}) == (None, None)

    def test_json_object(self):
        norm, err = _normalize_response_format({"type": "json_object"})
        assert err is None
        assert norm == {"type": "json_object"}

    def test_json_schema_full(self):
        norm, err = _normalize_response_format({
            "type": "json_schema",
            "json_schema": {
                "name": "Probe",
                "schema": {"type": "object", "properties": {"word": {"type": "string"}}},
                "strict": True,
            },
        })
        assert err is None
        assert norm == {
            "type": "json_schema",
            "json_schema": {
                "name": "Probe",
                "schema": {"type": "object", "properties": {"word": {"type": "string"}}},
                "strict": True,
            },
        }

    def test_json_schema_defaults_name(self):
        norm, err = _normalize_response_format({
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}},
        })
        assert err is None
        assert norm["json_schema"]["name"] == "response"
        assert "strict" not in norm["json_schema"]

    def test_json_schema_missing_schema_errors(self):
        norm, err = _normalize_response_format({"type": "json_schema", "json_schema": {"name": "x"}})
        assert norm is None
        assert err and "schema" in err

    def test_non_dict_errors(self):
        norm, err = _normalize_response_format("json")
        assert norm is None
        assert err is not None

    def test_unknown_type_errors(self):
        norm, err = _normalize_response_format({"type": "xml"})
        assert norm is None
        assert err and "xml" in err


# ---------------------------------------------------------------------------
# Pure-function tests — _response_format_from_text_format (Responses shape)
# ---------------------------------------------------------------------------


class TestResponseFormatFromTextFormat:
    def test_none(self):
        assert _response_format_from_text_format(None) == (None, None)

    def test_no_format_key(self):
        assert _response_format_from_text_format({}) == (None, None)

    def test_text_format(self):
        assert _response_format_from_text_format({"format": {"type": "text"}}) == (None, None)

    def test_json_object(self):
        norm, err = _response_format_from_text_format({"format": {"type": "json_object"}})
        assert err is None
        assert norm == {"type": "json_object"}

    def test_json_schema_converted(self):
        # Responses API nests name/schema/strict directly under `format`.
        norm, err = _response_format_from_text_format({
            "format": {
                "type": "json_schema",
                "name": "Probe",
                "schema": {"type": "object", "properties": {"word": {"type": "string"}}},
                "strict": True,
            }
        })
        assert err is None
        assert norm == {
            "type": "json_schema",
            "json_schema": {
                "name": "Probe",
                "schema": {"type": "object", "properties": {"word": {"type": "string"}}},
                "strict": True,
            },
        }

    def test_json_schema_missing_schema_errors(self):
        norm, err = _response_format_from_text_format({"format": {"type": "json_schema", "name": "x"}})
        assert norm is None
        assert err is not None


# ---------------------------------------------------------------------------
# HTTP integration — forwarding to _run_agent + rejection paths
# ---------------------------------------------------------------------------


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def _stub_run_agent(mock_run):
    async def _stub(**kwargs):
        mock_run.captured = kwargs
        return (
            {"final_response": "ok", "messages": [], "api_calls": 1},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    mock_run.side_effect = _stub


_SCHEMA = {
    "type": "object",
    "properties": {"word": {"type": "string"}},
    "required": ["word"],
    "additionalProperties": False,
}


class TestChatCompletionsStructuredOutput:
    @pytest.mark.asyncio
    async def test_response_format_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_mode": "chat_completions"}), \
                 patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                _stub_run_agent(mock_run)
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "Reply with PONG"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "Probe", "schema": _SCHEMA},
                    },
                })
            assert resp.status == 200, await resp.text()
            assert mock_run.captured["response_format"] == {
                "type": "json_schema",
                "json_schema": {"name": "Probe", "schema": _SCHEMA},
            }

    @pytest.mark.asyncio
    async def test_no_response_format_forwards_none(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                _stub_run_agent(mock_run)
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hi"}],
                })
            assert resp.status == 200, await resp.text()
            assert mock_run.captured["response_format"] is None

    @pytest.mark.asyncio
    async def test_malformed_response_format_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
            })
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["param"] == "response_format"

    @pytest.mark.asyncio
    async def test_json_object_unsupported_on_anthropic_returns_400(self, adapter):
        # json_object has no Anthropic output_config expression — reject up-front
        # rather than silently returning unconstrained text.
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_mode": "anthropic_messages"}):
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"},
                })
            assert resp.status == 400
            body = await resp.json()
        assert "anthropic_messages" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_unsupported_backend_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_mode": "bedrock_converse"}):
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"},
                })
            assert resp.status == 400
            body = await resp.json()
        assert "bedrock_converse" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_anthropic_json_schema_allowed(self, adapter):
        # json_schema on anthropic_messages is supported — the guard must not
        # reject it; it maps to output_config.format downstream.
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_mode": "anthropic_messages"}), \
                 patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                _stub_run_agent(mock_run)
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "Reply with PONG"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "Probe", "schema": _SCHEMA},
                    },
                })
            assert resp.status == 200, await resp.text()
            assert mock_run.captured["response_format"] == {
                "type": "json_schema",
                "json_schema": {"name": "Probe", "schema": _SCHEMA},
            }


class TestResponsesStructuredOutput:
    @pytest.mark.asyncio
    async def test_text_format_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_mode": "chat_completions"}), \
                 patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                _stub_run_agent(mock_run)
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent",
                    "input": "Reply with PONG",
                    "text": {"format": {"type": "json_schema", "name": "Probe", "schema": _SCHEMA}},
                })
            assert resp.status == 200, await resp.text()
            assert mock_run.captured["response_format"] == {
                "type": "json_schema",
                "json_schema": {"name": "Probe", "schema": _SCHEMA},
            }

    @pytest.mark.asyncio
    async def test_plain_input_forwards_none(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                _stub_run_agent(mock_run)
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent",
                    "input": "hi",
                })
            assert resp.status == 200, await resp.text()
            assert mock_run.captured["response_format"] is None

    @pytest.mark.asyncio
    async def test_unsupported_backend_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_mode": "bedrock_converse"}):
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent",
                    "input": "hi",
                    "text": {"format": {"type": "json_schema", "name": "Probe", "schema": _SCHEMA}},
                })
            assert resp.status == 400
            body = await resp.json()
        assert "bedrock_converse" in body["error"]["message"]


# ---------------------------------------------------------------------------
# build_api_kwargs — response_format reaches the chat.completions call
# ---------------------------------------------------------------------------


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_chat_agent(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


class TestBuildApiKwargsInjection:
    def test_response_format_attached_when_set(self, monkeypatch):
        agent = _make_chat_agent(monkeypatch)
        agent._gateway_response_format = {
            "type": "json_schema",
            "json_schema": {"name": "Probe", "schema": _SCHEMA},
        }
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "Probe", "schema": _SCHEMA},
        }

    def test_no_response_format_when_unset(self, monkeypatch):
        agent = _make_chat_agent(monkeypatch)
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "response_format" not in kwargs
