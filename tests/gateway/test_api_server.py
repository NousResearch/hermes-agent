"""
Tests for the OpenAI-compatible API server gateway adapter.

Tests cover:
- Chat Completions endpoint (request parsing, response format)
- Responses API endpoint (request parsing, response format)
- previous_response_id chaining (store/retrieve)
- Auth (valid key, invalid key, no key configured)
- /v1/models endpoint
- /health endpoint
- System prompt extraction
- Error handling (invalid JSON, missing fields)
"""

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _IdempotencyCache,
    _CORS_HEADERS,
    _derive_chat_session_id,
    check_api_server_requirements,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# check_api_server_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:
    def test_returns_true_when_aiohttp_available(self):
        assert check_api_server_requirements() is True

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_api_server_requirements() is False


# ---------------------------------------------------------------------------
# ResponseStore
# ---------------------------------------------------------------------------


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Adding a 4th should evict resp_1
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert len(store) == 3

    def test_access_refreshes_lru(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Access resp_1 to move it to end
        store.get("resp_1")
        # Now resp_2 is the oldest — adding a 4th should evict resp_2
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_2") is None
        assert store.get("resp_1") is not None

    def test_update_existing_key(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "v1"})
        store.put("resp_1", {"output": "v2"})
        assert store.get("resp_1") == {"output": "v2"}
        assert len(store) == 1

    def test_delete_existing(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.delete("resp_1") is True
        assert store.get("resp_1") is None
        assert len(store) == 0

    def test_delete_missing(self):
        store = ResponseStore(max_size=10)
        assert store.delete("resp_missing") is False


# ---------------------------------------------------------------------------
# _IdempotencyCache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    @pytest.mark.asyncio
    async def test_concurrent_same_key_and_fingerprint_runs_once(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return ("response", {"total_tokens": 1})

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        gate.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == second_result == ("response", {"total_tokens": 1})

    @pytest.mark.asyncio
    async def test_different_fingerprint_does_not_reuse_inflight_task(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            result = calls
            if calls == 2:
                started.set()
            await gate.wait()
            return result

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-2", compute))

        await started.wait()
        assert calls == 2

        gate.set()
        results = await asyncio.gather(first, second)

        assert sorted(results) == [1, 2]

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_drop_shared_inflight_task(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return "response"

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        await asyncio.sleep(0)
        assert calls == 1

        gate.set()
        assert await second == "response"


# ---------------------------------------------------------------------------
# Adapter initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_default_config(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8642
        assert adapter._api_key == ""
        assert adapter.platform == Platform.API_SERVER

    def test_custom_config_from_extra(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "host": "0.0.0.0",
                "port": 9999,
                "key": "sk-test",
                "cors_origins": ["http://localhost:3000"],
            },
        )
        adapter = APIServerAdapter(config)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 9999
        assert adapter._api_key == "sk-test"
        assert adapter._cors_origins == ("http://localhost:3000",)

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_HOST", "10.0.0.1")
        monkeypatch.setenv("API_SERVER_PORT", "7777")
        monkeypatch.setenv("API_SERVER_KEY", "sk-env")
        monkeypatch.setenv("API_SERVER_CORS_ORIGINS", "http://localhost:3000, http://127.0.0.1:3000")
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "10.0.0.1"
        assert adapter._port == 7777
        assert adapter._api_key == "sk-env"
        assert adapter._cors_origins == (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        )


# ---------------------------------------------------------------------------
# Auth checking
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_configured_allows_all(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        assert adapter._check_auth(mock_request) is None

    def test_valid_key_passes(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer sk-test123"}
        assert adapter._check_auth(mock_request) is None

    def test_invalid_key_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401

    def test_missing_auth_header_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401

    def test_malformed_auth_header_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401


# ---------------------------------------------------------------------------
# Helpers for HTTP tests
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    """Create an adapter with optional API key.

    ``cors_origins`` defaults to the empty string so tests are hermetic
    and don't inherit ``API_SERVER_CORS_ORIGINS`` from the shell
    environment (which would silently enable CORS for whatever origins
    the developer happens to have configured locally).  Pass an explicit
    list/tuple/string to enable CORS for a test.
    """
    extra = {}
    if api_key:
        extra["key"] = api_key
    extra["cors_origins"] = "" if cors_origins is None else cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app from the adapter (without starting the full server)."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_get("/v1/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_get(
        "/v1/responses/{response_id}/events",
        adapter._handle_get_response_events,
    )
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_security_headers_present(self, adapter):
        """Responses should include basic security headers."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("Referrer-Policy") == "no-referrer"

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "hermes-agent"

    @pytest.mark.asyncio
    async def test_v1_health_alias_returns_ok(self, adapter):
        """GET /v1/health should return the same response as /health."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "hermes-agent"


# ---------------------------------------------------------------------------
# /health/detailed endpoint
# ---------------------------------------------------------------------------


class TestHealthDetailedEndpoint:
    @pytest.mark.asyncio
    async def test_health_detailed_returns_ok(self, adapter):
        """GET /health/detailed returns status, platform, and runtime fields."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "active_agents": 2,
            "exit_reason": None,
            "updated_at": "2026-04-14T00:00:00Z",
        }):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["platform"] == "hermes-agent"
                assert data["gateway_state"] == "running"
                assert data["platforms"] == {"telegram": {"state": "connected"}}
                assert data["active_agents"] == 2
                assert isinstance(data["pid"], int)
                assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_health_detailed_no_runtime_status(self, adapter):
        """When gateway_state.json is missing, fields are None."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value=None):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["gateway_state"] is None
                assert data["platforms"] == {}

    @pytest.mark.asyncio
    async def test_health_detailed_does_not_require_auth(self, auth_adapter):
        """Health detailed endpoint should be accessible without auth, like /health."""
        app = _create_app(auth_adapter)
        with patch("gateway.status.read_runtime_status", return_value=None):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200


# ---------------------------------------------------------------------------
# /v1/models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_hermes_agent(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "hermes-agent"
            assert data["data"][0]["owned_by"] == "hermes"

    @pytest.mark.asyncio
    async def test_models_returns_profile_name(self):
        """When running under a named profile, /v1/models advertises the profile name."""
        with patch("gateway.platforms.api_server.APIServerAdapter._resolve_model_name", return_value="lucas"):
            adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["data"][0]["id"] == "lucas"
            assert data["data"][0]["root"] == "lucas"

    @pytest.mark.asyncio
    async def test_models_returns_explicit_model_name(self):
        """Explicit model_name in config overrides profile name."""
        extra = {"model_name": "my-custom-agent"}
        config = PlatformConfig(enabled=True, extra=extra)
        adapter = APIServerAdapter(config)
        assert adapter._model_name == "my-custom-agent"

    def test_resolve_model_name_explicit(self):
        assert APIServerAdapter._resolve_model_name("my-bot") == "my-bot"

    def test_resolve_model_name_default_profile(self):
        """Default profile falls back to 'hermes-agent'."""
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            assert APIServerAdapter._resolve_model_name("") == "hermes-agent"

    def test_resolve_model_name_named_profile(self):
        """Named profile uses the profile name as model name."""
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="lucas"):
            assert APIServerAdapter._resolve_model_name("") == "lucas"

    @pytest.mark.asyncio
    async def test_models_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_models_with_valid_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/v1/models",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200


# ---------------------------------------------------------------------------
# /v1/chat/completions endpoint
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert "Invalid JSON" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "messages" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test", "messages": []})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_stream_true_returns_sse(self, adapter):
        """stream=true returns SSE format with the full response."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                # Simulate streaming: invoke stream_delta_callback with tokens
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello!")
                    cb(None)  # End signal
                return (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                assert resp.headers.get("X-Accel-Buffering") == "no"
                body = await resp.text()
                assert "data: " in body
                assert "[DONE]" in body
                assert "Hello!" in body

    @pytest.mark.asyncio
    async def test_stream_sends_keepalive_during_quiet_tool_gap(self, adapter):
        """Idle SSE streams should send keepalive comments while tools run silently."""
        import asyncio
        import gateway.platforms.api_server as api_server_mod

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Working")
                    await asyncio.sleep(0.65)
                    cb("...done")
                return (
                    {"final_response": "Working...done", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with (
                patch.object(api_server_mod, "CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS", 0.01),
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "do the thing"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert ": keepalive" in body
                assert "Working" in body
                assert "...done" in body
                assert "[DONE]" in body

    @pytest.mark.asyncio
    async def test_stream_survives_tool_call_none_sentinel(self, adapter):
        """stream_delta_callback(None) mid-stream (tool calls) must NOT kill the SSE stream.

        The agent fires stream_delta_callback(None) to tell the CLI display to
        close its response box before executing tool calls.  The API server's
        _on_delta must filter this out so the SSE response stays open and the
        final answer (streamed after tool execution) reaches the client.
        """
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    # Simulate: agent streams partial text, then fires None
                    # (tool call box-close signal), then streams the final answer
                    cb("Thinking")
                    cb(None)          # mid-stream None from tool calls
                    await asyncio.sleep(0.05)  # simulate tool execution delay
                    cb(" about it...")
                    cb(None)          # another None (possible second tool round)
                    await asyncio.sleep(0.05)
                    cb(" The answer is 42.")
                return (
                    {"final_response": "Thinking about it... The answer is 42.", "messages": [], "api_calls": 3},
                    {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "What is the answer?"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # The final answer text must appear in the SSE stream
                assert "The answer is 42." in body
                # All partial text must be present too
                assert "Thinking" in body
                assert " about it..." in body

    @pytest.mark.asyncio
    async def test_stream_includes_tool_progress(self, adapter):
        """tool_progress_callback fires → progress appears as custom SSE event, not in delta.content."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                tp_cb = kwargs.get("tool_progress_callback")
                # Simulate tool progress before streaming content
                if tp_cb:
                    tp_cb("tool.started", "terminal", "ls -la", {"command": "ls -la"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Here are the files.")
                return (
                    {"final_response": "Here are the files.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list files"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # Tool progress must appear as a custom SSE event, not in
                # delta.content — prevents model from learning to imitate
                # markers instead of calling tools (#6972).
                assert "event: hermes.tool.progress" in body
                assert '"tool": "terminal"' in body
                assert '"label": "ls -la"' in body
                # The progress marker must NOT appear inside any
                # chat.completion.chunk delta.content field.
                import json as _json
                for line in body.splitlines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk = _json.loads(line[len("data: "):])
                        except _json.JSONDecodeError:
                            continue
                        if chunk.get("object") == "chat.completion.chunk":
                            for choice in chunk.get("choices", []):
                                content = choice.get("delta", {}).get("content", "")
                                # Tool emoji markers must never leak into content
                                assert "ls -la" not in content or content == "Here are the files."
                # Final content must also be present
                assert "Here are the files." in body

    @pytest.mark.asyncio
    async def test_stream_tool_progress_skips_internal_events(self, adapter):
        """Internal events (name starting with _) are not streamed."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                tp_cb = kwargs.get("tool_progress_callback")
                if tp_cb:
                    tp_cb("tool.started", "_thinking", "some internal state", {})
                    tp_cb("tool.started", "web_search", "Python docs", {"query": "Python docs"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Found it.")
                return (
                    {"final_response": "Found it.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "search"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                # Internal _thinking event should NOT appear anywhere
                assert "some internal state" not in body
                # Real tool progress should appear as custom SSE event
                assert "event: hermes.tool.progress" in body
                assert '"tool": "web_search"' in body
                assert '"label": "Python docs"' in body

    @pytest.mark.asyncio
    async def test_no_user_message_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "system", "content": "You are helpful."}],
                },
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_successful_completion(self, adapter):
        """Test a successful chat completion with mocked agent."""
        mock_result = {
            "final_response": "Hello! How can I help you today?",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "chat.completion"
            assert data["id"].startswith("chatcmpl-")
            assert data["model"] == "hermes-agent"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert "usage" in data

    @pytest.mark.asyncio
    async def test_system_prompt_extracted(self, adapter):
        """System messages from the client are passed as ephemeral_system_prompt."""
        mock_result = {
            "final_response": "I am a pirate! Arrr!",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are a pirate."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            # Check that _run_agent was called with the system prompt
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("ephemeral_system_prompt") == "You are a pirate."
            assert call_kwargs.kwargs.get("user_message") == "Hello"

    @pytest.mark.asyncio
    async def test_conversation_history_passed(self, adapter):
        """Previous user/assistant messages become conversation_history."""
        mock_result = {"final_response": "3", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "1+1=?"},
                            {"role": "assistant", "content": "2"},
                            {"role": "user", "content": "Now add 1 more"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["user_message"] == "Now add 1 more"
            assert len(call_kwargs["conversation_history"]) == 2
            assert call_kwargs["conversation_history"][0] == {"role": "user", "content": "1+1=?"}
            assert call_kwargs["conversation_history"][1] == {"role": "assistant", "content": "2"}

    @pytest.mark.asyncio
    async def test_agent_error_returns_500(self, adapter):
        """Agent exception returns 500."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = RuntimeError("Provider failed")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 500
            data = await resp.json()
            assert "Provider failed" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_stable_session_id_across_turns(self, adapter):
        """Same conversation (same first user message) produces the same session_id."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        session_ids = []
        async with TestClient(TestServer(app)) as cli:
            # Turn 1: single user message
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                session_ids.append(mock_run.call_args.kwargs["session_id"])

            # Turn 2: same first message, conversation grew
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                            {"role": "user", "content": "How are you?"},
                        ],
                    },
                )
                session_ids.append(mock_run.call_args.kwargs["session_id"])

        assert session_ids[0] == session_ids[1], "Session ID should be stable across turns"
        assert session_ids[0].startswith("api-"), "Derived session IDs should have api- prefix"

    @pytest.mark.asyncio
    async def test_different_conversations_get_different_session_ids(self, adapter):
        """Different first messages produce different session_ids."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        session_ids = []
        async with TestClient(TestServer(app)) as cli:
            for first_msg in ["Hello", "Goodbye"]:
                with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                    await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-agent",
                            "messages": [{"role": "user", "content": first_msg}],
                        },
                    )
                    session_ids.append(mock_run.call_args.kwargs["session_id"])

        assert session_ids[0] != session_ids[1]


# ---------------------------------------------------------------------------
# _derive_chat_session_id unit tests
# ---------------------------------------------------------------------------


class TestDeriveChatSessionId:
    def test_deterministic(self):
        """Same inputs always produce the same session ID."""
        a = _derive_chat_session_id("sys", "hello")
        b = _derive_chat_session_id("sys", "hello")
        assert a == b

    def test_prefix(self):
        assert _derive_chat_session_id(None, "hi").startswith("api-")

    def test_different_system_prompt(self):
        a = _derive_chat_session_id("You are a pirate.", "Hello")
        b = _derive_chat_session_id("You are a robot.", "Hello")
        assert a != b

    def test_different_first_message(self):
        a = _derive_chat_session_id(None, "Hello")
        b = _derive_chat_session_id(None, "Goodbye")
        assert a != b

    def test_none_system_prompt(self):
        """None system prompt doesn't crash."""
        sid = _derive_chat_session_id(None, "test")
        assert isinstance(sid, str) and len(sid) > 4


# ---------------------------------------------------------------------------
# /v1/responses endpoint
# ---------------------------------------------------------------------------


class TestResponsesEndpoint:
    @pytest.mark.asyncio
    async def test_missing_input_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_successful_response_with_string_input(self, adapter):
        """String input is wrapped in a user message."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "What is the capital of France?",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "response"
            assert data["id"].startswith("resp_")
            assert data["status"] == "completed"
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["content"][0]["type"] == "output_text"
            assert data["output"][0]["content"][0]["text"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_successful_response_with_array_input(self, adapter):
        """Array input with role/content objects."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {"role": "user", "content": "Hello"},
                            {"role": "user", "content": "What is 2+2?"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # Last message is user_message, rest are history
            assert call_kwargs["user_message"] == "What is 2+2?"
            assert len(call_kwargs["conversation_history"]) == 1

    @pytest.mark.asyncio
    async def test_instructions_as_ephemeral_prompt(self, adapter):
        """The instructions field maps to ephemeral_system_prompt."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Talk like a pirate.",
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Talk like a pirate."

    @pytest.mark.asyncio
    async def test_previous_response_id_chaining(self, adapter):
        """Test that responses can be chained via previous_response_id."""
        mock_result_1 = {
            "final_response": "2",
            "messages": [{"role": "assistant", "content": "2"}],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_1, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 1+1?"},
                )

            assert resp1.status == 200
            data1 = await resp1.json()
            response_id = data1["id"]

            # Second request chaining from the first
            mock_result_2 = {
                "final_response": "3",
                "messages": [{"role": "assistant", "content": "3"}],
                "api_calls": 1,
            }

            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_2, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": response_id,
                    },
                )

            assert resp2.status == 200
            # The conversation_history should contain the full history from the first response
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["conversation_history"]) > 0
            assert call_kwargs["user_message"] == "Now add 1 more"

    @pytest.mark.asyncio
    async def test_previous_response_id_preserves_session(self, adapter):
        """Chained responses via previous_response_id reuse the same session_id."""
        mock_result = {
            "final_response": "ok",
            "messages": [{"role": "assistant", "content": "ok"}],
            "api_calls": 1,
        }
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request — establishes a session
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )
            assert resp1.status == 200
            first_session_id = mock_run.call_args.kwargs["session_id"]
            data1 = await resp1.json()
            response_id = data1["id"]

            # Second request — chains from the first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Follow up",
                        "previous_response_id": response_id,
                    },
                )
            assert resp2.status == 200
            second_session_id = mock_run.call_args.kwargs["session_id"]

            # Session must be the same across the chain
            assert first_session_id == second_session_id

    @pytest.mark.asyncio
    async def test_invalid_previous_response_id_returns_404(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "follow up",
                    "previous_response_id": "resp_nonexistent",
                },
            )
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_store_false_does_not_store(self, adapter):
        """When store=false, the response is NOT stored."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "store": False,
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            # The response has an ID but it shouldn't be retrievable
            assert adapter._response_store.get(data["id"]) is None

    @pytest.mark.asyncio
    async def test_instructions_inherited_from_previous(self, adapter):
        """If no instructions provided, carry forward from previous response."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request with instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Be a pirate",
                    },
                )

            data1 = await resp1.json()
            resp_id = data1["id"]

            # Second request without instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Tell me more",
                        "previous_response_id": resp_id,
                    },
                )

            assert resp2.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Be a pirate"

    @pytest.mark.asyncio
    async def test_agent_error_returns_500(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = RuntimeError("Boom")
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )

            assert resp.status == 500

    @pytest.mark.asyncio
    async def test_invalid_input_type_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "hermes-agent", "input": 42},
            )
            assert resp.status == 400


class TestResponsesStreaming:
    @pytest.mark.asyncio
    async def test_stream_true_returns_responses_sse(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello")
                    cb(" world")
                return (
                    {"final_response": "Hello world", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                body = await resp.text()
                assert "event: response.created" in body
                assert "event: response.output_text.delta" in body
                assert "event: response.output_text.done" in body
                assert "event: response.completed" in body
                assert '"sequence_number":' in body
                assert '"logprobs": []' in body
                assert "Hello" in body
                assert " world" in body

    @pytest.mark.asyncio
    async def test_stream_emits_function_call_and_output_items(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                complete_cb = kwargs.get("tool_complete_callback")
                text_cb = kwargs.get("stream_delta_callback")
                if start_cb:
                    start_cb("call_123", "read_file", {"path": "/tmp/test.txt"})
                if complete_cb:
                    complete_cb("call_123", "read_file", {"path": "/tmp/test.txt"}, '{"content":"hello"}')
                if text_cb:
                    text_cb("Done.")
                return (
                    {
                        "final_response": "Done.",
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"/tmp/test.txt"}',
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "call_123",
                                "content": '{"content":"hello"}',
                            },
                        ],
                        "api_calls": 1,
                    },
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "read the file", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()
                assert "event: response.output_item.added" in body
                assert "event: response.output_item.done" in body
                assert body.count("event: response.output_item.done") >= 2
                assert '"type": "function_call"' in body
                assert '"type": "function_call_output"' in body
                assert '"call_id": "call_123"' in body
                assert '"name": "read_file"' in body
                assert '"output": [{"type": "input_text", "text": "{\\"content\\":\\"hello\\"}"}]' in body

    @pytest.mark.asyncio
    async def test_streamed_response_is_stored_for_get(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Stored response")
                return (
                    {"final_response": "Stored response", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "store this", "stream": True},
                )
                body = await resp.text()
                response_id = None
                for line in body.splitlines():
                    if line.startswith("data: "):
                        try:
                            payload = json.loads(line[len("data: "):])
                        except json.JSONDecodeError:
                            continue
                        if payload.get("type") == "response.completed":
                            response_id = payload["response"]["id"]
                            break
                assert response_id

                get_resp = await cli.get(f"/v1/responses/{response_id}")
                assert get_resp.status == 200
                data = await get_resp.json()
                assert data["id"] == response_id
                assert data["status"] == "completed"
                assert data["output"][-1]["content"][0]["text"] == "Stored response"

    @pytest.mark.asyncio
    async def test_stream_cancelled_persists_incomplete_snapshot(self, adapter):
        """Server-side asyncio.CancelledError (shutdown, request timeout) must
        still leave an ``incomplete`` snapshot in ResponseStore so
        GET /v1/responses/{id} and previous_response_id chaining keep
        working.  Regression for PR #15171 follow-up.

        Calls _write_sse_responses directly so the test can await the
        handler to completion (TestClient disconnection races the server
        handler, which makes end-to-end assertion on the final stored
        snapshot flaky).
        """
        # Build a minimal fake request + stream queue the writer understands.
        fake_request = MagicMock()
        fake_request.headers = {}

        written_payloads: list = []

        class _FakeStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                written_payloads.append(payload)

        # Patch web.StreamResponse for the duration of the writer call.
        import gateway.platforms.api_server as api_mod
        import queue as _q

        stream_q: _q.Queue = _q.Queue()

        async def _agent_coro():
            # Feed one partial delta into the stream queue...
            stream_q.put("partial output")
            # ...then give the drain loop a moment to pick it up before
            # raising CancelledError to simulate a server-side cancel.
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_FakeStreamResponse()):
            with pytest.raises(asyncio.CancelledError):
                await adapter._write_sse_responses(
                    request=fake_request,
                    response_id=response_id,
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=[None],
                    conversation_history=[],
                    user_message="will be cancelled",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id=None,
                )

        # The in_progress snapshot was persisted on response.created,
        # and the CancelledError handler must have updated it to
        # ``incomplete`` with the partial text it saw.
        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must be retrievable after cancellation"
        assert stored["response"]["status"] == "incomplete"
        # Partial text captured before cancel should be preserved.
        output_text = "".join(
            part.get("text", "")
            for item in stored["response"].get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        assert "partial output" in output_text

    @pytest.mark.asyncio
    async def test_stream_continues_after_client_disconnect_when_stored(self, adapter):
        """Client disconnect mid-stream must NOT cancel the agent for
        store=True runs. The response should still reach response.completed
        and be persisted to ResponseStore so a later GET recovers it.
        """
        from aiohttp import web as _aw

        original_write = _aw.StreamResponse.write
        disconnect_after = {"count": 0, "trigger": 2}

        async def _flaky_write(self, data, *args, **kwargs):
            disconnect_after["count"] += 1
            if disconnect_after["count"] > disconnect_after["trigger"]:
                raise ConnectionResetError("simulated client disconnect")
            return await original_write(self, data, *args, **kwargs)

        agent_ran_to_completion = asyncio.Event()
        captured_response_id = {"value": None}

        async def _mock_run_agent(**kwargs):
            cb = kwargs.get("stream_delta_callback")
            if cb:
                # Yield control between deltas so the SSE loop has a chance
                # to attempt a write and hit the simulated disconnect.
                cb("Recovered")
                await asyncio.sleep(0)
                cb(" answer")
                await asyncio.sleep(0)
            agent_ran_to_completion.set()
            return (
                {"final_response": "Recovered answer", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent), \
                 patch.object(_aw.StreamResponse, "write", _flaky_write):
                # Fire the request — the stream will abort mid-way server-side
                # but the agent should still run to completion for store=True.
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "will drop",
                        "stream": True,
                        "store": True,
                    },
                )
                # Drain whatever events we did receive so we can record the id.
                try:
                    body = await resp.text()
                except Exception:
                    body = ""
                for line in body.splitlines():
                    if line.startswith("data: "):
                        try:
                            payload = json.loads(line[len("data: "):])
                        except json.JSONDecodeError:
                            continue
                        if payload.get("type") == "response.created":
                            captured_response_id["value"] = payload["response"]["id"]
                            break

            # Wait for the mocked agent to finish; without the fix the task
            # would be cancelled by the disconnect handler and this would
            # time out.
            await asyncio.wait_for(agent_ran_to_completion.wait(), timeout=5.0)

            assert captured_response_id["value"], "response.created must have been seen before the drop"

            # Give the persistence branch a moment to finish writing after the
            # agent task returned.
            for _ in range(50):
                if adapter._response_store.get(captured_response_id["value"]) is not None:
                    break
                await asyncio.sleep(0.05)

            # GET /v1/responses/{id} returns the completed snapshot.
            get_resp = await cli.get(f"/v1/responses/{captured_response_id['value']}")
            assert get_resp.status == 200
            data = await get_resp.json()
            assert data["status"] == "completed"
            assert data["output"][-1]["content"][0]["text"] == "Recovered answer"

    @pytest.mark.asyncio
    async def test_stream_interrupts_agent_on_disconnect_when_not_stored(self, adapter):
        """Legacy behavior for store=False: disconnect still cancels the
        agent so we don't spend tokens on a response no one will read.
        """
        from aiohttp import web as _aw

        original_write = _aw.StreamResponse.write
        # Let a couple of writes through so the mocked agent task has time
        # to populate agent_ref[0] with our fake before the drop fires.
        disconnect_after = {"count": 0, "trigger": 2}

        async def _flaky_write(self, data, *args, **kwargs):
            disconnect_after["count"] += 1
            if disconnect_after["count"] > disconnect_after["trigger"]:
                raise ConnectionResetError("simulated client disconnect")
            return await original_write(self, data, *args, **kwargs)

        interrupted = asyncio.Event()

        async def _mock_run_agent(**kwargs):
            agent_ref = kwargs.get("agent_ref")

            class _FakeAgent:
                def interrupt(self, _reason):
                    interrupted.set()

            if isinstance(agent_ref, list):
                agent_ref[0] = _FakeAgent()
            # Push one delta so the SSE loop does a second write and crosses
            # the disconnect trigger.
            cb = kwargs.get("stream_delta_callback")
            if cb:
                cb("hi")
            # Sleep long enough that cancellation actually wins the race.
            await asyncio.sleep(5)
            return (
                {"final_response": "never sent", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent), \
                 patch.object(_aw.StreamResponse, "write", _flaky_write):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "drop me",
                        "stream": True,
                        "store": False,
                    },
                )
                try:
                    await resp.read()
                except Exception:
                    pass

            # interrupt() must have been invoked on the agent.
            await asyncio.wait_for(interrupted.wait(), timeout=5.0)


# ---------------------------------------------------------------------------
# Auth on endpoints
# ---------------------------------------------------------------------------


class TestEndpointAuth:
    @pytest.mark.asyncio
    async def test_chat_completions_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_responses_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "test", "input": "hi"},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_models_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_health_does_not_require_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_platform_enum_has_api_server(self):
        assert Platform.API_SERVER.value == "api_server"

    def test_env_override_enables_api_server(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert Platform.API_SERVER in config.platforms
        assert config.platforms[Platform.API_SERVER].enabled is True

    def test_env_override_with_key(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_KEY", "sk-mykey")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert Platform.API_SERVER in config.platforms
        assert config.platforms[Platform.API_SERVER].extra.get("key") == "sk-mykey"

    def test_env_override_port_and_host(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv("API_SERVER_PORT", "9999")
        monkeypatch.setenv("API_SERVER_HOST", "0.0.0.0")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("port") == 9999
        assert config.platforms[Platform.API_SERVER].extra.get("host") == "0.0.0.0"

    def test_env_override_cors_origins(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv(
            "API_SERVER_CORS_ORIGINS",
            "http://localhost:3000, http://127.0.0.1:3000",
        )
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("cors_origins") == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_api_server_in_connected_platforms(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(enabled=True)
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER in connected

    def test_api_server_not_in_connected_when_disabled(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(enabled=False)
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER not in connected


# ---------------------------------------------------------------------------
# Multiple system messages
# ---------------------------------------------------------------------------


class TestMultipleSystemMessages:
    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(self, adapter):
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "system", "content": "Be concise."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            prompt = call_kwargs["ephemeral_system_prompt"]
            assert "You are helpful." in prompt
            assert "Be concise." in prompt


# ---------------------------------------------------------------------------
# send() method (not used but required by base)
# ---------------------------------------------------------------------------


class TestSendMethod:
    @pytest.mark.asyncio
    async def test_send_returns_not_supported(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        result = await adapter.send("chat1", "hello")
        assert result.success is False
        assert "HTTP request/response" in result.error


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_stored_response(self, adapter):
        """GET returns a previously stored response."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Create a response first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            response_id = data["id"]

            # Now GET it
            resp2 = await cli.get(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_not_found(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_get_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_any")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# DELETE /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestDeleteResponse:
    @pytest.mark.asyncio
    async def test_delete_stored_response(self, adapter):
        """DELETE removes a stored response and returns confirmation."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            data = await resp.json()
            response_id = data["id"]

            # Delete it
            resp2 = await cli.delete(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["deleted"] is True

            # Verify it's gone
            resp3 = await cli.get(f"/v1/responses/{response_id}")
            assert resp3.status == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/v1/responses/resp_nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/v1/responses/resp_any")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# Tool calls in output
# ---------------------------------------------------------------------------


class TestToolCallsInOutput:
    @pytest.mark.asyncio
    async def test_tool_calls_in_output(self, adapter):
        """When agent returns tool calls, they appear as function_call items."""
        mock_result = {
            "final_response": "The result is 42.",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "6*7"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "42",
                },
                {
                    "role": "assistant",
                    "content": "The result is 42.",
                },
            ],
            "api_calls": 2,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 6*7?"},
                )

            assert resp.status == 200
            data = await resp.json()
            output = data["output"]

            # Should have: function_call, function_call_output, message
            assert len(output) == 3
            assert output[0]["type"] == "function_call"
            assert output[0]["name"] == "calculator"
            assert output[0]["arguments"] == '{"expression": "6*7"}'
            assert output[0]["call_id"] == "call_abc123"
            assert output[1]["type"] == "function_call_output"
            assert output[1]["call_id"] == "call_abc123"
            assert output[1]["output"] == "42"
            assert output[2]["type"] == "message"
            assert output[2]["content"][0]["text"] == "The result is 42."

    @pytest.mark.asyncio
    async def test_no_tool_calls_still_works(self, adapter):
        """Without tool calls, output is just a message."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"


# ---------------------------------------------------------------------------
# Usage / token counting
# ---------------------------------------------------------------------------


class TestUsageCounting:
    @pytest.mark.asyncio
    async def test_responses_usage(self, adapter):
        """Responses API returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["input_tokens"] == 100
            assert data["usage"]["output_tokens"] == 50
            assert data["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_chat_completions_usage(self, adapter):
        """Chat completions returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["prompt_tokens"] == 200
            assert data["usage"]["completion_tokens"] == 80
            assert data["usage"]["total_tokens"] == 280


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_truncation_auto_limits_history(self, adapter):
        """With truncation=auto, history over 100 messages is trimmed."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        # Pre-seed a stored response with a long history
        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(150)]
        adapter._response_store.put("resp_prev", {
            "response": {"id": "resp_prev", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_prev",
                        "truncation": "auto",
                    },
                )

        assert resp.status == 200
        call_kwargs = mock_run.call_args.kwargs
        # History should be truncated to 100
        assert len(call_kwargs["conversation_history"]) <= 100

    @pytest.mark.asyncio
    async def test_no_truncation_keeps_full_history(self, adapter):
        """Without truncation=auto, long history is passed as-is."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(150)]
        adapter._response_store.put("resp_prev2", {
            "response": {"id": "resp_prev2", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_prev2",
                    },
                )

        assert resp.status == 200
        call_kwargs = mock_run.call_args.kwargs
        assert len(call_kwargs["conversation_history"]) == 150


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_origin_allowed_for_non_browser_client(self, adapter):
        assert adapter._origin_allowed("") is True

    def test_origin_rejected_by_default(self, adapter):
        assert adapter._origin_allowed("http://evil.example") is False

    def test_origin_allowed_for_allowlist_match(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._origin_allowed("http://localhost:3000") is True

    def test_cors_headers_for_origin_disabled_by_default(self, adapter):
        assert adapter._cors_headers_for_origin("http://localhost:3000") is None

    def test_cors_headers_for_origin_matches_allowlist(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        headers = adapter._cors_headers_for_origin("http://localhost:3000")
        assert headers is not None
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert "POST" in headers["Access-Control-Allow-Methods"]

    def test_cors_headers_for_origin_rejects_unknown_origin(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._cors_headers_for_origin("http://evil.example") is None

    @pytest.mark.asyncio
    async def test_cors_headers_not_present_by_default(self, adapter):
        """CORS is disabled unless explicitly configured."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_browser_origin_rejected_by_default(self, adapter):
        """Browser-originated requests are rejected unless explicitly allowed."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://evil.example"})
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_cors_options_preflight_rejected_by_default(self, adapter):
        """Browser preflight is rejected unless CORS is explicitly configured."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://evil.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_cors_headers_present_for_allowed_origin(self):
        """Allowed origins receive explicit CORS headers."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://localhost:3000"})
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
            assert "DELETE" in resp.headers.get("Access-Control-Allow-Methods", "")

    @pytest.mark.asyncio
    async def test_cors_allows_idempotency_key_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Idempotency-Key",
                },
            )
            assert resp.status == 200
            assert "Idempotency-Key" in resp.headers.get("Access-Control-Allow-Headers", "")

    @pytest.mark.asyncio
    async def test_cors_sets_vary_origin_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://localhost:3000"})
            assert resp.status == 200
            assert resp.headers.get("Vary") == "Origin"

    @pytest.mark.asyncio
    async def test_cors_options_preflight_allowed_for_configured_origin(self):
        """Configured origins can complete browser preflight."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "Authorization" in resp.headers.get("Access-Control-Allow-Headers", "")


    @pytest.mark.asyncio
    async def test_cors_preflight_sets_max_age(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Max-Age") == "600"
# ---------------------------------------------------------------------------
# Conversation parameter
# ---------------------------------------------------------------------------


class TestConversationParameter:
    @pytest.mark.asyncio
    async def test_conversation_creates_new(self, adapter):
        """First request with a conversation name works (new conversation)."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post("/v1/responses", json={
                    "input": "hi",
                    "conversation": "my-chat",
                })
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "completed"
                # Conversation mapping should be set
                assert adapter._response_store.get_conversation("my-chat") is not None

    @pytest.mark.asyncio
    async def test_conversation_chains_automatically(self, adapter):
        """Second request with same conversation name chains to first."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "First response", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # First request
                resp1 = await cli.post("/v1/responses", json={
                    "input": "hello",
                    "conversation": "test-conv",
                })
                assert resp1.status == 200
                data1 = await resp1.json()
                resp1_id = data1["id"]

                # Second request — should chain
                mock_run.return_value = (
                    {"final_response": "Second response", "messages": [], "api_calls": 1},
                    {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                )
                resp2 = await cli.post("/v1/responses", json={
                    "input": "follow up",
                    "conversation": "test-conv",
                })
                assert resp2.status == 200

                # The second call should have received conversation history from the first
                assert mock_run.call_count == 2
                second_call_kwargs = mock_run.call_args_list[1]
                history = second_call_kwargs.kwargs.get("conversation_history",
                          second_call_kwargs[1].get("conversation_history", []) if len(second_call_kwargs) > 1 else [])
                # History should be non-empty (contains messages from first response)
                assert len(history) > 0

    @pytest.mark.asyncio
    async def test_conversation_and_previous_response_id_conflict(self, adapter):
        """Cannot use both conversation and previous_response_id."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={
                "input": "hi",
                "conversation": "my-chat",
                "previous_response_id": "resp_abc123",
            })
            assert resp.status == 400
            data = await resp.json()
            assert "Cannot use both" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_separate_conversations_are_isolated(self, adapter):
        """Different conversation names have independent histories."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Response A", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Conversation A
                await cli.post("/v1/responses", json={"input": "conv-a msg", "conversation": "conv-a"})
                # Conversation B
                mock_run.return_value = (
                    {"final_response": "Response B", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "conv-b msg", "conversation": "conv-b"})

                # They should have different response IDs in the mapping
                assert adapter._response_store.get_conversation("conv-a") != adapter._response_store.get_conversation("conv-b")

    @pytest.mark.asyncio
    async def test_conversation_store_false_no_mapping(self, adapter):
        """If store=false, conversation mapping is not updated."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Ephemeral", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post("/v1/responses", json={
                    "input": "hi",
                    "conversation": "ephemeral-chat",
                    "store": False,
                })
                assert resp.status == 200
                # Conversation mapping should NOT be set since store=false
                assert adapter._response_store.get_conversation("ephemeral-chat") is None


# ---------------------------------------------------------------------------
# X-Hermes-Session-Id header (session continuity)
# ---------------------------------------------------------------------------


class TestSessionIdHeader:
    @pytest.mark.asyncio
    async def test_new_session_response_includes_session_id_header(self, adapter):
        """Without X-Hermes-Session-Id, a new session is created and returned in the header."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "Hi"}]},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Hermes-Session-Id") is not None

    @pytest.mark.asyncio
    async def test_provided_session_id_is_used_and_echoed(self, auth_adapter):
        """When X-Hermes-Session-Id is provided, it's passed to the agent and echoed in the response."""
        mock_result = {"final_response": "Continuing!", "messages": [], "api_calls": 1}
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "previous message"},
            {"role": "assistant", "content": "previous reply"},
        ]
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "my-session-123", "Authorization": "Bearer sk-secret"},
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "Continue"}]},
                )

            assert resp.status == 200
            assert resp.headers.get("X-Hermes-Session-Id") == "my-session-123"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["session_id"] == "my-session-123"

    @pytest.mark.asyncio
    async def test_provided_session_id_loads_history_from_db(self, auth_adapter):
        """When X-Hermes-Session-Id is provided, history comes from SessionDB not request body."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = db_history
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "existing-session", "Authorization": "Bearer sk-secret"},
                    # Request body has different history — should be ignored
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "old msg from client"},
                            {"role": "assistant", "content": "old reply from client"},
                            {"role": "user", "content": "new question"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # History must come from DB, not from the request body
            assert call_kwargs["conversation_history"] == db_history
            assert call_kwargs["user_message"] == "new question"

    @pytest.mark.asyncio
    async def test_db_failure_falls_back_to_empty_history(self, auth_adapter):
        """If SessionDB raises, history falls back to empty and request still succeeds."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        # Simulate DB failure: _session_db is None and SessionDB() constructor raises
        auth_adapter._session_db = None
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
                 patch("hermes_state.SessionDB", side_effect=Exception("DB unavailable")):
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "some-session", "Authorization": "Bearer sk-secret"},
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "Hi"}]},
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["conversation_history"] == []
            assert call_kwargs["session_id"] == "some-session"


# ---------------------------------------------------------------------------
# ResponseStore event-log helpers (Phase 2 replay primitives)
# ---------------------------------------------------------------------------


class TestResponseStoreEventLog:
    def test_append_and_get_events_preserves_order(self):
        store = ResponseStore(max_size=10)
        store.append_event("resp_1", 0, "response.created", {"x": 0})
        store.append_event("resp_1", 1, "response.output_text.delta", {"x": 1})
        store.append_event("resp_1", 2, "response.completed", {"x": 2})

        events = store.get_events("resp_1")
        assert [e["sequence_number"] for e in events] == [0, 1, 2]
        assert [e["event_type"] for e in events] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ]
        assert events[1]["data"] == {"x": 1}

    def test_get_events_respects_after_cursor(self):
        store = ResponseStore(max_size=10)
        for seq in range(5):
            store.append_event("resp_1", seq, "e", {"seq": seq})
        tail = store.get_events("resp_1", after=2)
        assert [e["sequence_number"] for e in tail] == [3, 4]

    def test_append_event_is_idempotent_on_duplicate_seq(self):
        store = ResponseStore(max_size=10)
        store.append_event("resp_1", 0, "e", {"a": 1})
        store.append_event("resp_1", 0, "e", {"a": 2})  # duplicate seq
        events = store.get_events("resp_1")
        assert len(events) == 1
        assert events[0]["data"] == {"a": 1}  # first-write wins

    def test_has_events_and_latest_event_sequence(self):
        store = ResponseStore(max_size=10)
        assert store.has_events("resp_x") is False
        assert store.latest_event_sequence("resp_x") is None
        store.append_event("resp_x", 0, "e", {})
        store.append_event("resp_x", 7, "e", {})
        assert store.has_events("resp_x") is True
        assert store.latest_event_sequence("resp_x") == 7

    def test_delete_response_clears_event_log(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "done"})
        store.append_event("resp_1", 0, "e", {})
        store.append_event("resp_1", 1, "e", {})
        assert store.has_events("resp_1") is True
        store.delete("resp_1")
        assert store.has_events("resp_1") is False
        assert store.get("resp_1") is None

    def test_lru_eviction_cascades_to_event_log(self):
        store = ResponseStore(max_size=2)
        store.put("resp_1", {"output": "one"})
        store.append_event("resp_1", 0, "e", {})
        store.put("resp_2", {"output": "two"})
        store.append_event("resp_2", 0, "e", {})
        # Adding a 3rd response evicts resp_1, including its event log
        store.put("resp_3", {"output": "three"})
        assert store.get("resp_1") is None
        assert store.has_events("resp_1") is False
        assert store.has_events("resp_2") is True


# ---------------------------------------------------------------------------
# GET /v1/responses/{id}/events — replay + live-tail endpoint
# ---------------------------------------------------------------------------


def _parse_sse_events(body: str):
    """Parse an SSE body into a list of {event, data} dicts.

    Silently skips keepalive and ``: stream closed`` comment lines as well
    as any ``data:`` line that isn't valid JSON.  Order preserved.
    """
    out = []
    current = {}
    for line in body.splitlines():
        if not line:
            if current:
                out.append(current)
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            raw = line[len("data:"):].strip()
            try:
                current["data"] = json.loads(raw)
            except json.JSONDecodeError:
                current["data"] = None
        # ":" comment lines are ignored.
    if current:
        out.append(current)
    return out


class TestResponsesEventsReplay:
    @pytest.mark.asyncio
    async def test_events_replays_stored_events_after_completion(self, adapter):
        """After a response is fully streamed, GET /events returns every
        stored event in order and closes without hanging on a live tail."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello")
                    cb(" world")
                return (
                    {"final_response": "Hello world", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                post_resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True, "store": True},
                )
                body = await post_resp.text()

            # Pick the response_id out of the response.created event.
            response_id = None
            for evt in _parse_sse_events(body):
                data = evt.get("data") or {}
                if data.get("type") == "response.created":
                    response_id = data["response"]["id"]
                    break
            assert response_id is not None

            # Now replay — the response is already completed.
            replay = await cli.get(f"/v1/responses/{response_id}/events?after=-1")
            assert replay.status == 200
            assert "text/event-stream" in replay.headers.get("Content-Type", "")
            replay_body = await replay.text()
            events = [e for e in _parse_sse_events(replay_body) if e.get("data")]
            # The first replayed event must be response.created with seq=0
            assert events, "at least one replayed event expected"
            first = events[0]
            assert first["data"]["type"] == "response.created"
            assert first["data"]["sequence_number"] == 0
            # Sequence numbers strictly increase by 1 starting from 0.
            seqs = [e["data"]["sequence_number"] for e in events]
            assert seqs == list(range(len(seqs)))
            # Terminal event is response.completed.
            assert events[-1]["data"]["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_events_after_cursor_only_returns_newer_events(self, adapter):
        """?after=N must only send events with sequence_number > N."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello")
                    cb(" world")
                return (
                    {"final_response": "Hello world", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                post_resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True, "store": True},
                )
                body = await post_resp.text()

            # Find the response id and the top sequence number so we can
            # request "give me nothing new" with after=max_seq.
            events = [e for e in _parse_sse_events(body) if e.get("data")]
            response_id = None
            max_seq = -1
            for e in events:
                data = e["data"]
                if data.get("type") == "response.created":
                    response_id = data["response"]["id"]
                seq = data.get("sequence_number")
                if isinstance(seq, int) and seq > max_seq:
                    max_seq = seq
            assert response_id is not None
            assert max_seq >= 0

            # after=max_seq must return zero real events.
            replay = await cli.get(f"/v1/responses/{response_id}/events?after={max_seq}")
            assert replay.status == 200
            replay_body = await replay.text()
            replayed = [e for e in _parse_sse_events(replay_body) if e.get("data")]
            assert replayed == [], f"expected no events, got {replayed!r}"

            # after=0 must return every event with seq > 0 and none with seq == 0.
            replay2 = await cli.get(f"/v1/responses/{response_id}/events?after=0")
            body2 = await replay2.text()
            replayed2 = [e for e in _parse_sse_events(body2) if e.get("data")]
            assert replayed2, "some events should be replayed"
            assert all(e["data"]["sequence_number"] > 0 for e in replayed2)

    @pytest.mark.asyncio
    async def test_events_returns_404_for_unknown_response(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_does_not_exist/events")
            assert resp.status == 404
            data = await resp.json()
            assert data["error"]["code"] == "response_not_found"

    @pytest.mark.asyncio
    async def test_events_rejects_non_integer_after(self, adapter):
        # Pre-populate the store so the id lookup won't 404 first.
        adapter._response_store.put("resp_live", {"response": {"id": "resp_live"}})
        adapter._response_store.append_event("resp_live", 0, "response.created", {})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_live/events?after=not-a-number")
            assert resp.status == 400
            data = await resp.json()
            assert data["error"]["code"] == "invalid_query_parameter"

    @pytest.mark.asyncio
    async def test_events_tails_live_stream_and_closes_on_completion(self, adapter):
        """Subscribe to /events while the response is still streaming and
        confirm we receive every live event + the terminal response.completed,
        then the stream closes."""
        app = _create_app(adapter)

        # Gate the mock agent so the POST is still running when the
        # replay client subscribes.
        agent_gate = asyncio.Event()
        created_seen = asyncio.Event()

        async def _mock_run_agent(**kwargs):
            cb = kwargs.get("stream_delta_callback")
            # Stream 1 delta immediately so response.created fires and
            # the response gets registered in _active_responses.
            if cb:
                cb("Hello")
            # Signal the test it can issue the replay GET now.
            created_seen.set()
            # Wait for the test to subscribe.
            await agent_gate.wait()
            if cb:
                cb(" world")
            return (
                {"final_response": "Hello world", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                # Kick off the streaming POST and an events GET in parallel.
                async def _drive_post():
                    r = await cli.post(
                        "/v1/responses",
                        json={
                            "model": "hermes-agent",
                            "input": "go",
                            "stream": True,
                            "store": True,
                        },
                    )
                    return await r.text()

                post_task = asyncio.create_task(_drive_post())

                # Wait until the agent has emitted response.created + the
                # first delta, so the response_id is discoverable and the
                # response is registered as active.
                await asyncio.wait_for(created_seen.wait(), timeout=5.0)

                # Find the response_id by peeking at the writer's own
                # event log (guaranteed to contain response.created by now
                # since that is the very first _write_event call).
                for _ in range(50):
                    ids = [rid for rid in adapter._active_responses.keys()]
                    if ids:
                        response_id = ids[0]
                        break
                    await asyncio.sleep(0.02)
                else:
                    raise AssertionError("response never became active")

                # Subscribe to /events with a fresh cursor — we want the
                # replay handler to relay every live event after this
                # point, then the final response.completed, then close.
                replay_task = asyncio.create_task(
                    cli.get(f"/v1/responses/{response_id}/events?after=-1")
                )
                # Give the handler a moment to subscribe.
                await asyncio.sleep(0.05)

                # Let the agent finish.
                agent_gate.set()

                replay_resp = await asyncio.wait_for(replay_task, timeout=5.0)
                assert replay_resp.status == 200
                replay_body = await asyncio.wait_for(replay_resp.text(), timeout=5.0)
                post_body = await asyncio.wait_for(post_task, timeout=5.0)

            replay_events = [e for e in _parse_sse_events(replay_body) if e.get("data")]
            post_events = [e for e in _parse_sse_events(post_body) if e.get("data")]

            # Every sequence number the POST client saw must also appear
            # in the replay stream exactly once (idempotency by seq).
            post_seqs = [
                e["data"]["sequence_number"]
                for e in post_events
                if "sequence_number" in e["data"]
            ]
            replay_seqs = [
                e["data"]["sequence_number"]
                for e in replay_events
                if "sequence_number" in e["data"]
            ]
            assert replay_seqs == sorted(replay_seqs), "replay must be in order"
            assert len(replay_seqs) == len(set(replay_seqs)), "no duplicate seqs"
            for seq in post_seqs:
                assert seq in replay_seqs, f"missing seq {seq} in replay"

            # Replay stream must end with response.completed.
            assert replay_events[-1]["data"]["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_events_stored_response_without_event_log_still_404s_unless_stored(self, adapter):
        """A response stored via direct put() but without any event log
        rows still allows a 200 empty replay (so legacy snapshots don't
        accidentally 404).  The inverse — nothing stored, no events — is
        the 404 path tested above."""
        adapter._response_store.put("resp_legacy", {"response": {"id": "resp_legacy"}})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_legacy/events?after=-1")
            assert resp.status == 200
            body = await resp.text()
            events = [e for e in _parse_sse_events(body) if e.get("data")]
            assert events == []

