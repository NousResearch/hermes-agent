"""
Tests for the api_server tool_progress_events config option (#12020).

When set to false, hermes.tool.progress SSE events should be suppressed
so frontends that only handle standard OpenAI delta chunks work without
parse errors.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter(**extra_kw) -> APIServerAdapter:
    """Create an adapter with optional extra config keys."""
    config = PlatformConfig(enabled=True, extra=extra_kw)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    from gateway.platforms.api_server import cors_middleware, security_headers_middleware
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


class TestToolProgressEventsConfig:
    """Config parsing for tool_progress_events."""

    def test_default_is_true(self):
        """By default, tool_progress_events is True (backward compatible)."""
        adapter = _make_adapter()
        assert adapter._tool_progress_events is True

    def test_explicit_true(self):
        adapter = _make_adapter(tool_progress_events=True)
        assert adapter._tool_progress_events is True

    def test_explicit_false(self):
        adapter = _make_adapter(tool_progress_events=False)
        assert adapter._tool_progress_events is False

    def test_string_false(self):
        adapter = _make_adapter(tool_progress_events="false")
        assert adapter._tool_progress_events is False

    def test_string_true(self):
        adapter = _make_adapter(tool_progress_events="true")
        assert adapter._tool_progress_events is True

    @pytest.mark.asyncio
    async def test_capabilities_reflects_true(self):
        """Capabilities endpoint shows tool_progress_events=True by default."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            data = await resp.json()
            assert data["features"]["tool_progress_events"] is True

    @pytest.mark.asyncio
    async def test_capabilities_reflects_false(self):
        """Capabilities endpoint shows tool_progress_events=False when disabled."""
        adapter = _make_adapter(tool_progress_events=False)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            data = await resp.json()
            assert data["features"]["tool_progress_events"] is False


class TestToolProgressEventsStreaming:
    """Verify tool progress events are suppressed when disabled."""

    @pytest.mark.asyncio
    async def test_no_tool_progress_sse_when_disabled(self):
        """When tool_progress_events=False, no hermes.tool.progress events in SSE."""
        adapter = _make_adapter(tool_progress_events=False)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                # Simulate tool_start_callback being called
                tool_start_cb = kwargs.get("tool_start_callback")
                tool_complete_cb = kwargs.get("tool_complete_callback")
                stream_cb = kwargs.get("stream_delta_callback")
                if tool_start_cb:
                    tool_start_cb("call_123", "terminal", {"command": "ls"})
                if stream_cb:
                    stream_cb("result text")
                if tool_complete_cb:
                    tool_complete_cb("call_123", "terminal", {"command": "ls"}, "done")
                if stream_cb:
                    stream_cb(None)
                return (
                    {"final_response": "result text", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                # Should NOT contain hermes.tool.progress events
                assert "hermes.tool.progress" not in body
                # Should still contain normal delta content
                assert "result text" in body

    @pytest.mark.asyncio
    async def test_tool_progress_sse_present_when_enabled(self):
        """When tool_progress_events=True (default), tool progress events are emitted."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                tool_start_cb = kwargs.get("tool_start_callback")
                stream_cb = kwargs.get("stream_delta_callback")
                if tool_start_cb:
                    tool_start_cb("call_456", "terminal", {"command": "ls"})
                if stream_cb:
                    stream_cb("hello")
                    stream_cb(None)
                return (
                    {"final_response": "hello", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                # Should contain hermes.tool.progress events
                assert "hermes.tool.progress" in body
