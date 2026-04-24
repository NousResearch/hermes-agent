"""Tests for reasoning.delta streaming over the /v1/runs SSE channel.

Covers the fix in PR #15169: the API server adapter must wire a
reasoning_callback when constructing AIAgent so that reasoning/thinking
tokens stream in real time instead of arriving only as a single
reasoning.available snapshot after the run completes.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app_with_runs(adapter: APIServerAdapter) -> web.Application:
    """Minimal aiohttp app exposing only the /v1/runs endpoints under test."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


# ---------------------------------------------------------------------------
# Unit: _create_agent threads reasoning_callback through to AIAgent
# ---------------------------------------------------------------------------


class TestCreateAgentReasoningCallback:
    """Verify the kwarg plumbing between _create_agent and AIAgent."""

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_passes_reasoning_callback_to_aiagent(self):
        adapter = _make_adapter()
        sentinel = MagicMock(name="reasoning_callback")

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None), \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": [],
            }
            mock_model.return_value = "test/model"
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(reasoning_callback=sentinel)

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args.kwargs
            assert call_kwargs.get("reasoning_callback") is sentinel, (
                "reasoning_callback was not forwarded to AIAgent — without this "
                "the agent will fire _fire_reasoning_delta into a no-op and "
                "reasoning tokens will never reach the SSE stream."
            )

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_default_reasoning_callback_is_none(self):
        """Default value preserves backward-compat for callers that don't opt in."""
        adapter = _make_adapter()

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None), \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": [],
            }
            mock_model.return_value = "test/model"
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            call_kwargs = mock_agent_cls.call_args.kwargs
            assert call_kwargs.get("reasoning_callback") is None


# ---------------------------------------------------------------------------
# Integration: reasoning.delta SSE events flow end-to-end
# ---------------------------------------------------------------------------


class TestRunsReasoningDeltaSSE:
    """End-to-end SSE stream of reasoning.delta for /v1/runs."""

    @pytest.mark.asyncio
    async def test_reasoning_delta_events_reach_sse_stream(self):
        """The callback wired by _handle_runs must publish reasoning.delta
        events onto the client SSE queue when the agent fires reasoning
        tokens during run_conversation."""
        adapter = _make_adapter()

        captured = {"reasoning_callback": None}

        def _fake_create_agent(**kwargs):
            # Capture the reasoning_callback the handler wired up so the
            # stub agent can invoke it from inside run_conversation (which
            # itself runs in an executor thread, mirroring production).
            captured["reasoning_callback"] = kwargs.get("reasoning_callback")
            fake = MagicMock()
            fake.session_prompt_tokens = 0
            fake.session_completion_tokens = 0
            fake.session_total_tokens = 0

            def _run_conversation(**_kw):
                cb = captured["reasoning_callback"]
                assert cb is not None, (
                    "reasoning_callback must be wired by _handle_runs — "
                    "without it reasoning tokens cannot stream."
                )
                cb("Let me ")
                cb("think…")
                # None / empty-string are sentinels that should NOT emit an event
                cb(None)
                cb("")
                return {"final_response": "done", "messages": [], "api_calls": 1}

            fake.run_conversation = _run_conversation
            return fake

        app = _create_app_with_runs(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent):
                # Kick off a run
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hi"},
                )
                assert resp.status == 202, await resp.text()
                run_id = (await resp.json())["run_id"]

                # Consume the SSE stream until we see run.completed
                sse = await cli.get(f"/v1/runs/{run_id}/events")
                assert sse.status == 200
                assert "text/event-stream" in sse.headers.get("Content-Type", "")

                body = b""
                async for chunk in sse.content.iter_any():
                    body += chunk
                    if b"run.completed" in body or b"run.failed" in body:
                        break
                text = body.decode("utf-8", errors="replace")

        assert '"event": "reasoning.delta"' in text, (
            f"reasoning.delta event missing from SSE stream:\n{text}"
        )
        assert '"text": "Let me "' in text
        assert '"text": "think\\u2026"' in text or '"text": "think…"' in text
        # Empty / None payloads must be suppressed
        assert text.count('"event": "reasoning.delta"') == 2, (
            f"expected exactly 2 reasoning.delta events (non-empty text only), got:\n{text}"
        )
        # run.completed still fires — PR doesn't regress the existing terminal event
        assert '"event": "run.completed"' in text

    @pytest.mark.asyncio
    async def test_reasoning_callback_errors_do_not_break_stream(self):
        """A failure inside the reasoning callback must not abort the run.
        The callback wraps put_nowait in try/except so the agent keeps running
        even if the client has already disconnected."""
        adapter = _make_adapter()

        def _fake_create_agent(**kwargs):
            fake = MagicMock()
            fake.session_prompt_tokens = 0
            fake.session_completion_tokens = 0
            fake.session_total_tokens = 0

            def _run_conversation(**_kw):
                cb = kwargs.get("reasoning_callback")
                # Simulate a very long text — callback must tolerate anything
                # the agent hands it without propagating exceptions back.
                cb("x" * 4096)
                return {"final_response": "ok", "messages": [], "api_calls": 1}

            fake.run_conversation = _run_conversation
            return fake

        app = _create_app_with_runs(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                sse = await cli.get(f"/v1/runs/{run_id}/events")
                body = b""
                async for chunk in sse.content.iter_any():
                    body += chunk
                    if b"run.completed" in body or b"run.failed" in body:
                        break
                text = body.decode("utf-8", errors="replace")

        assert '"event": "run.completed"' in text
        assert '"event": "run.failed"' not in text
