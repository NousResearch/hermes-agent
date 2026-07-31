"""Tests for reasoning streaming over the /v1/runs and /v1/chat/completions
SSE channels.

Covers the fixes in PR #75562:

* ``/v1/runs`` streams the model's real chain-of-thought as incremental
  ``reasoning.delta`` events (not a one-shot ``reasoning.available``
  snapshot), so multi-chunk reasoning from GLM/DeepSeek/Kimi/Qwen thinking
  models is preserved in arrival order before ``run.completed``.  The
  ``reasoning.available`` snapshot path is intentionally left intact for
  callers that still rely on the post-run summary - only the live callback
  wiring switches to ``reasoning.delta``, matching the contract proposed in
  #15169.
* ``/v1/chat/completions`` streams ``delta.reasoning_content`` chunks, kept
  separate from ``delta.content`` so OpenAI-compatible clients that read
  ``delta.reasoning_content`` render true reasoning instead of nothing or a
  content echo.
* ``_create_agent`` / ``_run_agent`` thread ``reasoning_callback`` through
  to ``AIAgent`` so ``_fire_reasoning_delta`` actually reaches the wire
  instead of firing into a no-op.
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


def _create_app_with_chat(adapter: APIServerAdapter) -> web.Application:
    """Minimal aiohttp app exposing only /v1/chat/completions under test."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


# ---------------------------------------------------------------------------
# Unit: _create_agent threads reasoning_callback through to AIAgent
# ---------------------------------------------------------------------------


class TestCreateAgentReasoningCallback:
    """Verify the kwarg plumbing between _create_agent and AIAgent.

    _run_agent defines its own _reasoning_cb and forwards it as
    ``reasoning_callback=`` to _create_agent, which must in turn hand it to
    AIAgent so _fire_reasoning_delta has a listener.
    """

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
                "reasoning_callback was not forwarded to AIAgent - without this "
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
# Integration: reasoning.delta SSE events flow end-to-end over /v1/runs
# ---------------------------------------------------------------------------


class TestRunsReasoningDeltaSSE:
    """End-to-end SSE stream of reasoning.delta for /v1/runs."""

    @pytest.mark.asyncio
    async def test_reasoning_delta_events_reach_sse_stream(self):
        """The callback wired by _handle_runs/_run_agent must publish
        reasoning.delta events onto the client SSE queue when the agent fires
        reasoning tokens during run_conversation, preserving multi-chunk
        order before run.completed."""
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
                    "reasoning_callback must be wired by _handle_runs - "
                    "without it reasoning tokens cannot stream."
                )
                cb("Let me ")
                cb("think\u2026")
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
        assert '"text": "think\\u2026"' in text or '"text": "think\u2026"' in text
        # Empty / None payloads must be suppressed
        assert text.count('"event": "reasoning.delta"') == 2, (
            f"expected exactly 2 reasoning.delta events (non-empty text only), got:\n{text}"
        )
        # Multi-chunk ordering: "Let me " must precede "think…" in the stream
        assert text.index('"text": "Let me "') < text.index(
            '"text": "think\\u2026"' if '"text": "think\\u2026"' in text
            else '"text": "think\u2026"'
        ), f"reasoning.delta chunks arrived out of order:\n{text}"
        # reasoning.delta must arrive before the terminal run.completed
        assert text.index('"event": "reasoning.delta"') < text.index(
            '"event": "run.completed"'
        ), f"reasoning.delta streamed after run.completed:\n{text}"
        # run.completed still fires - PR doesn't regress the existing terminal event
        assert '"event": "run.completed"' in text

    @pytest.mark.asyncio
    async def test_reasoning_callback_errors_do_not_break_stream(self):
        """A failure inside the reasoning callback must not abort the run.
        The callback wraps the thread-safe put in try/except so the agent
        keeps running even if the client has already disconnected."""
        adapter = _make_adapter()

        def _fake_create_agent(**kwargs):
            fake = MagicMock()
            fake.session_prompt_tokens = 0
            fake.session_completion_tokens = 0
            fake.session_total_tokens = 0

            def _run_conversation(**_kw):
                cb = kwargs.get("reasoning_callback")
                # Simulate a very long text - callback must tolerate anything
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


# ---------------------------------------------------------------------------
# Integration: delta.reasoning_content is separated from delta.content over
# /v1/chat/completions (streaming)
# ---------------------------------------------------------------------------


class TestChatCompletionsReasoningContentSeparation:
    """PR #75562: /v1/chat/completions streams ``delta.reasoning_content``
    chunks that are kept separate from ``delta.content`` so OpenAI-compatible
    clients render true reasoning instead of a content echo."""

    @pytest.mark.asyncio
    async def test_reasoning_content_separate_from_content(self):
        adapter = _make_adapter()

        async def _fake_run_agent(**kwargs):
            rc = kwargs.get("reasoning_callback")
            dc = kwargs.get("stream_delta_callback")
            # Real thinking models emit reasoning deltas BEFORE the answer.
            if rc is not None:
                rc("Let me think\u2026")
            if dc is not None:
                dc("answer")
            return (
                {
                    "final_response": "answer",
                    "messages": [],
                    "api_calls": 1,
                    "completed": True,
                },
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        app = _create_app_with_chat(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", _fake_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test/model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200, await resp.text()
                body = b""
                async for chunk in resp.content.iter_any():
                    body += chunk
                    if b"[DONE]" in body:
                        break
                text = body.decode("utf-8", errors="replace")

        reasoning_marker = (
            '"reasoning_content": "think\\u2026"'
            if '"reasoning_content": "think\\u2026"' in text
            else '"reasoning_content": "think\u2026"'
        )
        assert '"reasoning_content": "Let me think' in text or reasoning_marker in text, (
            f"delta.reasoning_content missing from chat SSE:\n{text}"
        )
        assert '"content": "answer"' in text, (
            f"delta.content missing from chat SSE:\n{text}"
        )
        # Reasoning must stream before content (thinking models reason first);
        # this is the exact "answer echo" regression the PR fixes.
        assert text.index("reasoning_content") < text.index('"content": "answer"'), (
            f"reasoning_content should precede content:\n{text}"
        )

    @pytest.mark.asyncio
    async def test_non_thinking_model_emits_no_reasoning_content(self):
        """When the model never fires reasoning_callback, the chat SSE must
        contain only delta.content chunks - no spurious reasoning block."""
        adapter = _make_adapter()

        async def _fake_run_agent(**kwargs):
            dc = kwargs.get("stream_delta_callback")
            # Non-thinking model: reasoning_callback is wired but never called.
            if dc is not None:
                dc("just an answer")
            return (
                {
                    "final_response": "just an answer",
                    "messages": [],
                    "api_calls": 1,
                    "completed": True,
                },
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        app = _create_app_with_chat(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", _fake_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test/model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200, await resp.text()
                body = b""
                async for chunk in resp.content.iter_any():
                    body += chunk
                    if b"[DONE]" in body:
                        break
                text = body.decode("utf-8", errors="replace")

        assert "reasoning_content" not in text, (
            f"non-thinking model should not emit reasoning_content:\n{text}"
        )
        assert '"content": "just an answer"' in text
