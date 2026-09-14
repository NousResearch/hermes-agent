"""Real reasoning reaches the API-server surfaces; the answer echo does not.

A top-level agent's assistant content used to be relayed as ``reasoning.available``, so every
surface rendered the reply twice — once as the reply, once as a "thinking" block identical to it.
The model's actual chain-of-thought was never on these wires at all: ``reasoning_callback`` was
simply not passed down.

These tests pin both halves:

* ``agent.turn_response_intake._relay_thinking`` relays only a subagent's first content line.
* ``_create_agent`` forwards ``reasoning_callback``, and each surface renders it in its own
  vocabulary — ``reasoning.delta`` on ``/v1/runs``, ``delta.reasoning_content`` on
  ``/v1/chat/completions``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter, cors_middleware, security_headers_middleware)


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _app(adapter: APIServerAdapter, routes) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    return app


def _runs_app(adapter: APIServerAdapter) -> web.Application:
    return _app(adapter, [
        ("POST", "/v1/runs", adapter._handle_runs),
        ("GET", "/v1/runs/{run_id}/events", adapter._handle_run_events)])


def _chat_app(adapter: APIServerAdapter) -> web.Application:
    return _app(adapter, [("POST", "/v1/chat/completions", adapter._handle_chat_completions)])


async def _drain(resp, stop: bytes) -> str:
    body = b""
    async for chunk in resp.content.iter_any():
        body += chunk
        if stop in body:
            break
    return body.decode("utf-8", errors="replace")


# ── the echo itself ─────────────────────────────────────────────────────────


def _relay(depth: int, content: str) -> list:
    from agent.turn_response_intake import _relay_thinking
    seen = []
    agent = SimpleNamespace(
        _delegate_depth=depth,
        tool_progress_callback=lambda *args, **kwargs: seen.append(args))
    _relay_thinking(agent, content)
    return seen


def test_a_top_level_answer_is_not_relayed_as_reasoning():
    """The regression: an agent's own answer came back as a `_thinking` preview identical to it."""
    assert _relay(0, "The capital of France is Paris.") == []


def test_a_subagents_first_line_still_reaches_the_parent():
    """A child's answer IS the parent's reasoning surface — that relay must survive."""
    assert _relay(1, "found the file\nmore detail") == [("_thinking", "found the file")]


def test_reasoning_tags_are_still_stripped_from_a_subagent_relay():
    assert _relay(1, "<think>weighing options</think>") == [("_thinking", "weighing options")]


# ── the plumbing ────────────────────────────────────────────────────────────


def _agent_kwargs_from_create(adapter, **create_kwargs) -> dict:
    with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
         patch("gateway.run._resolve_gateway_model", return_value="test/model"), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_kwargs.return_value = {
            "api_key": "test-key", "base_url": None, "provider": None,
            "api_mode": None, "command": None, "args": []}
        mock_agent_cls.return_value = MagicMock()
        adapter._create_agent(**create_kwargs)
        return mock_agent_cls.call_args.kwargs


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_create_agent_forwards_reasoning_callback_to_the_agent():
    """Without this the agent fires ``_fire_reasoning_delta`` into a no-op and no reasoning
    can reach any wire."""
    sentinel = MagicMock(name="reasoning_callback")
    assert _agent_kwargs_from_create(
        _make_adapter(), reasoning_callback=sentinel).get("reasoning_callback") is sentinel


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_create_agent_defaults_reasoning_callback_to_none():
    """Surfaces that render no reasoning keep passing nothing."""
    assert _agent_kwargs_from_create(_make_adapter()).get("reasoning_callback") is None


# ── /v1/runs ────────────────────────────────────────────────────────────────


def _stub_agent(run_conversation) -> MagicMock:
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.run_conversation = run_conversation
    return agent


@pytest.mark.asyncio
async def test_runs_streams_reasoning_delta_in_order_before_completion():
    adapter = _make_adapter()
    captured = {}

    def _create(**kwargs):
        captured["cb"] = kwargs.get("reasoning_callback")

        def _run(**_kw):
            cb = captured["cb"]
            assert cb is not None, "the run surface must wire reasoning_callback"
            cb("Let me ")
            cb("weigh it")
            cb(None)   # sentinels the callback must swallow rather than emit
            cb("")
            return {"final_response": "done", "messages": [], "api_calls": 1}

        return _stub_agent(_run)

    async with TestClient(TestServer(_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", side_effect=_create):
            resp = await cli.post("/v1/runs", json={"input": "hi"})
            assert resp.status == 202, await resp.text()
            run_id = (await resp.json())["run_id"]
            sse = await cli.get(f"/v1/runs/{run_id}/events")
            assert sse.status == 200
            text = await _drain(sse, b"run.completed")

    assert text.count('"event": "reasoning.delta"') == 2, text
    assert text.index('"text": "Let me "') < text.index('"text": "weigh it"'), text
    assert text.index('"event": "reasoning.delta"') < text.index('"event": "run.completed"'), text


@pytest.mark.asyncio
async def test_runs_without_reasoning_emits_no_reasoning_events():
    """A non-thinking model shows no reasoning at all, where it used to show its own answer."""
    adapter = _make_adapter()

    def _create(**_kwargs):
        return _stub_agent(lambda **_kw: {"final_response": "done", "messages": [], "api_calls": 1})

    async with TestClient(TestServer(_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", side_effect=_create):
            resp = await cli.post("/v1/runs", json={"input": "hi"})
            run_id = (await resp.json())["run_id"]
            text = await _drain(await cli.get(f"/v1/runs/{run_id}/events"), b"run.completed")

    assert "reasoning.delta" not in text, text
    assert "reasoning.available" not in text, text
    assert '"event": "run.completed"' in text


# ── /v1/chat/completions ────────────────────────────────────────────────────


def _fake_run_agent(reasoning=(), content=()):
    async def _run(**kwargs):
        for text in reasoning:
            kwargs["reasoning_callback"](text)
        for text in content:
            kwargs["stream_delta_callback"](text)
        return ({"final_response": "".join(content), "messages": [], "api_calls": 1,
                 "completed": True},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    return _run


async def _chat_sse(adapter, run_agent) -> str:
    async with TestClient(TestServer(_chat_app(adapter))) as cli:
        with patch.object(adapter, "_run_agent", run_agent):
            resp = await cli.post("/v1/chat/completions", json={
                "model": "test/model", "messages": [{"role": "user", "content": "hi"}],
                "stream": True})
            assert resp.status == 200, await resp.text()
            return await _drain(resp, b"[DONE]")


@pytest.mark.asyncio
async def test_chat_stream_keeps_reasoning_in_its_own_delta_field():
    """``delta.reasoning_content`` is what OpenAI-compatible clients for thinking models read;
    mixing it into ``delta.content`` is what made the answer look doubled."""
    text = await _chat_sse(
        _make_adapter(), _fake_run_agent(reasoning=["weighing"], content=["answer"]))

    assert '"reasoning_content": "weighing"' in text, text
    assert '"content": "answer"' in text, text
    assert text.index("reasoning_content") < text.index('"content": "answer"'), text


@pytest.mark.asyncio
async def test_chat_stream_without_reasoning_carries_no_reasoning_field():
    text = await _chat_sse(_make_adapter(), _fake_run_agent(content=["just an answer"]))

    assert "reasoning_content" not in text, text
    assert '"content": "just an answer"' in text, text
