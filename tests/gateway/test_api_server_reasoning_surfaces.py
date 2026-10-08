"""Real reasoning reaches the API-server surfaces; the answer echo does not.

A top-level agent's assistant content was relayed to the progress callback as
``reasoning.available``. That content is the *answer*, so a surface rendering it showed the reply
twice — once as the reply, once as a "thinking" block matching it word for word — and a model that
emits no reasoning at all still produced one, because the echo never depended on any reasoning
existing.

The model's real chain-of-thought rides ``reasoning_callback``
(``stream_delivery._fire_reasoning_delta``). ``/v1/chat/completions`` already consumes it (#99552);
``/api/sessions/{id}/chat/stream`` and ``/v1/runs`` did not, so the echo was all they had.

``tests/gateway/test_api_server_reasoning_stream.py`` covers the writers #99552 already wired;
these tests pin the two surfaces it did not reach, and the echo itself:

* ``agent.turn_response_intake._relay_thinking`` relays only a subagent's first content line.
* each remaining surface renders the real deltas in its own vocabulary — the existing
  ``tool.progress``/``_thinking`` event on the session stream, ``reasoning.delta`` on ``/v1/runs``.
"""

import json as _json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter, cors_middleware, security_headers_middleware)
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


def _make_adapter(session_db=None) -> APIServerAdapter:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    if session_db is not None:
        adapter._session_db = session_db
    return adapter


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


def _session_app(adapter: APIServerAdapter) -> web.Application:
    return _app(adapter, [
        ("POST", "/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)])


async def _drain(resp, stop: bytes) -> str:
    body = b""
    async for chunk in resp.content.iter_any():
        body += chunk
        if stop in body:
            break
    return body.decode("utf-8", errors="replace")


def _sse_payloads(body: str, event: str) -> list:
    """Every ``data:`` payload carrying ``event``, in wire order."""
    out = []
    for block in body.split("\n\n"):
        if f'"event": "{event}"' not in block and f"event: {event}" not in block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                with_data = _json.loads(line[len("data: "):])
                if with_data.get("event", event) == event:
                    out.append(with_data)
    return out


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
    """The regression: an agent's own answer came back as a ``_thinking`` preview identical to it."""
    assert _relay(0, "The capital of France is Paris.") == []


def test_a_subagents_first_line_still_reaches_the_parent():
    """A child's answer IS the parent's reasoning surface — that relay must survive."""
    assert _relay(1, "found the file\nmore detail") == [("_thinking", "found the file")]


def test_reasoning_tags_are_still_stripped_from_a_subagent_relay():
    assert _relay(1, "<think>weighing options</think>") == [("_thinking", "weighing options")]


def test_a_relay_failure_never_breaks_the_turn():
    from agent.turn_response_intake import _relay_thinking

    def _boom(*_args, **_kwargs):
        raise RuntimeError("consumer went away")

    _relay_thinking(SimpleNamespace(_delegate_depth=1, tool_progress_callback=_boom), "text")


# ── the plumbing ────────────────────────────────────────────────────────────


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_create_agent_forwards_reasoning_callback_to_the_agent():
    """Both surfaces below depend on this hand-off: without it the agent fires
    ``_fire_reasoning_delta`` into a no-op and no reasoning can reach any wire."""
    sentinel = MagicMock(name="reasoning_callback")
    with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
         patch("gateway.run._resolve_gateway_model", return_value="test/model"), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_kwargs.return_value = {
            "api_key": "test-key", "base_url": None, "provider": None,
            "api_mode": None, "command": None, "args": []}
        mock_agent_cls.return_value = MagicMock()
        _make_adapter()._create_agent(reasoning_callback=sentinel)
        assert mock_agent_cls.call_args.kwargs.get("reasoning_callback") is sentinel


# ── /api/sessions/{id}/chat/stream ──────────────────────────────────────────


async def _session_sse(adapter, session_id, fake_run) -> str:
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(_session_app(adapter))) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "hi"})
            assert resp.status == 200, await resp.text()
            return await resp.text()


@pytest.mark.asyncio
async def test_session_stream_renders_real_reasoning_on_its_existing_thinking_event(session_db):
    """The wire contract is untouched — same ``tool.progress``/``_thinking`` event. Only the
    source changes, from the answer echo to the model's reasoning deltas."""
    adapter = _make_adapter(session_db)
    session_id = session_db.create_session("reasoning-session", "api_server")

    async def fake_run(**kwargs):
        reasoning = kwargs["reasoning_callback"]
        assert reasoning is not None, "the session stream must wire reasoning_callback"
        reasoning("Let me ")
        reasoning("weigh it")
        reasoning("")      # empty deltas are noise, not a thinking block
        reasoning(None)
        kwargs["stream_delta_callback"]("Paris.")
        return ({"final_response": "Paris.", "session_id": session_id,
                 "messages": [{"role": "assistant", "content": "Paris."}]}, {"total_tokens": 3})

    body = await _session_sse(adapter, session_id, fake_run)

    thinking = [p for p in _sse_payloads(body, "tool.progress") if p.get("tool_name") == "_thinking"]
    assert [p["delta"] for p in thinking] == ["Let me ", "weigh it"], body
    assert body.index("weigh it") < body.index("run.completed"), body


@pytest.mark.asyncio
async def test_session_stream_without_reasoning_shows_no_thinking_block(session_db):
    """A non-thinking model now shows nothing, where it used to show its own answer back."""
    adapter = _make_adapter(session_db)
    session_id = session_db.create_session("no-reasoning-session", "api_server")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"]("Paris.")
        return ({"final_response": "Paris.", "session_id": session_id,
                 "messages": [{"role": "assistant", "content": "Paris."}]}, {"total_tokens": 3})

    body = await _session_sse(adapter, session_id, fake_run)

    assert "_thinking" not in body, body
    assert "Paris." in body


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
