"""Client-owned functions must never enter the Hermes execution loop."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.run  # Load the runtime before per-test filesystem guards.
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {
    "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}}]
CHOICE = {"type": "function", "function": {"name": "terminal"}}
CALL = {"id": "call_client_1", "type": "function", "function": {
    "name": "terminal", "arguments": '{"value":"client-only"}'}}


def completion(content=None, calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(index=0, message=SimpleNamespace(
        role="assistant", content=content, tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13))


@pytest.fixture
def provider():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    upstream = Mock()
    upstream.chat.completions.create.return_value = completion(calls=[SimpleNamespace(
        **{**CALL, "function": SimpleNamespace(**CALL["function"])})])
    runtime = {"provider": "openai", "api_mode": "chat_completions", "api_key": "test", "base_url": "http://unused"}
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime), \
         patch("gateway.run._resolve_gateway_model", return_value="test-model"), \
         patch("agent.auxiliary_client._create_openai_client", return_value=upstream), \
         patch.object(adapter, "_run_agent", new=AsyncMock(return_value=({"final_response": "old path"}, {}))) as run:
        yield adapter, upstream, run


@pytest.mark.asyncio
async def test_stream_returns_indexed_tool_calls_and_done(provider):
    adapter, _upstream, run = provider
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Call terminal"}], "tools": TOOLS,
            "tool_choice": CHOICE, "stream": True, "stream_options": {"include_usage": True}})
        assert response.status == 200
        assert response.content_type == "text/event-stream"
        text = await response.text()
    frames = [json.loads(line[6:]) for line in text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert text.endswith("data: [DONE]\n\n")
    deltas = [frame["choices"][0]["delta"] for frame in frames if frame["choices"]]
    assert any(delta.get("tool_calls") == [{"index": 0, **CALL}] for delta in deltas)
    assert any(frame["choices"] and frame["choices"][0]["finish_reason"] == "tool_calls" for frame in frames)
    assert frames[-1]["usage"]["total_tokens"] == 13
    run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, {"tool_choice": "none"}])
async def test_tool_result_followup_without_redeclaring_tools(provider, options):
    adapter, upstream, run = provider
    upstream.chat.completions.create.return_value = completion(content="nonce-123")
    messages = [{"role": "user", "content": "Call terminal"},
                {"role": "assistant", "content": None, "tool_calls": [CALL]},
                {"role": "tool", "tool_call_id": CALL["id"], "content": "nonce-123"}]
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/chat/completions", json={"messages": messages, **options})
        assert response.status == 200
        data = await response.json()
    assert data["choices"][0]["message"]["content"] == "nonce-123"
    assert upstream.chat.completions.create.call_args.kwargs["messages"] == messages
    run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"tools": "terminal"}, {"tools": [{"type": "web_search"}]},
    {"tools": [{"type": "function", "function": {}}]},
    {"tool_choice": "invalid"}, {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
    {"messages": ["bad"]}, {"parallel_tool_calls": "false"}, {"n": 2},
    {"tool_choice": {"type": "function", "function": {"name": []}}},
    {"stream": True, "stream_options": "invalid"},
])
async def test_invalid_client_function_requests_fail_before_provider(provider, changes):
    adapter, upstream, run = provider
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Call terminal"}], "tools": TOOLS, **changes})
        assert response.status == 400
        assert (await response.json())["error"]["type"] == "invalid_request_error"
    upstream.chat.completions.create.assert_not_called()
    run.assert_not_called()


@pytest.mark.asyncio
async def test_client_inference_counts_against_concurrency_cap(provider):
    adapter, upstream, run = provider
    observed = []
    result = upstream.chat.completions.create.return_value
    def respond(**kwargs):
        observed.append(adapter._inflight_agent_runs)
        return result
    upstream.chat.completions.create.side_effect = respond
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Call terminal"}], "tools": TOOLS})
        assert response.status == 200
    assert observed == [1]
    assert adapter._inflight_agent_runs == 0
    run.assert_not_called()


@pytest.mark.asyncio
async def test_active_client_inference_uses_only_one_admission_slot(provider):
    adapter, upstream, run = provider
    adapter._max_concurrent_runs = 2
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    result = upstream.chat.completions.create.return_value

    def respond(**kwargs):
        if kwargs["messages"][0]["content"] == "first":
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=10), "blocked provider was not released"
        return result

    upstream.chat.completions.create.side_effect = respond
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "first"}], "tools": TOOLS}))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            counts = (adapter._inflight_agent_runs, adapter._pending_agent_requests,
                      adapter.active_agent_work_count())
            second = await asyncio.wait_for(client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "second"}], "tools": TOOLS}), timeout=5)
            assert (counts, second.status) == ((1, 0, 1), 200)
            assert adapter.active_agent_work_count() == 1
        finally:
            release.set()
            first_response = await asyncio.wait_for(first, timeout=5)
        assert first_response.status == 200
    assert adapter.active_agent_work_count() == 0
    assert adapter._pending_agent_requests == 0
    assert upstream.chat.completions.create.call_count == 2
    assert upstream.close.call_count == 2
    run.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_transport_fails_closed(provider):
    adapter, upstream, run = provider
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
            "provider": "anthropic", "api_mode": "anthropic_messages", "api_key": "test", "base_url": "http://unused"}):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}], "tools": TOOLS})
            assert response.status == 500
            assert "not supported" in (await response.json())["error"]["message"]
    upstream.chat.completions.create.assert_not_called()
    run.assert_not_called()


@pytest.mark.asyncio
async def test_tool_free_requests_keep_agent_path(provider):
    adapter, upstream, run = provider
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}]})
        assert response.status == 200
        assert (await response.json())["choices"][0]["message"]["content"] == "old path"
    run.assert_awaited_once()
    upstream.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_provider_failure_closes_client_and_releases_slot(provider):
    adapter, upstream, run = provider
    upstream.chat.completions.create.side_effect = RuntimeError("provider unavailable")
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}], "tools": TOOLS})
        assert response.status == 500
        assert (await response.json())["error"]["type"] == "server_error"
    upstream.close.assert_called_once()
    assert adapter._inflight_agent_runs == 0
    run.assert_not_called()


@pytest.mark.asyncio
async def test_client_function_is_returned_not_executed():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    upstream = Mock()
    upstream.chat.completions.create.return_value = completion(calls=[SimpleNamespace(
        **{**CALL, "function": SimpleNamespace(**CALL["function"])})])
    runtime = {"provider": "openai", "api_mode": "chat_completions", "api_key": "test", "base_url": "http://unused"}
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime), \
         patch("gateway.run._resolve_gateway_model", return_value="test-model"), \
         patch("agent.auxiliary_client._create_openai_client", return_value=upstream), \
         patch.object(adapter, "_run_agent", new=AsyncMock(return_value=({"final_response": "old path"}, {}))) as run:
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Call terminal"}], "tools": TOOLS, "tool_choice": CHOICE})
            assert response.status == 200
            data = await response.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": None, "tool_calls": [CALL]}
    assert data["usage"]["total_tokens"] == 13
    run.assert_not_called()
    kwargs = upstream.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == CHOICE
    upstream.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [CHOICE, "auto", "none", "required"])
async def test_codex_preserves_forced_choice_and_tool_result_history(choice):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    upstream = Mock()
    upstream.base_url = "https://chatgpt.com/backend-api/codex"
    upstream._hermes_aux_effective_provider = "openai-codex"
    upstream.responses.create.return_value = SimpleNamespace(
        status="completed", usage=SimpleNamespace(input_tokens=10, output_tokens=3, total_tokens=13),
        output=[SimpleNamespace(type="function_call", id="fc_1", call_id=CALL["id"],
                                name="terminal", arguments=CALL["function"]["arguments"])])
    messages = [{"role": "user", "content": "Call terminal"},
                {"role": "assistant", "content": None, "tool_calls": [CALL]},
                {"role": "tool", "tool_call_id": CALL["id"], "content": "nonce-123"}]
    runtime = {"provider": "openai-codex", "api_mode": "codex_responses", "api_key": "test",
               "base_url": upstream.base_url}
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime), \
         patch("gateway.run._resolve_gateway_model", return_value="gpt-6-astra-900k"), \
         patch("agent.auxiliary_client._create_openai_client", return_value=upstream), \
         patch.object(adapter, "_run_agent", new=AsyncMock()) as run:
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", json={
                "messages": messages, "tools": TOOLS, "tool_choice": choice, "parallel_tool_calls": False})
            assert response.status == 200
            data = await response.json()
    assert data["choices"][0]["message"]["tool_calls"] == [CALL]
    kwargs = upstream.responses.create.call_args.kwargs
    kwargs = {**kwargs, **kwargs.get("extra_body", {})}
    assert kwargs["tool_choice"] == ({"type": "function", "name": "terminal"} if isinstance(choice, dict) else choice)
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["model"] == "gpt-6-astra"
    assert any(item.get("type") == "function_call_output" and item["output"] == "nonce-123"
               for item in kwargs["input"])
    run.assert_not_called()
    upstream.chat.completions.create.assert_not_called()
    upstream.close.assert_called_once()
