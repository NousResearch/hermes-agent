"""Response observers run under the physical Relay model attempt, before SDK parsing."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import UUID, uuid4

import anthropic
import httpx
import nemo_relay
import openai
import pytest

from agent import relay_llm, relay_runtime
from agent.agent_runtime_helpers import create_openai_client
from agent.anthropic_adapter import _new_sdk_client
from agent.served_model import install_llm_response_hook
from hermes_cli import plugins


@pytest.fixture
def response_plugin(tmp_path, monkeypatch):
    home = tmp_path / "response-home"
    directory = home / "plugins" / "response-observer"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("name: response-observer\nversion: '1.0'\n", encoding="utf-8")
    (directory / "__init__.py").write_text(
        "import nemo_relay\n"
        "def observe(*, status_code, get_header):\n"
        "    with nemo_relay.use_scope_stack(nemo_relay.fork_scope_stack()):\n"
        "        nemo_relay.scope.event('test.http.response', data={\n"
        "            'response_id': get_header('x-test-response-id'), 'status': status_code})\n"
        "def register(ctx):\n"
        "    ctx.register_hook('llm_http_response', observe)\n", encoding="utf-8",
    )
    (home / "config.yaml").write_text("plugins:\n  enabled: [response-observer]\n", encoding="utf-8")
    empty = tmp_path / "bundled"
    empty.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: empty)
    manager = plugins.get_plugin_manager()
    yield manager
    manager.unload()


@pytest.fixture
def model_events(response_plugin):
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="response-session", platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn", task_id="task")
    lease.host.retain_managed_execution("test.responses")
    events = []
    nemo_relay.subscribers.register("test.responses", events.append)
    try:
        yield events
    finally:
        nemo_relay.subscribers.flush()
        nemo_relay.subscribers.deregister("test.responses")
        lease.host.release_managed_execution("test.responses")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _client(provider, handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs = {"api_key": "test-key", "base_url": "https://api.example.com/v1", "http_client": http}
    if provider == "anthropic":
        return _new_sdk_client(anthropic, {**kwargs, "max_retries": 0}, {})
    agent = SimpleNamespace(provider="response-test", model="test-model", _client_log_context=lambda: "test")
    return create_openai_client(agent, kwargs, reason="test", shared=False)


def _request(provider="openai"):
    request = {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}
    return {**request, "max_tokens": 16} if provider == "anthropic" else request


def _body(provider="openai"):
    if provider == "anthropic":
        return {"id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
                "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}]}


def _execute(client, provider="openai"):
    create = client.messages.create if provider == "anthropic" else client.chat.completions.create
    return relay_llm.execute(
        _request(provider), lambda request: create(**request), name=provider, model_name="test-model",
        session_id="response-session",
        metadata={"api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions"},
    )


def _response(request, forwarded, status=200, *, headers=None, **body):
    response_id = str(uuid4())
    forwarded[response_id] = (request.headers["traceparent"], status)
    return httpx.Response(status, headers={**(headers or {}), "x-test-response-id": response_id}, **body)


def _assert_response_parents(events, forwarded, *, attempts=1):
    nemo_relay.subscribers.flush()
    responses = [event for event in events if event.name == "test.http.response"]
    starts = {event.uuid: event for event in events if event.category == "llm" and event.scope_category == "start"}
    assert len(starts) == attempts
    assert len(responses) == len(forwarded)
    assert {response.data["response_id"] for response in responses} == forwarded.keys()
    assert {response.parent_uuid for response in responses} == starts.keys()
    for response in responses:
        model = starts[response.parent_uuid]
        traceparent, status = forwarded[response.data["response_id"]]
        assert traceparent.split("-")[2] == UUID(model.uuid).hex[-16:]
        assert response.data == {"response_id": response.data["response_id"], "status": status}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("status", [200, 403])
def test_response_observers_cover_success_and_sdk_errors(provider, status, model_events):
    forwarded = {}

    def respond(request):
        body = _body(provider) if status == 200 else {"error": {"type": "permission_error", "message": "denied"}}
        return _response(request, forwarded, status, json=body)

    with _client(provider, respond) as client:
        if status == 403:
            error = anthropic.PermissionDeniedError if provider == "anthropic" else openai.PermissionDeniedError
            with pytest.raises(error):
                _execute(client, provider)
        else:
            result = _execute(client, provider)
            assert (result.content[0].text if provider == "anthropic" else result.choices[0].message.content) == "hello"
    _assert_response_parents(model_events, forwarded)


def test_stream_response_is_observed_before_a_later_body_failure(model_events):
    class InterruptedBody(httpx.SyncByteStream):
        def __iter__(self):
            yield (b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"test-model",'
                   b'"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n')
            raise httpx.ReadError("body interrupted")

    forwarded = {}

    def respond(request):
        return _response(request, forwarded, stream=InterruptedBody(), headers={"content-type": "text/event-stream"})

    with _client("openai", respond) as client:
        stream = relay_llm.stream(
            _request(), lambda request: client.chat.completions.create(**request, stream=True),
            name="openai", model_name="test-model", session_id="response-session", finalizer=dict,
            metadata={"api_mode": "chat_completions"},
        )
        assert next(stream).choices[0].delta.content == "hello"
        with pytest.raises(httpx.ReadError, match="body interrupted"):
            list(stream)
    _assert_response_parents(model_events, forwarded)


def test_sdk_retry_observes_both_responses_under_one_attempt(model_events, monkeypatch):
    forwarded = {}

    def respond(request):
        status = 200 if forwarded else 429
        return _response(request, forwarded, status, json=_body() if status == 200 else {"error": {"message": "retry"}})

    with _client("openai", respond) as client:
        client.max_retries = 1
        monkeypatch.setattr(client, "_calculate_retry_timeout", lambda *args: 0)
        assert _execute(client).choices[0].message.content == "hello"
    assert [status for _, status in forwarded.values()] == [429, 200]
    _assert_response_parents(model_events, forwarded)


def test_concurrent_calls_keep_separate_response_parents(model_events):
    gate = threading.Barrier(2)
    forwarded = {}

    def respond(request):
        gate.wait(timeout=10)
        return _response(request, forwarded, 403, json={"error": {"message": "denied"}})

    def call():
        with pytest.raises(openai.PermissionDeniedError):
            _execute(client)

    with _client("openai", respond) as client, ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(contextvars.copy_context().run, call) for _ in range(2)]
        for future in futures:
            future.result(timeout=30)
    _assert_response_parents(model_events, forwarded, attempts=2)


def test_plugin_unload_stops_response_reporting_for_a_cached_client(response_plugin, model_events):
    def respond(request):
        return httpx.Response(403, json={"error": {"message": "denied"}}, headers={"x-test-response-id": str(uuid4())})

    with _client("openai", respond) as client:
        with pytest.raises(openai.PermissionDeniedError):
            _execute(client)
        nemo_relay.subscribers.flush()
        before = [event for event in model_events if event.name == "test.http.response"]
        assert len(before) == 1
        response_plugin.unload("response-observer")
        with pytest.raises(openai.PermissionDeniedError):
            _execute(client)
    nemo_relay.subscribers.flush()
    assert [event for event in model_events if event.name == "test.http.response"] == before


def test_async_sdk_response_keeps_the_physical_model_parent(model_events):
    forwarded = {}

    def respond(request):
        return _response(request, forwarded, json=_body())

    async def call():
        async with openai.AsyncOpenAI(
            api_key="test-key", base_url="https://api.example.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            install_llm_response_hook(client)
            result = await relay_llm.execute_async(
                _request(), lambda request: client.chat.completions.create(**request),
                name="openai", model_name="test-model", session_id="response-session",
                metadata={"api_mode": "chat_completions"},
            )
            assert result.choices[0].message.content == "hello"
    asyncio.run(call())
    _assert_response_parents(model_events, forwarded)
