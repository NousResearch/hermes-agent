"""Tests for the core Relay-managed physical LLM attempt adapter."""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_llm, relay_runtime


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-1",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    lease.host.retain_managed_execution("test.relay_llm")
    try:
        yield lease.host.relay, turn
    finally:
        lease.host.release_managed_execution("test.relay_llm")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_stream_uses_rewritten_request_and_post_intercept_chunks(relay_turn):
    relay, turn = relay_turn
    captured_requests = []
    validated_requests = []

    def rewrite_request(name, request, annotated):
        del name
        content = {**request.content, "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(
            relay.LLMRequest(request.headers, content),
            annotated,
        )

    def rewrite_stream(request, next_call):
        async def generate():
            upstream = await next_call(request)
            async for chunk in upstream:
                updated = dict(chunk)
                choices = [dict(choice) for choice in updated.get("choices", [])]
                if choices:
                    delta = dict(choices[0].get("delta") or {})
                    if delta.get("content"):
                        delta["content"] = delta["content"].upper()
                    choices[0]["delta"] = delta
                    updated["choices"] = choices
                yield updated

        return generate()

    def raw_stream(request):
        captured_requests.append(request)
        return iter([
            SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hello", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ])

    relay.intercepts.register_llm_request(
        "hermes-test-request",
        1,
        False,
        rewrite_request,
    )
    relay.intercepts.register_llm_stream_execution(
        "hermes-test-stream",
        1,
        rewrite_stream,
    )
    try:
        stream = relay_llm.stream(
            {
                "model": "test-model",
                "messages": [],
                "extra_headers": {"authorization": "Bearer provider-token"},
            },
            raw_stream,
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            finalizer=lambda: {
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "HELLO"},
                        "finish_reason": "stop",
                    }
                ],
            },
            metadata={
                "api_mode": "custom",
                "api_request_id": "request-1",
                "call_role": "primary",
            },
            request_validator=validated_requests.append,
        )
        chunks = list(stream)
    finally:
        relay.intercepts.deregister_llm_stream_execution("hermes-test-stream")
        relay.intercepts.deregister_llm_request("hermes-test-request")

    assert captured_requests[0]["temperature"] == 0.25
    assert validated_requests == captured_requests
    assert captured_requests[0]["extra_headers"] == {
        "authorization": "Bearer provider-token"
    }
    assert chunks[0].choices[0].delta.content == "HELLO"
    assert stream.output_modified is True
    assert turn.logical_llm_calls == {}












def test_anthropic_stream_accumulator_merges_plain_provider_object():
    accumulator = relay_llm.AnthropicStreamAccumulator()
    accumulator.observe({
        "type": "message_start",
        "message": {
            "id": "message-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "usage": {"input_tokens": 10},
        },
    })
    accumulator.observe({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": "hello"},
    })

    response = accumulator.response(
        SimpleNamespace(
            id="message-1",
            type="message",
            role="assistant",
            model="claude-test",
            content=[],
            stop_reason=None,
            usage={"input_tokens": 10},
        )
    )

    assert response.id == "message-1"
    assert response.content[0].text == "hello"
    assert response.usage.input_tokens == 10


def test_jsonable_does_not_probe_dynamic_attributes():
    class DynamicProviderObject:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected dynamic attribute lookup: {name}")

        def __str__(self):
            return "opaque-provider-object"

    assert relay_llm._jsonable(DynamicProviderObject()) == "opaque-provider-object"






@pytest.mark.asyncio
async def test_async_provider_callback_preserves_caller_context(relay_turn):
    del relay_turn
    caller_value = contextvars.ContextVar(
        "async_llm_caller_value",
        default="default",
    )
    caller_value.set("caller")

    async def provider(_request):
        await asyncio.sleep(0)
        return {"caller_value": caller_value.get()}

    result = await relay_llm.execute_async(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-async-context",
        },
    )

    assert result == {"caller_value": "caller"}




def test_anthropic_stream_callbacks_do_not_reenter_captured_context(
    relay_turn,
    monkeypatch,
):
    del relay_turn
    caller_value = contextvars.ContextVar(
        "anthropic_stream_caller_value",
        default="default",
    )
    caller_value.set("caller")
    callback_context = contextvars.copy_context()
    real_copy_context = contextvars.copy_context
    copy_count = 0

    def capture_callback_context():
        nonlocal copy_count
        copy_count += 1
        if copy_count == 1:
            return callback_context
        return real_copy_context()

    monkeypatch.setattr(
        relay_llm.contextvars,
        "copy_context",
        capture_callback_context,
    )
    observed = []
    accumulator = relay_llm.AnthropicStreamAccumulator()

    def observe_chunk(chunk):
        observed.append(caller_value.get())
        accumulator.observe(chunk)

    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
    ]
    stream = relay_llm.stream(
        {
            "model": "claude-test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        lambda _request: iter(chunks),
        session_id="session-1",
        name="anthropic",
        model_name="claude-test",
        finalizer=accumulator.finalize,
        on_chunk=observe_chunk,
        metadata={
            "api_mode": "anthropic_messages",
            "api_request_id": "request-anthropic-context-reentry",
        },
    )

    entered = threading.Event()
    release = threading.Event()

    def hold_callback_context() -> None:
        def wait() -> None:
            entered.set()
            assert release.wait(timeout=5)

        callback_context.run(wait)

    holder = threading.Thread(target=hold_callback_context)
    holder.start()
    assert entered.wait(timeout=1)
    try:
        assert list(stream) == chunks
    finally:
        release.set()
        holder.join(timeout=1)

    assert holder.is_alive() is False
    assert observed == ["caller", "caller"]


def test_explicit_stream_close_surfaces_provider_close_failure(relay_turn):
    del relay_turn

    class FailingCloseStream:
        def __init__(self):
            self._chunks = iter([{"delta": "partial"}])
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._chunks)

        def close(self):
            self.close_calls += 1
            raise RuntimeError("provider close failed")

    raw_stream = FailingCloseStream()
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: raw_stream,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": "partial"},
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-close-failure",
        },
    )

    assert next(stream) == {"delta": "partial"}
    with pytest.raises(RuntimeError, match="provider close failed"):
        stream.close()

    assert raw_stream.close_calls == 1
    stream.close()




def test_non_stream_defers_logical_success_and_reuses_scope_for_retry(relay_turn):
    _relay, turn = relay_turn
    metadata = {"api_mode": "custom", "api_request_id": "request-retry"}

    first = relay_llm.execute(
        {"model": "test-model", "messages": []},
        lambda _request: {"content": "invalid"},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata=metadata,
        defer_logical_completion=True,
    )
    first_handle = turn.logical_llm_calls["request-retry"]

    second = relay_llm.execute(
        {"model": "test-model", "messages": []},
        lambda _request: {"content": "valid"},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata=metadata,
        defer_logical_completion=True,
    )

    assert first == {"content": "invalid"}
    assert second == {"content": "valid"}
    assert turn.logical_llm_calls == {"request-retry": first_handle}

    relay_llm.complete_logical_call("request-retry", outcome="success")

    assert turn.logical_llm_calls == {}


def test_non_stream_validates_final_relay_request(relay_turn):
    relay, _turn = relay_turn
    validated_requests = []
    provider_requests = []

    def rewrite_request(name, request, annotated):
        del name
        content = {**request.content, "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(
            relay.LLMRequest(request.headers, content),
            annotated,
        )

    relay.intercepts.register_llm_request(
        "hermes-test-validator",
        1,
        False,
        rewrite_request,
    )
    try:
        relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda request: provider_requests.append(request) or {"content": "ok"},
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            request_validator=validated_requests.append,
        )
    finally:
        relay.intercepts.deregister_llm_request("hermes-test-validator")

    assert validated_requests == provider_requests
    assert validated_requests[0]["temperature"] == 0.25


def test_non_stream_result_survives_logical_scope_close_failure(
    relay_turn, monkeypatch
):
    relay, turn = relay_turn
    original_pop = relay.scope.pop
    pop_calls = 0

    def fail_first_pop(*args, **kwargs):
        nonlocal pop_calls
        pop_calls += 1
        if pop_calls == 1:
            raise RuntimeError("simulated logical scope close failure")
        return original_pop(*args, **kwargs)

    monkeypatch.setattr(relay.scope, "pop", fail_first_pop)
    raw_response = SimpleNamespace(model="test-model", content="raw")

    result = relay_llm.execute(
        {"model": "test-model", "messages": []},
        lambda _request: raw_response,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-close"},
    )

    assert result is raw_response
    assert "request-close" in turn.logical_llm_calls
    relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
    assert turn.logical_llm_calls == {}
















def test_stream_flushes_buffered_provider_chunks_after_relay_failure(
    relay_turn, monkeypatch
):
    relay, turn = relay_turn
    raw_chunks = [{"delta": "first"}, {"delta": "second"}]

    async def fail_with_buffered_chunk(
        _name,
        request,
        callback,
        observe_chunk,
        finalizer,
        **_kwargs,
    ):
        async def generate():
            upstream = callback(request)
            first = await anext(upstream)
            observe_chunk(first)
            yield first
            second = await anext(upstream)
            observe_chunk(second)
            with pytest.raises(StopAsyncIteration):
                await anext(upstream)
            finalizer()
            raise RuntimeError("simulated buffered Relay failure")

        return generate()

    monkeypatch.setattr(relay.llm, "stream_execute", fail_with_buffered_chunk)
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: iter(raw_chunks),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": "complete"},
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-buffered-failure",
        },
    )

    assert list(stream) == raw_chunks
    assert turn.logical_llm_calls == {}
















def test_bypassed_stream_still_honors_chunk_acceptance(relay_turn):
    _relay, turn = relay_turn
    turn.lease.host.release_managed_execution("test.relay_llm")
    provider_closed = []

    def provider_stream(_request):
        try:
            yield {"delta": "accepted"}
            yield {"delta": "rejected"}
            yield {"delta": "unreachable"}
        finally:
            provider_closed.append(True)

    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        provider_stream,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=dict,
        accept_chunk=lambda chunk: chunk["delta"] != "rejected",
    )

    assert list(stream) == [{"delta": "accepted"}]
    assert provider_closed == [True]


def test_anthropic_codec_preserves_tool_history_and_cached_system_blocks(relay_turn):
    _relay, _turn = relay_turn
    request = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 512,
        "system": [
            {
                "type": "text",
                "text": "You are Hermes.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Run pwd"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "terminal",
                        "input": {"command": "pwd"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": [{"type": "text", "text": "/tmp/worktree"}],
                    }
                ],
            },
        ],
    }
    original_wire = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    observed_body_wire = ""

    def provider(final_request):
        nonlocal observed_body_wire
        provider_body = {
            key: value for key, value in final_request.items() if key != "extra_headers"
        }
        observed_body_wire = json.dumps(
            provider_body,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Done"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }

    relay_llm.execute(
        request,
        provider,
        session_id="session-1",
        name="anthropic",
        model_name="claude-sonnet-4-5",
        metadata={
            "api_mode": "anthropic_messages",
            "api_request_id": "request-anthropic",
        },
    )

    assert observed_body_wire == original_wire






@pytest.mark.asyncio
async def test_async_non_stream_returns_namespaced_interceptor_result(
    relay_turn,
    monkeypatch,
):
    relay, _turn = relay_turn

    async def post_execute(_name, request, callback, **_kwargs):
        response = await callback(request)
        return {
            **response,
            "post_interceptor": True,
            "usage": {"input_tokens": 10},
        }

    monkeypatch.setattr(relay.llm, "execute", post_execute)

    async def provider(_request):
        return {"content": "raw"}

    result = await relay_llm.execute_async(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-async-post"},
    )

    assert result.content == "raw"
    assert result.post_interceptor is True
    assert result.usage.input_tokens == 10


def test_non_stream_preserves_provider_error_from_relay_wrapper_suffix(
    relay_turn, monkeypatch
):
    relay, turn = relay_turn

    class ProviderError(Exception):
        pass

    provider_error = ProviderError("provider failed")

    async def wrapping_execute(_name, request, callback, **_kwargs):
        try:
            return callback(request)
        except Exception as exc:
            raise RuntimeError(
                f"internal error: {type(exc).__name__}: {exc} (retried 3x)"
            ) from None

    monkeypatch.setattr(relay.llm, "execute", wrapping_execute)

    with pytest.raises(ProviderError) as caught:
        relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: (_ for _ in ()).throw(provider_error),
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            metadata={"api_mode": "custom", "api_request_id": "request-error"},
        )

    assert caught.value is provider_error
    assert "request-error" in turn.logical_llm_calls












def test_codec_baseline_failure_is_explicit(relay_turn, monkeypatch, caplog):
    relay, _turn = relay_turn
    request_body = {"model": "test-model", "messages": []}
    request = relay.LLMRequest({}, request_body)

    class FailingCodec:
        def decode(self, _request):
            raise RuntimeError("simulated codec failure")

    monkeypatch.setattr(relay_llm, "_codec", lambda *_args, **_kwargs: FailingCodec())

    with caplog.at_level("WARNING", logger="agent.relay_llm"):
        baseline = relay_llm._codec_round_trip_request_body(
            relay,
            request,
            relay_request_body=request_body,
            metadata={"api_mode": "chat_completions"},
        )

    assert baseline is None
    assert "ignoring request rewrites" in caplog.text


def test_stream_current_unwraps_completed_response(tmp_path, monkeypatch):
    """Auxiliary streaming (the MoA aggregator) must surface a completed
    provider response raw instead of crashing when the client ignores
    ``stream=True`` and returns a response object (AnthropicAuxiliaryClient
    and other OpenAI-compatible shims).

    Pre-Relay, ``call_llm(stream=True)`` returned the raw response and the
    consumer's ``hasattr(stream, "choices")`` check handled it (#11732,
    #55933). The Relay integration wrapped the call in a ManagedLlmStream
    without threading ``completed_response_predicate``, regressing that path
    into ``TypeError: 'types.SimpleNamespace' object is not iterable``.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-moa",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-moa",
        task_id="task-moa",
    )
    try:
        completed = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done"),
                    finish_reason="stop",
                )
            ],
            model="kimi-k3",
        )
        result = relay_llm.stream_current(
            {"model": "kimi-k3", "stream": True},
            lambda request: completed,
            name="kimi-coding",
            model_name="kimi-k3",
            finalizer=dict,
            completed_response_predicate=lambda value: hasattr(value, "choices"),
        )
        # Unwrapped raw response — NOT a stream wrapper whose iteration would
        # have raised TypeError pre-fix.
        assert result is completed
    finally:
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_stream_current_streams_iterators_with_predicate(tmp_path, monkeypatch):
    """A genuine chunk iterator still flows through as a stream when the
    completed-response predicate is supplied."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-moa",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-moa",
        task_id="task-moa",
    )
    try:
        result = relay_llm.stream_current(
            {"model": "m", "stream": True},
            lambda request: iter([{"delta": "a"}, {"delta": "b"}]),
            name="provider",
            model_name="m",
            finalizer=dict,
            completed_response_predicate=lambda value: hasattr(value, "choices"),
        )
        assert list(result) == [{"delta": "a"}, {"delta": "b"}]
    finally:
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()



def _completed_response(content: str = "done") -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def _choices_predicate(value) -> bool:
    return hasattr(value, "choices")


def test_stream_managed_traps_direct_completed_response(relay_turn):
    """Managed path: a factory returning a completed response (adapter
    ignoring stream=True) is trapped as final_response instead of iterated."""
    relay, turn = relay_turn
    del relay, turn

    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda request: _completed_response(),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {},
        completed_response_predicate=_choices_predicate,
    )
    assert list(stream) == []
    assert stream.final_response is not None
    assert stream.final_response.choices[0].message.content == "done"


def test_stream_current_inside_managed_callback_returns_raw(relay_turn):
    """Managed path: an auxiliary stream_current() call made from inside a
    managed provider callback (the MoA facade's call_llm(stream=True) shape)
    must return the raw factory result; the outer stream traps a completed
    response as its final_response instead of crashing on a nested event
    loop or surfacing an empty stream."""
    relay, turn = relay_turn
    del relay, turn

    def outer_factory(request):
        return relay_llm.stream_current(
            {"model": "test-model", "messages": []},
            lambda inner_request: _completed_response(),
            name="moa-aggregator",
            model_name="test-model",
            finalizer=lambda: {},
            completed_response_predicate=_choices_predicate,
        )

    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        outer_factory,
        session_id="session-1",
        name="moa",
        model_name="test-model",
        finalizer=lambda: {},
        completed_response_predicate=_choices_predicate,
    )
    assert list(stream) == []
    assert stream.final_response is not None
    assert stream.final_response.choices[0].message.content == "done"
