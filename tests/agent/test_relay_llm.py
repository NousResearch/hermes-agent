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
        )
        chunks = list(stream)
    finally:
        relay.intercepts.deregister_llm_stream_execution("hermes-test-stream")
        relay.intercepts.deregister_llm_request("hermes-test-request")

    assert captured_requests[0]["temperature"] == 0.25
    assert captured_requests[0]["extra_headers"] == {
        "authorization": "Bearer provider-token"
    }
    assert chunks[0].choices[0].delta.content == "HELLO"
    assert stream.output_modified is True
    assert turn.logical_llm_calls == {}


def test_deferred_stream_preserves_provider_error_and_logical_scope_for_retry(
    relay_turn,
):
    _relay, turn = relay_turn

    class ProviderError(Exception):
        pass

    provider_error = ProviderError("provider failed")

    def failing_stream(_request):
        def generate():
            raise provider_error
            yield  # pragma: no cover

        return generate()

    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        failing_stream,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=dict,
        metadata={
            "api_mode": "chat_completions",
            "api_request_id": "request-2",
        },
        defer_logical_completion=True,
    )

    with pytest.raises(ProviderError) as caught:
        list(stream)

    assert caught.value is provider_error
    assert "request-2" in turn.logical_llm_calls


def test_stream_provider_error_is_not_replaced_by_finalizer_error(relay_turn):
    _relay, turn = relay_turn

    class ProviderError(Exception):
        pass

    provider_error = ProviderError("provider failed before first chunk")
    finalizer_called = False

    def failing_stream(_request):
        def generate():
            raise provider_error
            yield  # pragma: no cover

        return generate()

    def failing_finalizer():
        nonlocal finalizer_called
        finalizer_called = True
        raise RuntimeError("missing terminal response")

    stream = relay_llm.stream(
        {"model": "test-model", "input": "hi"},
        failing_stream,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=failing_finalizer,
        metadata={
            "api_mode": "codex_responses",
            "api_request_id": "request-provider-before-finalizer",
        },
        defer_logical_completion=True,
    )

    with pytest.raises(ProviderError) as caught:
        list(stream)

    assert caught.value is provider_error
    assert finalizer_called is False
    assert "request-provider-before-finalizer" in turn.logical_llm_calls


def test_non_deferred_partial_stream_close_cancels_logical_call(
    relay_turn,
    monkeypatch,
):
    relay, turn = relay_turn
    original_pop = relay.scope.pop
    terminal_outputs = []

    def record_pop(handle, *args, **kwargs):
        terminal_outputs.append(kwargs.get("output"))
        return original_pop(handle, *args, **kwargs)

    monkeypatch.setattr(relay.scope, "pop", record_pop)
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: iter([{"delta": "partial"}, {"delta": "unused"}]),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": "partial"},
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-partial-close",
        },
    )

    assert next(stream) == {"delta": "partial"}
    assert "request-partial-close" in turn.logical_llm_calls

    stream.close()

    assert "request-partial-close" not in turn.logical_llm_calls
    assert {"outcome": "cancelled"} in terminal_outputs


def test_direct_stream_close_reaches_original_provider_resource(monkeypatch):
    class ProviderStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter([{"delta": "partial"}, {"delta": "unused"}])

        def close(self):
            self.closed = True

    provider_stream = ProviderStream()
    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (None, None, None),
    )

    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: provider_stream,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=dict,
    )

    assert next(stream) == {"delta": "partial"}
    stream.close()

    assert provider_stream.closed is True


def test_anthropic_stream_accumulator_merges_terminal_usage():
    accumulator = relay_llm.AnthropicStreamAccumulator()
    accumulator.observe({
        "type": "message_start",
        "message": {
            "id": "message-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 1,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            },
        },
    })
    accumulator.observe({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 12},
    })

    response = accumulator.finalize()

    assert response["usage"] == {
        "input_tokens": 100,
        "output_tokens": 12,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 30,
    }


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


def test_non_stream_preserves_raw_provider_response_identity(relay_turn):
    _relay, _turn = relay_turn
    raw_response = SimpleNamespace(model="test-model", content="raw")

    result = relay_llm.execute(
        {"model": "test-model", "messages": []},
        lambda _request: raw_response,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-raw"},
    )

    assert result is raw_response


def test_non_stream_provider_callback_preserves_caller_context(relay_turn):
    del relay_turn
    caller_value = contextvars.ContextVar("llm_caller_value", default="default")
    caller_value.set("caller")

    result = relay_llm.execute(
        {"model": "test-model", "messages": []},
        lambda _request: {"caller_value": caller_value.get()},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-context"},
    )

    assert result == {"caller_value": "caller"}


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


def test_stream_provider_callbacks_preserve_caller_context(relay_turn):
    del relay_turn
    caller_value = contextvars.ContextVar(
        "stream_llm_caller_value",
        default="default",
    )
    caller_value.set("caller")
    observed = []

    def stream_factory(_request):
        observed.append(("factory", caller_value.get()))

        def generate():
            observed.append(("next", caller_value.get()))
            yield {"delta": "hello"}

        return generate()

    def on_chunk(_chunk):
        observed.append(("chunk", caller_value.get()))

    def finalizer():
        observed.append(("finalizer", caller_value.get()))
        return {"content": "hello"}

    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        stream_factory,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=finalizer,
        on_chunk=on_chunk,
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-stream-context",
        },
    )

    assert list(stream) == [{"delta": "hello"}]
    assert observed == [
        ("factory", "caller"),
        ("next", "caller"),
        ("chunk", "caller"),
        ("finalizer", "caller"),
    ]


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


def test_non_stream_does_not_forward_relay_session_headers(relay_turn):
    _relay, _turn = relay_turn
    captured_requests = []

    relay_llm.execute(
        {
            "model": "test-model",
            "messages": [],
            "extra_headers": {"x-provider-header": "provider-value"},
        },
        lambda request: captured_requests.append(request) or {"content": "ok"},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-headers"},
    )

    assert captured_requests[0]["extra_headers"] == {
        "x-provider-header": "provider-value"
    }


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


def test_non_stream_returns_provider_response_after_relay_post_processing_failure(
    relay_turn, monkeypatch, caplog
):
    relay, turn = relay_turn
    raw_response = SimpleNamespace(model="test-model", content="raw")

    async def fail_after_callback(_name, request, callback, **_kwargs):
        callback(request)
        raise RuntimeError("simulated Relay post-processing failure")

    monkeypatch.setattr(relay.llm, "execute", fail_after_callback)

    with caplog.at_level("WARNING", logger="agent.relay_llm"):
        result = relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: raw_response,
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "custom",
                "api_request_id": "request-post-failure",
            },
        )

    assert result is raw_response
    assert turn.logical_llm_calls == {}
    assert "returning the provider response" in caplog.text


def test_non_stream_does_not_swallow_interrupt_after_provider_success(
    relay_turn, monkeypatch
):
    relay, turn = relay_turn

    async def interrupt_after_callback(_name, request, callback, **_kwargs):
        callback(request)
        raise KeyboardInterrupt

    monkeypatch.setattr(relay.llm, "execute", interrupt_after_callback)

    with pytest.raises(KeyboardInterrupt):
        relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: {"content": "already returned"},
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "custom",
                "api_request_id": "request-post-interrupt",
            },
        )

    assert "request-post-interrupt" in turn.logical_llm_calls
    relay_llm.complete_logical_call("request-post-interrupt", outcome="cancelled")


@pytest.mark.asyncio
async def test_async_non_stream_preserves_raw_provider_response_identity(relay_turn):
    _relay, _turn = relay_turn
    raw_response = SimpleNamespace(model="test-model", content="raw")

    async def provider(_request):
        return raw_response

    result = await relay_llm.execute_current_async(
        {"model": "test-model", "messages": []},
        provider,
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-async"},
    )

    assert result is raw_response


@pytest.mark.asyncio
async def test_async_non_stream_returns_provider_response_after_relay_failure(
    relay_turn, monkeypatch, caplog
):
    relay, turn = relay_turn
    raw_response = SimpleNamespace(model="test-model", content="raw")

    async def provider(_request):
        return raw_response

    async def fail_after_callback(_name, request, callback, **_kwargs):
        await callback(request)
        raise RuntimeError("simulated Relay post-processing failure")

    monkeypatch.setattr(relay.llm, "execute", fail_after_callback)

    with caplog.at_level("WARNING", logger="agent.relay_llm"):
        result = await relay_llm.execute_current_async(
            {"model": "test-model", "messages": []},
            provider,
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "custom",
                "api_request_id": "request-async-post-failure",
            },
        )

    assert result is raw_response
    assert turn.logical_llm_calls == {}
    assert "returning the provider response" in caplog.text


@pytest.mark.asyncio
async def test_async_non_stream_does_not_swallow_cancellation_after_provider_success(
    relay_turn, monkeypatch
):
    relay, turn = relay_turn

    async def provider(_request):
        return {"content": "already returned"}

    async def cancel_after_callback(_name, request, callback, **_kwargs):
        await callback(request)
        raise asyncio.CancelledError

    monkeypatch.setattr(relay.llm, "execute", cancel_after_callback)

    with pytest.raises(asyncio.CancelledError):
        await relay_llm.execute_current_async(
            {"model": "test-model", "messages": []},
            provider,
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "custom",
                "api_request_id": "request-async-post-cancel",
            },
        )

    assert "request-async-post-cancel" in turn.logical_llm_calls
    relay_llm.complete_logical_call(
        "request-async-post-cancel",
        outcome="cancelled",
    )


@pytest.mark.asyncio
async def test_async_non_stream_defers_logical_success_for_validation(relay_turn):
    _relay, turn = relay_turn

    async def provider(_request):
        return {"content": "pending-validation"}

    await relay_llm.execute_current_async(
        {"model": "test-model", "messages": []},
        provider,
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-async-defer"},
        defer_logical_completion=True,
    )

    assert "request-async-defer" in turn.logical_llm_calls

    relay_llm.complete_logical_call("request-async-defer", outcome="success")

    assert turn.logical_llm_calls == {}


def test_stream_finishes_after_relay_post_processing_failure(
    relay_turn, monkeypatch, caplog
):
    relay, turn = relay_turn

    async def fail_after_stream(
        _name,
        request,
        callback,
        observe_chunk,
        finalizer,
        **_kwargs,
    ):
        async def generate():
            upstream = callback(request)
            async for chunk in upstream:
                observe_chunk(chunk)
                yield chunk
            finalizer()
            raise RuntimeError("simulated Relay post-processing failure")

        return generate()

    monkeypatch.setattr(relay.llm, "stream_execute", fail_after_stream)
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: iter([{"delta": "complete"}]),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": "complete"},
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-stream-post-failure",
        },
    )

    with caplog.at_level("WARNING", logger="agent.relay_llm"):
        chunks = list(stream)

    assert chunks == [{"delta": "complete"}]
    assert turn.logical_llm_calls == {}
    assert "preserving the provider result" in caplog.text


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


def test_stream_constructor_flushes_provider_chunks_after_relay_failure(
    relay_turn, monkeypatch
):
    relay, turn = relay_turn
    raw_chunks = [{"delta": "first"}, {"delta": "second"}]

    async def fail_during_stream_setup(
        _name,
        request,
        callback,
        observe_chunk,
        finalizer,
        **_kwargs,
    ):
        upstream = callback(request)
        async for chunk in upstream:
            observe_chunk(chunk)
        finalizer()
        raise RuntimeError("simulated Relay setup failure")

    monkeypatch.setattr(relay.llm, "stream_execute", fail_during_stream_setup)
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: iter(raw_chunks),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": "complete"},
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-setup-failure",
        },
    )

    assert list(stream) == raw_chunks
    assert turn.logical_llm_calls == {}


def test_stream_does_not_swallow_interrupt_after_provider_success(
    relay_turn, monkeypatch
):
    relay, turn = relay_turn

    async def interrupt_after_stream(
        _name,
        request,
        callback,
        observe_chunk,
        finalizer,
        **_kwargs,
    ):
        async def generate():
            upstream = callback(request)
            async for chunk in upstream:
                observe_chunk(chunk)
                yield chunk
            finalizer()
            raise KeyboardInterrupt

        return generate()

    monkeypatch.setattr(relay.llm, "stream_execute", interrupt_after_stream)
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: iter([{"delta": "complete"}]),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": "complete"},
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-stream-post-interrupt",
        },
    )

    assert next(stream) == {"delta": "complete"}
    with pytest.raises(KeyboardInterrupt):
        next(stream)
    assert turn.logical_llm_calls == {}


def test_stream_does_not_swallow_hermes_finalizer_failure(relay_turn, monkeypatch):
    relay, _turn = relay_turn
    finalizer_error = RuntimeError("Hermes finalizer failed")

    def fail_finalizer():
        raise finalizer_error

    async def execute_stream(
        _name,
        request,
        callback,
        _observe_chunk,
        finalizer,
        **_kwargs,
    ):
        async def generate():
            upstream = callback(request)
            async for chunk in upstream:
                yield chunk
            finalizer()

        return generate()

    monkeypatch.setattr(relay.llm, "stream_execute", execute_stream)
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: iter([{"delta": "complete"}]),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=fail_finalizer,
        metadata={
            "api_mode": "custom",
            "api_request_id": "request-finalizer-failure",
        },
    )

    with pytest.raises(Exception) as caught:
        list(stream)

    assert caught.value is finalizer_error


def test_stream_defers_logical_success_for_response_validation(relay_turn):
    _relay, turn = relay_turn

    stream = relay_llm.stream(
        {"model": "test-model", "messages": []},
        lambda _request: iter([{"delta": "pending-validation"}]),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": "pending-validation"},
        metadata={"api_mode": "custom", "api_request_id": "request-stream-defer"},
        defer_logical_completion=True,
    )

    assert list(stream) == [{"delta": "pending-validation"}]
    assert stream.output_modified is False
    assert "request-stream-defer" in turn.logical_llm_calls

    relay_llm.complete_logical_call("request-stream-defer", outcome="success")

    assert turn.logical_llm_calls == {}


def test_current_attempt_bypasses_relay_without_an_active_turn(monkeypatch):
    monkeypatch.setattr(relay_runtime, "current_turn", lambda: None)
    request = {"model": "test-model", "messages": []}

    result = relay_llm.execute_current(
        request,
        lambda value: value,
        name="test-provider",
        model_name="test-model",
    )

    assert result is request


def test_non_stream_bypasses_relay_without_an_active_consumer(relay_turn, monkeypatch):
    relay, turn = relay_turn
    turn.lease.host.release_managed_execution("test.relay_llm")
    request = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    monkeypatch.setattr(
        relay.llm,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inactive Relay must not manage the provider call")
        ),
    )

    result = relay_llm.execute(
        request,
        lambda value: value,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
    )

    assert result is request


def test_stream_bypasses_relay_without_an_active_consumer(relay_turn, monkeypatch):
    relay, turn = relay_turn
    turn.lease.host.release_managed_execution("test.relay_llm")
    request = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    observed = []

    monkeypatch.setattr(
        relay.llm,
        "stream_execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inactive Relay must not manage the provider stream")
        ),
    )

    stream = relay_llm.stream(
        request,
        lambda value: (observed.append(value), iter([{"delta": "ok"}]))[1],
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=dict,
    )

    assert list(stream) == [{"delta": "ok"}]
    assert observed == [request]


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


def test_current_attempt_bypasses_a_closed_turn_from_a_copied_context(
    relay_turn,
    monkeypatch,
):
    _relay, turn = relay_turn
    stale_context = contextvars.copy_context()
    relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
    request = {"model": "test-model", "messages": []}

    def fail_execute(*_args, **_kwargs):
        raise AssertionError("a closed turn must not manage later provider work")

    monkeypatch.setattr(relay_llm, "execute", fail_execute)

    result = stale_context.run(
        relay_llm.execute_current,
        request,
        lambda value: value,
        name="test-provider",
        model_name="test-model",
    )

    assert result is request


def test_non_stream_returns_post_execution_interceptor_result(relay_turn, monkeypatch):
    relay, _turn = relay_turn

    async def post_execute(_name, request, callback, **_kwargs):
        response = callback(request)
        return {**response, "post_interceptor": True}

    monkeypatch.setattr(relay.llm, "execute", post_execute)

    result = relay_llm.execute(
        {"model": "test-model", "messages": []},
        lambda _request: {"content": "raw"},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-post"},
    )

    assert result.content == "raw"
    assert result.post_interceptor is True


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


def test_non_stream_does_not_mask_relay_error_after_callback_failure(
    relay_turn, monkeypatch
):
    relay, _turn = relay_turn
    provider_error = RuntimeError("provider failed")
    relay_error = RuntimeError("internal error: RelayPolicyError: policy blocked")

    async def translating_execute(_name, request, callback, **_kwargs):
        try:
            callback(request)
        except Exception:
            raise relay_error

    monkeypatch.setattr(relay.llm, "execute", translating_execute)

    with pytest.raises(RuntimeError) as caught:
        relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: (_ for _ in ()).throw(provider_error),
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            metadata={"api_mode": "custom", "api_request_id": "request-policy"},
        )

    assert caught.value is relay_error


def test_chat_codec_preserves_provider_message_extensions_after_rewrite(relay_turn):
    relay, _turn = relay_turn
    captured_requests = []

    def rewrite_request(name, request, annotated):
        del name
        annotated.params = {**(annotated.params or {}), "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(request, annotated)

    def provider(request):
        captured_requests.append(request)
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    relay.intercepts.register_llm_request(
        "hermes-provider-extension-request",
        1,
        False,
        rewrite_request,
    )
    try:
        relay_llm.execute(
            {
                "model": "test-model",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "provider scratchpad",
                    }
                ],
            },
            provider,
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "request-3",
            },
        )
    finally:
        relay.intercepts.deregister_llm_request(
            "hermes-provider-extension-request"
        )

    assert captured_requests[0]["temperature"] == 0.25
    assert captured_requests[0]["messages"][0]["reasoning_content"] == (
        "provider scratchpad"
    )


def test_request_rewrite_preserves_unmodified_provider_objects(relay_turn):
    relay, _turn = relay_turn
    timeout = object()
    captured_requests = []

    def rewrite_request(name, request, annotated):
        del name
        annotated.params = {**(annotated.params or {}), "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(request, annotated)

    relay.intercepts.register_llm_request(
        "hermes-provider-object-request",
        1,
        False,
        rewrite_request,
    )
    try:
        relay_llm.execute(
            {"model": "test-model", "messages": [], "timeout": timeout},
            lambda request: captured_requests.append(request) or {"content": "ok"},
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "request-provider-object",
            },
        )
    finally:
        relay.intercepts.deregister_llm_request(
            "hermes-provider-object-request"
        )

    assert captured_requests[0]["timeout"] is timeout
    assert captured_requests[0]["temperature"] == 0.25


def test_request_rewrite_preserves_fields_dropped_by_codec(relay_turn, monkeypatch):
    relay, _turn = relay_turn
    captured_requests = []
    vendor_body = {
        "routing": {"provider": "nim", "region": "us-west-2"},
        "trace_vendor_request": False,
    }

    async def lossy_execute(_name, request, callback, **_kwargs):
        content = {
            key: value
            for key, value in request.content.items()
            if key != "extra_body"
        }
        content["temperature"] = 0.25
        return callback(relay.LLMRequest(request.headers, content))

    monkeypatch.setattr(relay.llm, "execute", lossy_execute)
    monkeypatch.setattr(
        relay_llm,
        "_codec_round_trip_request_body",
        lambda *_args, relay_request_body, **_kwargs: {
            key: value
            for key, value in relay_request_body.items()
            if key != "extra_body"
        },
    )

    relay_llm.execute(
        {
            "model": "test-model",
            "messages": [],
            "temperature": 0.0,
            "extra_body": vendor_body,
        },
        lambda request: captured_requests.append(request) or {"content": "ok"},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={
            "api_mode": "chat_completions",
            "api_request_id": "request-lossy-codec",
        },
    )

    assert captured_requests == [
        {
            "model": "test-model",
            "messages": [],
            "temperature": 0.25,
            "extra_body": vendor_body,
        }
    ]


def test_request_rewrite_is_ignored_when_codec_baseline_fails(
    relay_turn, monkeypatch
):
    relay, _turn = relay_turn
    captured_requests = []
    original = {
        "model": "test-model",
        "messages": [],
        "temperature": 0.0,
        "extra_body": {"routing": {"provider": "nim"}},
    }

    async def lossy_execute(_name, request, callback, **_kwargs):
        rewritten = {
            key: value
            for key, value in request.content.items()
            if key != "extra_body"
        }
        rewritten["temperature"] = 0.25
        return callback(relay.LLMRequest(request.headers, rewritten))

    monkeypatch.setattr(relay.llm, "execute", lossy_execute)
    monkeypatch.setattr(
        relay_llm,
        "_codec_round_trip_request_body",
        lambda *_args, **_kwargs: None,
    )

    relay_llm.execute(
        original,
        lambda request: captured_requests.append(request) or {"content": "ok"},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={
            "api_mode": "chat_completions",
            "api_request_id": "request-codec-failure",
        },
    )

    assert captured_requests == [original]


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


def test_request_rewrite_can_remove_codec_represented_field(relay_turn, monkeypatch):
    relay, _turn = relay_turn
    captured_requests = []

    async def remove_temperature(_name, request, callback, **_kwargs):
        content = dict(request.content)
        content.pop("temperature")
        return callback(relay.LLMRequest(request.headers, content))

    monkeypatch.setattr(relay.llm, "execute", remove_temperature)

    relay_llm.execute(
        {
            "model": "test-model",
            "messages": [],
            "temperature": 0.25,
            "extra_body": {"routing": {"provider": "nim"}},
        },
        lambda request: captured_requests.append(request) or {"content": "ok"},
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={
            "api_mode": "chat_completions",
            "api_request_id": "request-remove-field",
        },
    )

    assert captured_requests == [
        {
            "model": "test-model",
            "messages": [],
            "extra_body": {"routing": {"provider": "nim"}},
        }
    ]
