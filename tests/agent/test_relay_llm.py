"""Tests for the core Relay-managed physical LLM attempt adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    try:
        yield lease.host.relay, turn
    finally:
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
            {"model": "test-model", "messages": []},
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
    assert chunks[0].choices[0].delta.content == "HELLO"
    assert turn.logical_llm_calls == {}


def test_stream_preserves_provider_error_and_keeps_logical_scope_for_retry(relay_turn):
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
    )

    with pytest.raises(ProviderError) as caught:
        list(stream)

    assert caught.value is provider_error
    assert "request-2" in turn.logical_llm_calls


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

    assert result == {"content": "raw", "post_interceptor": True}


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
