"""Physical cached-client/lazy-stream identity and cleanup, inert network."""
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, bind_attempt_identity, current_attempt_identity,
    current_local_refusal, intercepted_openai_class, wrap_httpx_client_transports,
)
from agent.gemini_native_adapter import GeminiNativeClient
from tests.run_agent.test_413_compression import agent as primary_agent  # noqa: F401
from agent.conversation_compression import ProviderBoundRequestOverLimit


def agent(model, window, provider="openai"):
    return SimpleNamespace(model=model, provider=provider, api_mode="chat_completions", base_url="https://inert.invalid", session_id=model, context_compressor=SimpleNamespace(context_length=window))


def test_lazy_gemini_real_dispatch_consumption_and_close_keep_identity():
    seen, closed = [], []
    class InertResponseStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}\n\n'
        def close(self):
            closed.append(current_attempt_identity())
    def receive(request):
        seen.append(current_attempt_identity())
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=InertResponseStream())
    with httpx.Client(transport=httpx.MockTransport(receive)) as http:
        wrap_httpx_client_transports(http)
        client = GeminiNativeClient(api_key="inert", base_url="https://inert.invalid", http_client=http)
        a = agent("gemini-inert", 100000, "gemini")
        stream = _dispatch_nonstreaming_api_request(a, {"model": a.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": True}, make_client=lambda *args, **kw: client)
        assert seen == []  # physical send is genuinely lazy
        assert current_attempt_identity() is None
        next(stream)
        assert seen[0].model == a.model
        assert seen[0].window == 100000
        assert seen[0].family == "gemini_native"
        assert current_attempt_identity() is None
        stream.close()
        assert len(closed) == 1
        assert closed[0].model == a.model
        assert closed[0].family == "gemini_native"
        assert current_attempt_identity() is None
        assert current_local_refusal() is None


def test_shared_cached_sdk_two_thread_physical_retry_identity_and_cleanup(monkeypatch):
    seen, failures, results = [], [], []
    barrier = threading.Barrier(2)
    counts = {}
    lock = threading.Lock()
    def receive(request):
        model = json.loads(request.content)["model"]
        with lock:
            counts[model] = counts.get(model, 0) + 1
            count = counts[model]
            seen.append((model, current_attempt_identity()))
        if count == 1:
            barrier.wait(timeout=5)
            raise httpx.ConnectError("inert remote transient", request=request)
        return httpx.Response(200, json={"id": "inert", "object": "chat.completion", "created": 0, "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})
    http = httpx.Client(transport=httpx.MockTransport(receive))
    wrap_httpx_client_transports(http)
    sdk = intercepted_openai_class(OpenAI)(api_key="inert", base_url="https://inert.invalid", http_client=http, max_retries=1)
    monkeypatch.setattr(sdk, "_calculate_retry_timeout", lambda *a, **kw: 0)
    monkeypatch.setattr("time.sleep", lambda *a: None)
    def work(model, window):
        try:
            a = agent(model, window)
            result = _dispatch_nonstreaming_api_request(a, {"model": model, "messages": [{"role": "user", "content": "hi"}]}, make_client=lambda *args, **kw: sdk)
            results.append(result.choices[0].message.content)
            assert current_attempt_identity() is None
            assert current_local_refusal() is None
        except BaseException as exc:
            failures.append(exc)
    threads = [threading.Thread(target=work, args=("model-a", 1000)), threading.Thread(target=work, args=("model-b", 2000))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert not any(t.is_alive() for t in threads)
        assert failures == []
        assert results == ["ok", "ok"]
        assert counts == {"model-a": 2, "model-b": 2}
        for model, identity in seen:
            assert identity.model == model
            assert identity.window == (1000 if model == "model-a" else 2000)
        assert current_attempt_identity() is None
    finally:
        sdk.close()


def test_actual_dispatch_sdk_extra_body_replacement_is_not_rejected_before_merge():
    calls = []
    def receive(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "inert", "object": "chat.completion", "created": 0, "model": "inert", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})
    with httpx.Client(transport=httpx.MockTransport(receive)) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_openai_class(OpenAI)(api_key="inert", base_url="https://inert.invalid", http_client=http)
        a = agent("inert", 1000)
        result = _dispatch_nonstreaming_api_request(a, {"model": "inert", "messages": [{"role": "user", "content": "x" * 4400}], "extra_body": {"messages": [{"role": "user", "content": "hi"}]}}, make_client=lambda *args, **kw: sdk)
        assert result.choices[0].message.content == "ok"
        assert calls == [{"model": "inert", "messages": [{"role": "user", "content": "hi"}]}]
        assert current_attempt_identity() is None


def test_converse_source_routing_sentinels_not_final_context():
    from agent.conversation_compression import refuse_over_limit_provider_dispatch
    from agent.bedrock_adapter import build_converse_kwargs
    a = agent("inert", 1000, "bedrock")
    a.api_mode = "bedrock_converse"
    kwargs = build_converse_kwargs(model="inert", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=1)
    kwargs.update(__bedrock_region__="us-east-1", __bedrock_converse__=True)
    refuse_over_limit_provider_dispatch(a, kwargs)
    assert kwargs["__bedrock_region__"] == "us-east-1"


def test_actual_outer_loop_local_refusal_no_sdk_sleep_fallback_and_one_refund(primary_agent, monkeypatch):
    from unittest.mock import patch
    a = primary_agent
    a.tools = []
    a.model, a.provider, a.api_mode = "inert", "openrouter", "chat_completions"
    a.context_compressor.context_length = 1000
    a.context_compressor.threshold_tokens = 500
    a.max_tokens = 1
    calls, sleeps = [], []
    def grow(request):
        body = json.loads(request.content)
        body["messages"] = [{"role": "user", "content": "x" * 4400}]
        request.stream = httpx.ByteStream(json.dumps(body).encode())
        if hasattr(request, "_content"):
            del request._content
        request.read()
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200)), event_hooks={"request": [grow]}) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_openai_class(OpenAI)(api_key="inert", base_url="https://inert.invalid", http_client=http, max_retries=2)
        monkeypatch.setattr(a, "_create_request_openai_client", lambda **kw: sdk)
        monkeypatch.setattr(a, "_close_request_openai_client", lambda *args, **kw: None)
        # Polling waits are not SDK backoff or outer API retry. Record sleeps
        # during the bound request, plus long outer retry delays; inert polling
        # remains immediate without pretending that no polling exists.
        monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds) if current_attempt_identity() is not None or seconds >= 1 else None)
        with patch.object(a, "_try_activate_fallback") as fallback, patch.object(a, "_compress_context") as compact, patch.object(a, "_persist_session"), patch.object(a, "_save_trajectory"), patch.object(a, "_cleanup_task_resources"):
            result = a.run_conversation("hi")
        assert result["failed"] is True
        assert fallback.call_count == 0
        assert compact.call_count == 0
        assert calls == []
        assert sleeps == []
        assert a._api_call_count == 0
        assert a.iteration_budget.used == 0
        assert current_attempt_identity() is None
        assert current_local_refusal() is None


def test_actual_streaming_gemini_family_and_cleanup(primary_agent, monkeypatch):
    from agent.chat_completion_helpers import interruptible_streaming_api_call
    a = primary_agent
    a.model, a.provider, a.api_mode = "gemini-inert", "gemini", "chat_completions"
    a.context_compressor.context_length = 100000
    seen = []
    def receive(request):
        seen.append(current_attempt_identity())
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}\n\n')
    with httpx.Client(transport=httpx.MockTransport(receive)) as http:
        wrap_httpx_client_transports(http)
        client = GeminiNativeClient(api_key="inert", base_url="https://inert.invalid", http_client=http)
        monkeypatch.setattr(a, "_create_request_openai_client", lambda **kw: client)
        monkeypatch.setattr(a, "_close_request_openai_client", lambda *args, **kw: None)
        result = interruptible_streaming_api_call(a, {"model": a.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
        assert result.choices[0].message.content == "ok"
        assert seen[0].family == "gemini_native"
        assert seen[0].model == "gemini-inert"
        assert current_attempt_identity() is None
        assert current_local_refusal() is None


def test_actual_model_provider_fallback_adoption_guards_new_physical_route(primary_agent, monkeypatch):
    from agent.chat_completion_helpers import try_activate_fallback
    a = primary_agent
    a.model, a.provider, a.base_url = "primary-model", "openai", "https://primary.invalid"
    a._fallback_chain = [{"provider": "openrouter", "model": "fallback-model", "base_url": "https://fallback.invalid", "api_key": "inert", "api_mode": "chat_completions"}]
    a._fallback_index = 0
    a._credential_pool = None
    calls, sleeps, identities = [], [], []
    def hook(request):
        identities.append(current_attempt_identity())
        request.stream = httpx.ByteStream(json.dumps({"model": "fallback-model", "messages": [{"role": "user", "content": "x" * 4400}]}).encode())
        if hasattr(request, "_content"):
            del request._content
        request.read()
    http = httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200, json={})), event_hooks={"request": [hook]})
    # Router output is auxiliary/non-covered; promotion must make it covered.
    sdk = OpenAI(api_key="inert", base_url="https://fallback.invalid", http_client=http, max_retries=2)
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *args, **kwargs: (sdk, "fallback-model"))
    monkeypatch.setattr("hermes_cli.fallback_config.resolve_entry_api_key", lambda _: "inert")
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _: None)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *args, **kw: 1000)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *args, **kw: {})
    monkeypatch.setattr("agent.chat_completion_helpers.get_provider_request_timeout", lambda *args: None)
    monkeypatch.setattr(a, "_ensure_lmstudio_runtime_loaded", lambda: None)
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(a, "_build_keepalive_http_client", lambda *args, **kw: http)
    monkeypatch.setattr("run_agent.OpenAI", intercepted_openai_class(OpenAI))
    try:
        assert try_activate_fallback(a) is True
        assert a.model == "fallback-model"
        assert a.provider == "openrouter"
        with pytest.raises(ProviderBoundRequestOverLimit):
            _dispatch_nonstreaming_api_request(a, {"model": a.model, "messages": [{"role": "user", "content": "hi"}]}, make_client=lambda *args, **kw: a.client)
        assert calls == []
        assert sleeps == []
        assert identities[0].model == "fallback-model"
        assert identities[0].endpoint == "https://fallback.invalid"
        assert identities[0].window == 1000
        assert current_attempt_identity() is None
        assert current_local_refusal() is None
    finally:
        sdk.close()
