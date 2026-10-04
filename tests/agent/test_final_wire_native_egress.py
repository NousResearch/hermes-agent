"""Actual native builders/SDKs and botocore endpoint, inert delegates only."""
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from anthropic import Anthropic, AnthropicBedrock
from botocore.config import Config
from botocore.awsrequest import AWSResponse
from botocore.session import Session

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.bedrock_adapter import build_converse_kwargs
from agent.conversation_compression import ProviderBoundRequestOverLimit
from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, bind_attempt_identity, intercepted_anthropic_class,
    wrap_httpx_client_transports, wrap_botocore_runtime_client, current_attempt_identity,
)
from agent.gemini_native_adapter import GeminiNativeClient


def ident(family, window=1000):
    return FinalAttemptIdentity(COVERED_MAIN, family, "claude-sonnet-4-5", "https://inert.invalid", window, "native")


def tools(size):
    return [{"type": "function", "function": {"name": "inert", "description": "inert",
        "parameters": {"type": "object", "properties": {"value": {"type": "string", "description": "x" * size}}}}}]


@pytest.mark.parametrize("family", ["anthropic_messages", "anthropic_bedrock", "gemini_native"])
@pytest.mark.parametrize("size", [10, 4400])
def test_actual_native_conversion_schema_egress(family, size, monkeypatch):
    seen, sleeps = [], []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    def receive(request):
        seen.append(json.loads(request.content))
        if family == "gemini_native":
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})
        return httpx.Response(200, json={"id": "inert", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5", "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
    http = httpx.Client(transport=httpx.MockTransport(receive))
    wrap_httpx_client_transports(http)
    rows = [{"role": "system", "content": "inert system"}, {"role": "user", "content": "hi"}]
    if family == "gemini_native":
        sdk = GeminiNativeClient(api_key="inert", base_url="https://inert.invalid", http_client=http)
        call = lambda: sdk.chat.completions.create(model="gemini-inert", messages=rows, tools=tools(size), max_tokens=1)
    else:
        cls = intercepted_anthropic_class(AnthropicBedrock if family == "anthropic_bedrock" else Anthropic)
        if family == "anthropic_bedrock":
            sdk = cls(aws_access_key="inert", aws_secret_key="inert", aws_region="us-east-1", base_url="https://inert.invalid", http_client=http, max_retries=2)
        else:
            sdk = cls(api_key="inert", base_url="https://inert.invalid", http_client=http, max_retries=2)
        kwargs = build_anthropic_kwargs("claude-sonnet-4-5", rows, tools(size), 1, None)
        call = lambda: sdk.messages.create(**kwargs)
    try:
        with bind_attempt_identity(ident(family)):
            if size > 1000:
                with pytest.raises(ProviderBoundRequestOverLimit):
                    call()
            else:
                call()
        assert len(seen) == (0 if size > 1000 else 1)
        assert sleeps == []
        if seen:
            body = seen[0]
            assert ("systemInstruction" if family == "gemini_native" else "system") in body
            assert "tools" in body
            if family == "anthropic_bedrock":
                assert "model" not in body
                assert "anthropic_version" in body
    finally:
        sdk.close()


class RawResponse:
    def __init__(self, body):
        self.body = body
    def stream(self, amt=None, decode_content=False):
        yield self.body


class InertSession:
    def __init__(self, statuses=(200,)):
        self.statuses = list(statuses)
        self.seen = []
    def send(self, request):
        self.seen.append((json.loads(request.body), current_attempt_identity()))
        status = self.statuses.pop(0)
        body = {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}
        if status != 200:
            body = {"message": "inert temporary failure"}
        return AWSResponse(request.url, status, {"content-type": "application/json"}, RawResponse(json.dumps(body).encode()))
    def close(self):
        pass


def converse_client(statuses=(200,)):
    client = Session().create_client("bedrock-runtime", region_name="us-east-1", endpoint_url="https://inert.invalid", aws_access_key_id="inert", aws_secret_access_key="inert", config=Config(retries={"mode": "standard", "max_attempts": 2}))
    delegate = InertSession(statuses)
    client._endpoint.http_session = delegate
    wrap_botocore_runtime_client(client)
    return client, delegate


def test_actual_native_default_and_post_thinking_caps_are_final_reservations(monkeypatch):
    from agent.final_wire_admission import project_final_body
    bodies = []
    def receive(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    with httpx.Client(transport=httpx.MockTransport(receive)) as http:
        wrap_httpx_client_transports(http)
        sdk = GeminiNativeClient(api_key="inert", base_url="https://inert.invalid", http_client=http)
        with bind_attempt_identity(ident("gemini_native", window=100000)):
            sdk.chat.completions.create(model="gemini-inert", messages=[{"role": "user", "content": "hi"}])
        assert bodies[0]["generationConfig"]["maxOutputTokens"] == 65535
        assert project_final_body(bodies[0], ident("gemini_native", window=100000)).resolved_r == 65535
        with bind_attempt_identity(ident("gemini_native")):
            with pytest.raises(ProviderBoundRequestOverLimit):
                sdk.chat.completions.create(model="gemini-inert", messages=[{"role": "user", "content": "hi"}])
        assert len(bodies) == 1
    kwargs = build_anthropic_kwargs("claude-sonnet-4-5", [{"role": "user", "content": "hi"}], [], 1, {"enabled": True, "effort": "high"})
    assert kwargs["max_tokens"] > 1
    assert "thinking" in kwargs
    seen = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_anthropic_class(Anthropic)(api_key="inert", base_url="https://inert.invalid", http_client=http, max_retries=2)
        with bind_attempt_identity(ident("anthropic_messages", window=kwargs["max_tokens"])):
            with pytest.raises(ProviderBoundRequestOverLimit):
                sdk.messages.create(**kwargs)
    assert seen == []
    assert project_final_body(kwargs, ident("anthropic_messages", window=100000)).resolved_r == kwargs["max_tokens"]


def test_actual_anthropic_cache_control_keeps_transmitted_context_supported():
    from agent.prompt_caching import apply_anthropic_cache_control
    seen = []
    def receive(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "inert", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5", "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
    with httpx.Client(transport=httpx.MockTransport(receive)) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_anthropic_class(Anthropic)(api_key="inert", base_url="https://inert.invalid", http_client=http)
        kwargs = build_anthropic_kwargs("claude-sonnet-4-5", [{"role": "user", "content": "hi"}], [], 1, None)
        kwargs["messages"] = apply_anthropic_cache_control(kwargs["messages"], native_anthropic=True)
        assert "cache_control" in str(kwargs)
        with bind_attempt_identity(ident("anthropic_messages")):
            sdk.messages.create(**kwargs)
    assert len(seen) == 1


def test_actual_converse_cache_points_keep_system_and_history_supported(monkeypatch):
    client, delegate = converse_client()
    monkeypatch.setattr("agent.bedrock_adapter._model_supports_prompt_cache", lambda _: True)
    kwargs = build_converse_kwargs("inert", [{"role": "system", "content": "system"}, {"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "last"}], tools=[], max_tokens=1)
    assert "cachePoint" in str(kwargs)
    try:
        with bind_attempt_identity(ident("bedrock_converse")):
            client.converse(**kwargs)
        assert len(delegate.seen) == 1
    finally:
        client.close()


@pytest.mark.parametrize("growth", [False, True])
def test_real_botocore_endpoint_after_last_before_send_event(growth, monkeypatch):
    client, delegate = converse_client()
    events, sleeps = [], []
    monkeypatch.setattr("botocore.endpoint.time.sleep", lambda delay: sleeps.append(delay))
    def mutate(request, **kw):
        events.append("before-send")
        if growth:
            body = json.loads(request.body)
            body["system"] = [{"text": "x" * 4400}]
            request.body = json.dumps(body).encode()
    client.meta.events.register_last("before-send.bedrock-runtime.Converse", mutate)
    kwargs = build_converse_kwargs("anthropic.claude-sonnet-4-5-20250929-v1:0", [{"role": "user", "content": "hi"}], tools=tools(10), max_tokens=1)
    try:
        with bind_attempt_identity(ident("bedrock_converse")):
            if growth:
                with pytest.raises(ProviderBoundRequestOverLimit):
                    client.converse(**kwargs)
            else:
                client.converse(**kwargs)
        assert events == ["before-send"]
        assert len(delegate.seen) == (0 if growth else 1)
        assert sleeps == []
    finally:
        client.close()


def test_real_botocore_remote_retry_preserved(monkeypatch):
    client, delegate = converse_client((500, 200))
    sleeps = []
    monkeypatch.setattr("botocore.endpoint.time.sleep", lambda delay: sleeps.append(delay))
    try:
        with bind_attempt_identity(ident("bedrock_converse")):
            client.converse(**build_converse_kwargs("inert", [{"role": "user", "content": "hi"}], max_tokens=1))
        assert len(delegate.seen) == 2
        assert len(sleeps) == 1
        assert all(i.family == "bedrock_converse" for _, i in delegate.seen)
        assert current_attempt_identity() is None
    finally:
        client.close()
