"""Selected HTTPX routes after auth/hooks/redirects and SDK/factory formation."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI
from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, ProviderBoundUnsupportedAccounting,
    bind_attempt_identity, intercepted_openai_class, wrap_httpx_client_transports,
)
from agent.conversation_compression import ProviderBoundRequestOverLimit
from agent.agent_runtime_helpers import create_openai_client
from agent.process_bootstrap import build_keepalive_http_client


def identity():
    return FinalAttemptIdentity(COVERED_MAIN, "chat_completions", "inert", "https://inert.invalid", 1000, "routes")


class CountingTransport(httpx.BaseTransport):
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.kwargs = kwargs
    def handle_request(self, request):
        self.calls.append(request)
        return httpx.Response(200, json={})


@pytest.mark.parametrize("route", ["default", "http_mount", "https_mount", "specific_mount", "proxy", "factory_plain", "factory_proxy"])
def test_actual_selected_default_mount_proxy_factory_route_refuses_before_delegate(monkeypatch, route):
    monkeypatch.setattr(httpx, "HTTPTransport", CountingTransport)
    monkeypatch.setattr("httpx._client.HTTPTransport", CountingTransport)
    if route.startswith("factory"):
        monkeypatch.setattr("agent.process_bootstrap._get_proxy_for_base_url", lambda _: "http://proxy.invalid" if route == "factory_proxy" else None)
        client = build_keepalive_http_client("https://inert.invalid")
    elif route == "default":
        client = httpx.Client(trust_env=False)
    elif route == "proxy":
        client = httpx.Client(proxy="http://proxy.invalid", trust_env=False)
    else:
        mount = {"http_mount": "http://", "https_mount": "https://", "specific_mount": "all://inert.invalid"}[route]
        client = httpx.Client(mounts={mount: CountingTransport()}, trust_env=False)
    url = "http://inert.invalid" if route == "http_mount" else "https://inert.invalid"
    assert isinstance(client, httpx.Client)
    selected = client._transport_for_url(httpx.URL(url))
    assert isinstance(selected, CountingTransport)
    wrap_httpx_client_transports(client)
    try:
        with bind_attempt_identity(identity()), pytest.raises(ProviderBoundRequestOverLimit):
            client.post(url, json={"messages": [{"role": "user", "content": "x" * 4400}]})
        assert selected.calls == []
    finally:
        client.close()


@pytest.mark.parametrize("stage", ["auth", "hook"])
def test_actual_post_sdk_auth_hook_growth_refuses_without_sdk_sleep(stage, monkeypatch):
    calls, sleeps = [], []
    def grow(request):
        body = json.loads(request.content)
        body["messages"] = [{"role": "user", "content": "x" * 4400}]
        request.stream = httpx.ByteStream(json.dumps(body).encode())
        if hasattr(request, "_content"):
            del request._content
        request.read()
    class GrowingAuth(httpx.Auth):
        def auth_flow(self, request):
            grow(request)
            yield request
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200)), auth=GrowingAuth() if stage == "auth" else None, event_hooks={"request": [grow]} if stage == "hook" else None) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_openai_class(OpenAI)(api_key="inert", base_url="https://inert.invalid", http_client=http, max_retries=2)
        with bind_attempt_identity(identity()), pytest.raises(ProviderBoundRequestOverLimit):
            sdk.chat.completions.create(model="inert", messages=[{"role": "user", "content": "hi"}])
    assert calls == []
    assert sleeps == []


def test_actual_307_redirect_second_request_hook_growth_refuses():
    calls = []
    def receive(request):
        calls.append(request.url.path)
        return httpx.Response(307, headers={"location": "/second"})
    def hook(request):
        if request.url.path == "/second":
            request.stream = httpx.ByteStream(json.dumps({"messages": [{"role": "user", "content": "x" * 4400}]}).encode())
            if hasattr(request, "_content"):
                del request._content
            request.read()
    with httpx.Client(transport=httpx.MockTransport(receive), follow_redirects=True, event_hooks={"request": [hook]}) as client:
        wrap_httpx_client_transports(client)
        with bind_attempt_identity(identity()), pytest.raises(ProviderBoundRequestOverLimit):
            client.post("https://inert.invalid/first", json={"messages": [{"role": "user", "content": "hi"}]})
    assert calls == ["/first"]


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("kwargs", [{}, {"http_client": None}])
def test_actual_primary_factory_none_never_silently_sdk_fallback(provider, kwargs):
    a = SimpleNamespace(provider=provider, _build_keepalive_http_client=lambda *a, **kw: None, _client_log_context=lambda: "inert")
    with patch("run_agent.OpenAI") as sdk, pytest.raises(ProviderBoundUnsupportedAccounting):
        create_openai_client(a, {"api_key": "inert", "base_url": "https://generativelanguage.googleapis.com" if provider == "gemini" else "https://inert.invalid", **kwargs}, reason="test", shared=True)
    assert sdk.call_count == 0


def test_auth_challenge_resend_remeasured_after_one_prior_allowed_delegate():
    calls = []
    class ChallengeAuth(httpx.Auth):
        def auth_flow(self, request):
            response = yield request
            if response.status_code == 401:
                body = json.loads(request.content)
                body["messages"] = [{"role": "user", "content": "x" * 4400}]
                request.stream = httpx.ByteStream(json.dumps(body).encode())
                if hasattr(request, "_content"):
                    del request._content
                request.read()
                yield request
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(401)), auth=ChallengeAuth()) as client:
        wrap_httpx_client_transports(client)
        with bind_attempt_identity(identity()):
            with pytest.raises(ProviderBoundRequestOverLimit):
                client.post("https://inert.invalid/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
    # Refusal is for the resend; never claim zero across a prior allowed send.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_actual_async_client_selected_transport_guard():
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200))) as client:
        wrap_httpx_client_transports(client)
        with bind_attempt_identity(identity()):
            await client.post("https://inert.invalid/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
            with pytest.raises(ProviderBoundRequestOverLimit):
                await client.post("https://inert.invalid/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 4400}], "max_tokens": 1})
    assert len(calls) == 1


def test_actual_gemini_primary_factory_supplied_http_client_is_guarded():
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200, json={}))) as http:
        a = SimpleNamespace(provider="gemini", _client_log_context=lambda: "inert")
        client = create_openai_client(a, {"api_key": "inert", "base_url": "https://generativelanguage.googleapis.com", "http_client": http}, reason="test", shared=True)
        native = FinalAttemptIdentity(COVERED_MAIN, "gemini_native", "gemini-inert", "https://generativelanguage.googleapis.com", 1000, "provided-client")
        with bind_attempt_identity(native), pytest.raises(ProviderBoundRequestOverLimit):
            client.chat.completions.create(model="gemini-inert", messages=[{"role": "user", "content": "x" * 4400}], max_tokens=1)
    assert calls == []
