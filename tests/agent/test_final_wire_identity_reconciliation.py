"""R2 immutable attempt identity must match current final body/URL route.

All model names/windows are synthetic; no live capacity lookup or assertion.
"""
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from anthropic import Anthropic, AnthropicBedrock

import agent.final_wire_admission as admission
from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent.gemini_native_adapter import GeminiNativeClient
from tests.agent.test_final_wire_review_regressions import identity
from tests.agent.test_final_wire_current_stream import Pool, body, replace_stream
from tests.agent.test_final_wire_native_egress import converse_client


@pytest.mark.parametrize("stage", ["sdk_extra", "middleware"])
@pytest.mark.parametrize("model", ["inert-original-model", "inert-smaller-model", "inert-unknown-model"])
def test_final_model_never_inherits_another_models_window(monkeypatch, stage, model):
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    sleeps, seen = [], []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    def hook(request):
        final = json.loads(request.content)
        if stage == "middleware":
            final["model"] = model
            replace_stream(request, final)
        seen.append((final, admission.current_attempt_identity()))
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=identity().endpoint,
                                                        http_client=http, max_retries=2)
        a = SimpleNamespace(model=identity().model, provider="openai", api_mode="chat_completions",
                            base_url=identity().endpoint, session_id="identity",
                            context_compressor=SimpleNamespace(context_length=10000))
        kwargs = body("hi")
        if stage == "sdk_extra":
            kwargs["extra_body"] = {"model": model}
        if model != a.model:
            with pytest.raises(admission.ProviderBoundInvalidAccounting):
                _dispatch_nonstreaming_api_request(a, kwargs, make_client=lambda *a, **kw: sdk)
        else:
            _dispatch_nonstreaming_api_request(a, kwargs, make_client=lambda *a, **kw: sdk)
    assert len(seen) == 1
    assert seen[0][0]["model"] == model
    assert seen[0][1].window == 10000
    assert pool.sent == ([seen[0][0]] if model == identity().model else [])
    assert sleeps == []
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.parametrize("url", ["https://other.invalid/v1/chat/completions",
                                 "https://inert.invalid/other/chat/completions",
                                 "https://inert.invalid/v1/responses",
                                 "http://inert.invalid/v1/chat/completions"])
def test_post_sdk_endpoint_or_family_change_refuses(monkeypatch, url):
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    def hook(request):
        request.url = httpx.URL(url)
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=identity().endpoint,
                                                        http_client=http, max_retries=2)
        with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundRequestOverLimit):
            sdk.chat.completions.create(**body("hi"))
    assert pool.sent == []
    assert sleeps == []


@pytest.mark.parametrize("family", ["anthropic_messages", "anthropic_bedrock", "gemini_native", "bedrock_converse"])
@pytest.mark.parametrize("changed", [False, True])
def test_real_native_sdk_url_model_reconciliation(monkeypatch, family, changed):
    model = "gemini-inert" if family == "gemini_native" else "claude-sonnet-4-5"
    ident = replace(identity(), family=family, model=model, endpoint="https://inert.invalid")
    sent, sleeps, urls = [], [], []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    if family == "bedrock_converse":
        client, delegate = converse_client()
        def mutate(request, **kw):
            if changed:
                request.url = request.url.replace(model, "inert-unknown")
            urls.append(request.url)
        client.meta.events.register_last("before-send.bedrock-runtime", mutate)
        call = lambda: client.converse(modelId=model, messages=[{"role": "user", "content": [{"text": "hi"}]}], inferenceConfig={"maxTokens": 1})
        sent = delegate.seen
    else:
        def hook(request):
            if changed:
                if family == "anthropic_messages":
                    final = json.loads(request.content)
                    final["model"] = "inert-unknown"
                    replace_stream(request, final)
                else:
                    request.url = httpx.URL(str(request.url).replace(model, "inert-unknown"))
            urls.append(str(request.url))
        def receive(request):
            sent.append((json.loads(request.content), admission.current_attempt_identity()))
            if family == "gemini_native":
                payload = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
            else:
                payload = {"id": "inert", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
            return httpx.Response(200, json=payload)
        http = httpx.Client(transport=httpx.MockTransport(receive), event_hooks={"request": [hook]})
        admission.wrap_httpx_client_transports(http)
        if family == "gemini_native":
            client = GeminiNativeClient(api_key="inert", base_url=ident.endpoint, http_client=http)
            call = lambda: client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
        else:
            cls = admission.intercepted_anthropic_class(AnthropicBedrock if family == "anthropic_bedrock" else Anthropic)
            auth = {"aws_access_key": "inert", "aws_secret_key": "inert", "aws_region": "us-east-1"} if family == "anthropic_bedrock" else {"api_key": "inert"}
            client = cls(**auth, base_url=ident.endpoint, http_client=http, max_retries=2)
            call = lambda: client.messages.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
    try:
        with admission.bind_attempt_identity(ident):
            if changed:
                with pytest.raises(admission.ProviderBoundInvalidAccounting):
                    call()
            else:
                call()
        assert len(urls) == 1
        assert len(sent) == (0 if changed else 1)
        assert sleeps == []
    finally:
        client.close()
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.parametrize("endpoint", ["https://inert.invalid/v1", "https://inert.invalid/v1/", "https://inert.invalid/v1/chat/completions"])
def test_base_or_exact_operation_endpoint_control(endpoint):
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    with httpx.Client(transport=transport) as http:
        admission.wrap_httpx_client_transports(http)
        with admission.bind_attempt_identity(replace(identity(), endpoint=endpoint)):
            http.post("https://inert.invalid/v1/chat/completions", json=body("hi"))
    assert pool.sent == [body("hi")]


@pytest.mark.parametrize("prefix", ["google/", "gemini/"])
def test_native_gemini_approved_prefix_and_base_normalization(prefix):
    sent = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(200, json={"candidates": []}))) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = GeminiNativeClient(api_key="inert", base_url="https://inert.invalid/v1beta/openai", http_client=http)
        ident = replace(identity(), family="gemini_native", model=prefix + "gemini-inert", endpoint="https://inert.invalid/v1beta/openai")
        with admission.bind_attempt_identity(ident):
            sdk.chat.completions.create(model=ident.model, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
    assert len(sent) == 1
    assert sent[0].url.path == "/v1beta/models/gemini-inert:generateContent"


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4.5", "claude-sonnet-4.5"])
def test_existing_anthropic_model_normalization_preserves_identity(model):
    from agent.anthropic_adapter import build_anthropic_kwargs
    sent = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200, json={"id": "inert", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5", "content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}))) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_anthropic_class(Anthropic)(api_key="inert", base_url="https://inert.invalid", http_client=http, max_retries=2)
        ident = replace(identity(), family="anthropic_messages", model=model, endpoint="https://inert.invalid")
        with admission.bind_attempt_identity(ident):
            sdk.messages.create(**build_anthropic_kwargs(model, [{"role": "user", "content": "hi"}], [], 1, None))
    assert len(sent) == 1


@pytest.mark.parametrize("model", [None, "inert-unknown", "gemini-inert"])
def test_native_body_model_must_agree_with_url_when_present(model):
    sent = []
    ident = replace(identity(), family="gemini_native", model="gemini-inert", endpoint="https://inert.invalid")
    final = {"model": model, "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}}
    with httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200))) as http:
        admission.wrap_httpx_client_transports(http)
        with admission.bind_attempt_identity(ident):
            # Gemini's schema omits model entirely. A conflicting emitted
            # model is invalid identity; even a matching/null extra model is
            # unsupported native context rather than silently certified.
            expected = admission.ProviderBoundInvalidAccounting if model == "inert-unknown" else admission.ProviderBoundUnsupportedAccounting
            with pytest.raises(expected):
                http.post(ident.endpoint + "/models/gemini-inert:generateContent", json=final)
    assert sent == []


@pytest.mark.parametrize("window", [1000, 2000])
def test_explicitly_resolved_final_model_uses_its_own_window(window):
    seen = []
    final_model = "inert-smaller-model"
    ident = replace(identity(), model=final_model, window=window)
    a = SimpleNamespace(model=final_model, provider="openai", api_mode="chat_completions", base_url=ident.endpoint,
                        context_compressor=SimpleNamespace(context_length=window))
    with httpx.Client(transport=httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200, json={"id": "inert", "object": "chat.completion", "choices": []}))) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=ident.endpoint, http_client=http, max_retries=2)
        kwargs = body("hi")
        kwargs["extra_body"] = {"model": final_model, "messages": [{"role": "user", "content": "x" * 4400}]}
        if window == 1000:
            with pytest.raises(admission.ProviderBoundRequestOverLimit):
                _dispatch_nonstreaming_api_request(a, kwargs, make_client=lambda *a, **kw: sdk)
        else:
            _dispatch_nonstreaming_api_request(a, kwargs, make_client=lambda *a, **kw: sdk)
    assert len(seen) == (1 if window == 2000 else 0)
    assert admission.current_attempt_identity() is None
