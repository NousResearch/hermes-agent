"""Local check of the applied Gemini-on-OpenCode-Go patch (baseline checkout)."""

import json

import httpx
import pytest

from agent.agent_runtime_helpers import create_openai_client
from agent.gemini_native_adapter import GeminiNativeClient, normalize_gemini_base_url
from agent.opencode_go_gemini import (
    OPENCODE_GO_GEMINI_BASE_URL,
    is_opencode_go_gemini_route,
    with_session_header,
)
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile

GO_LEGACY = "https://opencode.ai/zen/go/v1"
MODEL = "gemini-4-argon-contributor"


@pytest.mark.parametrize("provider,model,base_url,native", [
    ("opencode-go", MODEL, GO_LEGACY, True),
    ("go", MODEL, GO_LEGACY, True),
    ("opencode-go", MODEL, OPENCODE_GO_GEMINI_BASE_URL, True),
    ("opencode-go", "deepseek-v4.1-flash", GO_LEGACY, False),
    ("opencode-go", "glm-5.3", GO_LEGACY, False),
    ("opencode-zen", MODEL, "https://opencode.ai/zen/v1", False),
    ("opencode-go", MODEL, "https://example.invalid/v1", False),
])
def test_route_matrix(provider, model, base_url, native):
    assert is_opencode_go_gemini_route(provider, model, base_url) is native


def test_base_url_moves_to_the_google_wire_root():
    assert normalize_gemini_base_url(GO_LEGACY) == OPENCODE_GO_GEMINI_BASE_URL
    assert normalize_gemini_base_url("https://opencode.ai/zen/go") == OPENCODE_GO_GEMINI_BASE_URL
    assert normalize_gemini_base_url(OPENCODE_GO_GEMINI_BASE_URL) == OPENCODE_GO_GEMINI_BASE_URL
    assert normalize_gemini_base_url("https://generativelanguage.googleapis.com/v1beta") == \
        "https://generativelanguage.googleapis.com/v1beta"


def test_session_header_is_added_without_dropping_existing_headers():
    merged = with_session_header({"X-Test": "1"}, "opencode-go", GO_LEGACY)
    assert merged["X-Test"] == "1"
    assert merged["x-opencode-session"]


def _serve(requests, stream):
    def serve(request):
        requests.append(request)
        payload = ({"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
                   if "/models/" in request.url.path else
                   {"id": "x", "object": "chat.completion", "model": MODEL,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                 "finish_reason": "stop"}]})
        if stream:
            if "/models/" not in request.url.path:
                payload["object"] = "chat.completion.chunk"
                payload["choices"][0]["delta"] = payload["choices"][0].pop("message")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, json=payload)
    return serve


@pytest.mark.parametrize("stream", [False, True])
def test_main_turn_uses_generate_content(stream):
    from types import SimpleNamespace

    requests = []
    profile = get_provider_profile("opencode-go")
    agent = SimpleNamespace(provider="opencode-go", requested_provider="opencode-go", model=MODEL,
                            base_url=GO_LEGACY, _client_log_context=lambda: "test",
                            _build_keepalive_http_client=lambda *a, **k: None)
    client_kwargs = {"api_key": "test-key", "base_url": GO_LEGACY,
                     "default_headers": {**profile.default_headers, "x-test-default": "preserved"},
                     "http_client": httpx.Client(transport=httpx.MockTransport(_serve(requests, stream)))}
    client = create_openai_client(agent, client_kwargs, reason="main", shared=False)
    try:
        assert isinstance(client, GeminiNativeClient), type(client)
        assert client.base_url == OPENCODE_GO_GEMINI_BASE_URL
        kwargs = ChatCompletionsTransport().build_kwargs(
            MODEL, [{"role": "user", "content": "Reply OK"}], provider_profile=profile,
            provider_name="opencode-go", base_url=GO_LEGACY,
            reasoning_config={"enabled": False, "effort": "none"})
        kwargs.update(stream=stream, max_tokens=64)
        result = client.chat.completions.create(**kwargs)
        if stream:
            assert "".join(c.choices[0].delta.content or "" for c in result) == "OK"
        else:
            assert result.choices[0].message.content == "OK"
    finally:
        client.close()
    request = requests[-1]
    method = "streamGenerateContent" if stream else "generateContent"
    assert request.url.path == f"/inference/go/v1/models/{MODEL}:{method}", request.url
    assert request.headers["x-goog-api-key"] == "test-key"
    assert request.headers.get("x-opencode-session"), request.headers
    assert request.headers["x-test-default"] == "preserved"


def test_sibling_model_keeps_chat_completions():
    from types import SimpleNamespace

    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "model": "deepseek-v4.1-flash",
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                                      "finish_reason": "stop"}]})

    profile = get_provider_profile("opencode-go")
    agent = SimpleNamespace(provider="opencode-go", requested_provider="opencode-go",
                            model="deepseek-v4.1-flash", base_url=GO_LEGACY, _client_log_context=lambda: "test",
                            _build_keepalive_http_client=lambda *a, **k: None)
    client = create_openai_client(agent, {"api_key": "test-key", "base_url": GO_LEGACY,
                                          "http_client": httpx.Client(transport=httpx.MockTransport(serve))},
                                  reason="main", shared=False)
    try:
        assert not isinstance(client, GeminiNativeClient)
        kwargs = ChatCompletionsTransport().build_kwargs(
            "deepseek-v4.1-flash", [{"role": "user", "content": "Reply OK"}], provider_profile=profile,
            provider_name="opencode-go", base_url=GO_LEGACY,
            reasoning_config={"enabled": False, "effort": "none"})
        kwargs.update(max_tokens=16)
        client.chat.completions.create(**kwargs)
    finally:
        client.close()
    assert requests[-1].url.path.endswith("/chat/completions")


def test_reasoning_projects_to_thinking_config_on_the_go_route():
    profile = get_provider_profile("opencode-go")
    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}, model=MODEL, base_url=GO_LEGACY)
    assert extra_body.get("thinking_config")
    assert not top_level
    sibling_extra, sibling_top = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2", base_url=GO_LEGACY)
    assert not sibling_extra and sibling_top.get("reasoning_effort") in {"high", "max"}
