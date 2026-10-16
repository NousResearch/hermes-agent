"""Regression for #139526: Go Gemini uses the native relay wire."""

import json

import httpx
import pytest

from agent.auxiliary_client import resolve_provider_client


@pytest.mark.parametrize("entry", ["aux", "main"])
def test_go_gemini_request_uses_native_wire(monkeypatch, entry):
    requests = []

    def respond(_transport, request):
        requests.append(request)
        if request.url.path.endswith(":generateContent"):
            return httpx.Response(200, json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]},
                                "finishReason": "STOP"}],
            })
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 0,
            "model": "gemini-4-argon-contributor",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                         "finish_reason": "stop"}],
        })

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    agent = None
    model = "gemini-4-argon-contributor"
    if entry == "aux":
        client, model = resolve_provider_client(
            "opencode-go", model=model, explicit_api_key="fixture-go-key",
            explicit_base_url="https://opencode.ai/zen/go/v1",
        )
    else:
        from run_agent import AIAgent
        agent = AIAgent(
            provider="opencode-go", model=model, api_key="fixture-go-key",
            base_url="https://opencode.ai/zen/go/v1", skip_memory=True,
            skip_context_files=True, quiet_mode=True, enabled_toolsets=[],
        )
        client = agent._create_request_openai_client(reason="go-regression")
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply OK"}],
            extra_headers={"x-opencode-session": "fixture-session"},
        )
    finally:
        client.close()
        if agent is not None:
            agent._close_cached_request_openai_client(reason="test_cleanup")
            agent.client.close()
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == (
        "https://opencode.ai/inference/go/v1/models/"
        "gemini-4-argon-contributor:generateContent"
    )
    assert request.headers["x-goog-api-key"] == "fixture-go-key"
    assert request.headers["x-opencode-session"] == "fixture-session"
    assert "contents" in json.loads(request.content)


@pytest.mark.parametrize("stream", [False, True])
def test_go_native_preserves_controls_and_request_headers(monkeypatch, stream):
    from agent.opencode_gemini import make_go_gemini_client
    seen = []
    key = ["first-key"]

    def respond(_transport, request):
        seen.append(request)
        payload = {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]},
                                  "finishReason": "STOP"}]}
        if stream:
            return httpx.Response(200, content="data: " + json.dumps(payload) + "\n\n")
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    with make_go_gemini_client(
        "https://opencode.ai/inference/go/v1?fixture=preserved", api_key=lambda: key[0],
        default_headers={"x-opencode-session": "default-session"},
    ) as client:
        for value in ["first-key", "second-key"]:
            key[0] = value
            result = client.chat.completions.create(
                model="gemini-4-argon-contributor", messages=[{"role": "user", "content": "OK"}],
                stream=stream, max_tokens=123, extra_body={"reasoning": {"enabled": False}},
                extra_headers={"x-opencode-session": "request-session"},
            )
            if stream:
                list(result)
        assert client.default_headers["x-opencode-session"] == "default-session"
    assert [r.headers["x-goog-api-key"] for r in seen] == ["first-key", "second-key"]
    for request in seen:
        assert request.url.params["fixture"] == "preserved"
        assert request.headers["x-opencode-session"] == "request-session"
        config = json.loads(request.content)["generationConfig"]
        assert config["maxOutputTokens"] == 123
        assert config["thinkingConfig"]["includeThoughts"] is False


@pytest.mark.parametrize("provider,model,base", [
    ("opencode-go", "glm-fixture", "https://opencode.ai/zen/go/v1"),
    ("opencode-zen", "gemini-fixture", "https://opencode.ai/zen/v1"),
    ("opencode-go", "gemini-fixture", "https://proxy.example/v1"),
    ("custom:go", "gemini-fixture", "https://opencode.ai/zen/go/v1"),
])
def test_unrelated_or_explicit_custom_routes_are_unchanged(provider, model, base):
    from agent.opencode_gemini import go_gemini_base
    assert go_gemini_base(provider, model, base) is None


def test_go_aux_forwards_caller_output_limit():
    from agent.auxiliary_client import _build_call_kwargs
    kwargs = _build_call_kwargs(
        "opencode-go", "gemini-4-argon-contributor", [{"role": "user", "content": "OK"}],
        max_tokens=123, base_url="https://opencode.ai/zen/go/v1",
    )
    assert kwargs["max_tokens"] == 123


def test_main_model_switch_does_not_reuse_wrong_wire(monkeypatch):
    from run_agent import AIAgent
    from agent.gemini_native_adapter import GeminiNativeClient
    agent = AIAgent(
        provider="opencode-go", model="gemini-4-argon-contributor", api_key="fixture",
        base_url="https://opencode.ai/zen/go/v1", skip_memory=True,
        skip_context_files=True, quiet_mode=True, enabled_toolsets=[],
    )
    try:
        for model, native in [("gemini-4-argon-contributor", True), ("glm-fixture", False),
                              ("gemini-4-argon-contributor", True)]:
            agent.model = model
            client = agent._create_request_openai_client(reason="switch_fixture")
            assert isinstance(client, GeminiNativeClient) is native
            agent._close_request_openai_client(client, reason="request_complete")
    finally:
        agent._close_cached_request_openai_client(reason="test_cleanup")
        agent.client.close()


@pytest.mark.asyncio
async def test_async_aux_reaches_native_wire(monkeypatch):
    seen = []
    def respond(_transport, request):
        seen.append(request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    client, model = resolve_provider_client(
        "opencode-go", model="gemini-4-argon-contributor", async_mode=True,
        explicit_api_key="fixture", explicit_base_url="https://opencode.ai/zen/go/v1",
    )
    try:
        result = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "OK"}],
            extra_headers={"x-opencode-session": "async-fixture"},
        )
        assert result.choices[0].message.content == "OK"
    finally:
        client._sync.close()
    assert seen[0].url.path.endswith(":generateContent")
    assert seen[0].headers["x-opencode-session"] == "async-fixture"
