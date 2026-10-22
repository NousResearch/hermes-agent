"""Zen Gemini uses Google's wire without changing the relay's other model families.

Regression for the native-route gap identified in closed PR #36792.
"""

import asyncio
import json
from types import SimpleNamespace

from agent import auxiliary_client as aux

import httpx
import pytest

from agent.agent_runtime_helpers import create_openai_client
from agent.opencode_affinity import merge_session_affinity_headers
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.mark.parametrize(
    "provider, model, base_url, native",
    [
        ("opencode-zen", "gemini-4-argon", "https://opencode.ai/zen/v1", True),
        ("zen", "gemini-3.8-flash", "https://opencode.ai/zen/v1/", True),
        ("opencode-zen", "gemini-4-argon", "https://opencode.ai/zen", True),
        ("opencode-zen", "gemini-4-argon", "https://opencode.ai/inference/v1", True),
        ("opencode-zen", "glm-5.3", "https://opencode.ai/zen/v1", False),
        ("opencode-go", "gemini-4-argon", "https://opencode.ai/zen/go/v1", False),
        ("opencode-zen", "gemini-4-argon", "https://example.invalid/v1", False),
        ("opencode-zen", "gemini-4-argon", "https://opencode.ai/zen/go/v1", False),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_main_rebuild_preserves_the_models_wire_contract(provider, model, base_url, native, stream):
    requests = []

    def serve(request):
        requests.append(request)
        if "/models/" in request.url.path:
            payload = {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
        else:
            payload = {"id": "chat-test", "object": "chat.completion", "model": model,
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                    "finish_reason": "stop"}]}
        if stream:
            if "/models/" not in request.url.path:
                payload["object"] = "chat.completion.chunk"
                payload["choices"][0]["delta"] = payload["choices"][0].pop("message")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, json=payload)

    profile = get_provider_profile(provider)
    agent = SimpleNamespace(provider=provider, model=model, _client_log_context=lambda: "test",
                            _build_keepalive_http_client=lambda *a, **k: None)
    client_kwargs = {"api_key": "test-key", "base_url": base_url,
                     "default_headers": {**profile.default_headers, "x-test-default": "preserved"},
                     "http_client": httpx.Client(transport=httpx.MockTransport(serve))}
    client = create_openai_client(agent, client_kwargs, reason="fallback_restore", shared=False)
    try:
        kwargs = ChatCompletionsTransport().build_kwargs(
            model, [{"role": "user", "content": "Reply OK"}], provider_profile=profile,
            provider_name=provider, base_url=base_url, reasoning_config={"enabled": False, "effort": "none"},
        )
        kwargs.update(stream=stream, max_tokens=64)
        merge_session_affinity_headers(kwargs, provider, base_url, "zen-gemini-session")
        result = client.chat.completions.create(**kwargs)
        if stream:
            chunks = list(result)
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "OK"
        else:
            assert result.choices[0].message.content == "OK"
        request = requests[0]
        body = json.loads(request.content)
        if native:
            method = "streamGenerateContent" if stream else "generateContent"
            assert request.url.path.endswith(f"/v1/models/{model}:{method}"), request.url
            assert request.headers["x-goog-api-key"] == "test-key"
            assert "authorization" not in request.headers
            assert body["generationConfig"]["thinkingConfig"] == {"includeThoughts": False, "thinkingLevel": "low"}
            assert body["generationConfig"]["maxOutputTokens"] == 64
        else:
            assert request.url.path.endswith("/chat/completions")
            assert request.headers["authorization"] == "Bearer test-key"
            assert "x-goog-api-key" not in request.headers
            assert body["model"] == model
        assert request.headers["x-opencode-session"] == "zen-gemini-session"
        assert request.headers["x-test-default"] == "preserved"
        assert request.headers["user-agent"].startswith("HermesAgent/")
        assert client_kwargs["base_url"] == base_url
    finally:
        client.close()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "provider, native",
    [("opencode-zen", True), ("zen", True), ("custom:zen-bridge", True),
     ("custom:zen-pinned", False), ("opencode-go", False)],
)
def test_auxiliary_resolution_preserves_native_wire_and_small_budget(monkeypatch, provider, native, async_mode, stream):
    from hermes_constants import get_hermes_home

    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  default_headers:\n    x-test-default: preserved\n"
        "custom_providers:\n"
        "  - name: zen-bridge\n    base_url: https://opencode.ai/zen/v1\n    api_key: test-key\n"
        "  - name: zen-pinned\n    base_url: https://opencode.ai/zen/v1\n    api_key: test-key\n"
        "    api_mode: chat_completions\n",
        encoding="utf-8",
    )
    requests = []

    def send(_client, request, **_kwargs):
        requests.append(request)
        payload = (
            {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
            if "/models/" in request.url.path else
            {"id": "chat-test", "object": "chat.completion", "model": "gemini-4-argon",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]}
        )
        if stream:
            if "/models/" not in request.url.path:
                payload["object"] = "chat.completion.chunk"
                payload["choices"][0]["delta"] = payload["choices"][0].pop("message")
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, request=request, json=payload)

    async def async_send(client, request, **kwargs):
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    client, model = aux.resolve_provider_client(provider, model="gemini-4-argon", async_mode=async_mode,
                                               api_mode=None if provider.startswith("custom:") else "codex_responses")
    assert client is not None and model is not None
    token = aux.set_runtime_main("opencode-zen", model, base_url=str(client.base_url), session_id="aux-zen-session")
    kwargs = aux._build_call_kwargs(provider, model, [{"role": "user", "content": "Reply OK"}],
                                   base_url=str(client.base_url), max_tokens=32, task="title_generation",
                                   reasoning_config={"enabled": False, "effort": "none"})
    kwargs["stream"] = stream

    async def run_async():
        try:
            result = await client.chat.completions.create(**kwargs)
            if stream:
                return "".join([chunk.choices[0].delta.content or "" async for chunk in result])
            return result.choices[0].message.content
        finally:
            await client.close()

    try:
        if async_mode:
            text = asyncio.run(run_async())
        else:
            result = client.chat.completions.create(**kwargs)
            text = ("".join(chunk.choices[0].delta.content or "" for chunk in result)
                    if stream else result.choices[0].message.content)
        assert text == "OK"
        request = requests[0]
        body = json.loads(request.content)
        if native:
            method = "streamGenerateContent" if stream else "generateContent"
            assert request.url.path == f"/zen/v1/models/gemini-4-argon:{method}", request.url
            assert request.headers["x-goog-api-key"] == "test-key"
            assert body["generationConfig"]["maxOutputTokens"] == 32
            assert body["generationConfig"]["thinkingConfig"]["includeThoughts"] is False
        else:
            assert request.url.path.endswith("/chat/completions"), request.url
            assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["x-opencode-session"] == "aux-zen-session"
        assert request.headers["x-test-default"] == "preserved"
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)
        if not async_mode:
            client.close()


def test_same_endpoint_model_switch_changes_the_cached_client_protocol():
    from agent.gemini_native_adapter import GeminiNativeClient
    from openai import OpenAI
    from run_agent import AIAgent

    agent = AIAgent(api_key="test-key", base_url="https://opencode.ai/zen/v1", provider="opencode-zen",
                    model="glm-5.3", quiet_mode=True, skip_context_files=True, skip_memory=True)
    try:
        first = agent._create_request_openai_client(reason="test")
        assert isinstance(first, OpenAI)
        agent._close_request_openai_client(first, reason="request_complete")
        # Route binding installs the new identity before the fallback client is rebuilt;
        # the credential/URL/header kwargs can remain byte-identical on this mixed relay.
        agent.model = "gemini-4-argon"
        second = agent._create_request_openai_client(reason="test")
        assert isinstance(second, GeminiNativeClient)
        assert second is not first
        agent._close_request_openai_client(second, reason="request_complete")
        assert agent._create_request_openai_client(reason="test") is second
        agent._close_request_openai_client(second, reason="request_complete")
        agent.model = "glm-5.3"
        third = agent._create_request_openai_client(reason="test")
        assert isinstance(third, OpenAI)
        assert third is not second
        agent._close_request_openai_client(third, reason="request_complete")
    finally:
        agent.close()


def test_zen_native_output_reservation_tracks_the_effective_model():
    from agent.agent_runtime_helpers import _update_switch_compressor
    from agent.chat_completion_helpers import _update_fallback_context_compressor
    from agent.gemini_native_adapter import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    from run_agent import AIAgent

    agent = AIAgent(api_key="test-key", base_url="https://opencode.ai/zen/v1", provider="opencode-zen",
                    model="gemini-4-argon", quiet_mode=True, skip_context_files=True, skip_memory=True)
    try:
        assert agent.context_compressor.max_tokens == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
        agent.model = "glm-5.3"
        _update_fallback_context_compressor(agent)
        assert agent.context_compressor.max_tokens in (None, 0)
        agent.model = "gemini-4-argon"
        _update_switch_compressor(agent, [], None, {})
        assert agent.context_compressor.max_tokens == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    finally:
        agent.close()
