"""Behavior contracts for mixed-wire Zen runtime identities and transitions."""

import json

import httpx
import pytest

from agent import auxiliary_client as aux
from agent.gemini_native_adapter import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS, GeminiNativeClient
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_constants import get_hermes_home
from run_agent import AIAgent


@pytest.fixture
def zen_config(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")
    (get_hermes_home() / "config.yaml").write_text(
        "agent:\n  reasoning_effort: false\nmodel:\n  default_headers:\n    x-contract: preserved\n"
        "custom_providers:\n"
        "  - name: zen-bridge\n    base_url: https://opencode.ai/zen/v1\n    api_key: test-key\n"
        "  - name: zen-pinned\n    base_url: https://opencode.ai/zen/v1\n    api_key: test-key\n"
        "    api_mode: chat_completions\n",
        encoding="utf-8",
    )


def named_agent(name, model="gemini-4-argon", **kwargs):
    runtime = resolve_runtime_provider(requested=name, target_model=model)
    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == name
    return AIAgent(**{key: runtime[key] for key in (
        "provider", "requested_provider", "api_mode", "api_key", "base_url"
    )}, model=model, quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs)


def capture_http(monkeypatch):
    requests = []

    def send(_client, request, **_kwargs):
        requests.append(request)
        if "/models/" in request.url.path:
            payload = {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
        else:
            payload = {"id": "chat-test", "object": "chat.completion", "model": "test-model",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                    "finish_reason": "stop"}]}
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.mark.parametrize("name,native", [("custom:zen-pinned", False), ("custom:zen-bridge", True)])
def test_named_main_resolution_honors_transport_pin(zen_config, monkeypatch, name, native):
    requests = capture_http(monkeypatch)
    agent = named_agent(name)
    try:
        assert isinstance(agent.client, GeminiNativeClient) is native
        client = agent._create_request_openai_client(reason="contract")
        assert isinstance(client, GeminiNativeClient) is native
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "OK"}])
        client.chat.completions.create(**kwargs)
        agent._close_request_openai_client(client, reason="request_complete")
        assert ("/models/" in requests[-1].url.path) is native
        assert agent.context_compressor.max_tokens == (GEMINI_DEFAULT_MAX_OUTPUT_TOKENS if native else None)
    finally:
        agent.close()


def test_named_fallback_identity_reselects_request_cache_and_reasoning(zen_config, monkeypatch):
    from agent.route_binding import bind_route_entry, reinstall_runtime_snapshot

    requests = capture_http(monkeypatch)
    agent = named_agent("custom:zen-pinned", reasoning_config={"enabled": False, "effort": "none"})
    try:
        first = agent._create_request_openai_client(reason="pinned")
        agent._close_request_openai_client(first, reason="request_complete")
        assert bind_route_entry(agent, {}, "custom:zen-bridge", agent.model)
        second = agent._create_request_openai_client(reason="native-fallback")
        assert isinstance(second, GeminiNativeClient)
        assert second is not first
        second.chat.completions.create(**agent._build_api_kwargs([{"role": "user", "content": "OK"}]))
        agent._close_request_openai_client(second, reason="request_complete")
        assert json.loads(requests[-1].content)["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": False, "thinkingLevel": "low"}
        reinstall_runtime_snapshot(agent, agent._primary_runtime)
        third = agent._create_request_openai_client(reason="pinned-restored")
        assert not isinstance(third, GeminiNativeClient)
        assert third is not second
        agent._close_request_openai_client(third, reason="request_complete")
        assert agent.context_compressor.max_tokens is None
    finally:
        agent.close()


@pytest.mark.parametrize("reasoning,thinking", [
    ({"enabled": False, "effort": "high"}, {"includeThoughts": False, "thinkingLevel": "low"}),
    ({"enabled": True, "effort": "none"}, {"includeThoughts": False, "thinkingLevel": "low"}),
    ({"enabled": True, "effort": "high"}, {"includeThoughts": True, "thinkingLevel": "high"}),
    ({"enabled": True, "effort": "ultra"}, {"includeThoughts": True, "thinkingLevel": "high"}),
])
def test_named_main_projects_native_reasoning_without_long_decode(zen_config, monkeypatch, reasoning, thinking):
    requests = capture_http(monkeypatch)
    agent = named_agent("custom:zen-bridge", reasoning_config=reasoning)
    try:
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "OK"}])
        agent.client.chat.completions.create(**kwargs)
        config = json.loads(requests[-1].content)["generationConfig"]
        assert config["thinkingConfig"] == thinking
        assert config["maxOutputTokens"] == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
        assert "reasoning_effort" not in kwargs
        assert "max_output_tokens" not in kwargs.get("extra_body", {})
    finally:
        agent.close()


@pytest.mark.parametrize("primary,fallback", [("gemini-4-argon", "glm-5.3"), ("glm-5.3", "gemini-4-argon")])
def test_fallback_restore_reinstalls_primary_output_reservation(zen_config, primary, fallback):
    from agent.route_binding import bind_route_entry, reinstall_primary_runtime

    agent = AIAgent(provider="opencode-zen", model=primary, api_key="test-key",
                    base_url="https://opencode.ai/zen/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    try:
        original_budget = agent.context_compressor.max_tokens
        original_threshold = agent.context_compressor.threshold_tokens
        rt = agent._primary_runtime
        assert bind_route_entry(agent, {"api_key": "test-key"}, "opencode-zen", fallback)
        assert isinstance(agent.client, GeminiNativeClient) is fallback.startswith("gemini-")
        assert (agent.context_compressor.max_tokens or 0) == (GEMINI_DEFAULT_MAX_OUTPUT_TOKENS if fallback.startswith("gemini-") else 0)
        reinstall_primary_runtime(agent, rt, "opencode-zen", primary, lambda *a: True, lambda *a: None,
                                  prefetched=True)
        assert (agent.context_compressor.max_tokens or 0) == (original_budget or 0)
        assert agent.context_compressor.threshold_tokens == original_threshold
        assert isinstance(agent.client, GeminiNativeClient) is primary.startswith("gemini-")
        assert agent._fallback_activated is False
    finally:
        agent.close()


@pytest.mark.parametrize("main_model,override,wire", [
    ("gemini-4-argon", "glm-5.3", "chat"),
    ("glm-5.3", "gemini-4-argon", "gemini"),
    ("gemini-4-argon", "gpt-5.4", "responses"),
    ("gemini-4-argon", "claude-sonnet-4-6", "messages"),
])
@pytest.mark.parametrize("async_mode", [False, True])
def test_auto_model_override_selects_wire_before_client(zen_config, main_model, override, wire, async_mode):
    import asyncio
    from agent.gemini_native_adapter import AsyncGeminiNativeClient

    main_runtime = {"provider": "opencode-zen", "model": main_model,
                    "api_key": "test-key", "base_url": "https://opencode.ai/zen/v1",
                    "api_mode": "chat_completions"}
    client, model = aux.resolve_provider_client("auto", model=override, main_runtime=main_runtime,
                                               async_mode=async_mode)
    try:
        assert model == override
        native_type = AsyncGeminiNativeClient if async_mode else GeminiNativeClient
        assert isinstance(client, native_type) is (wire == "gemini")
        if wire == "responses":
            expected = aux.AsyncCodexAuxiliaryClient if async_mode else aux.CodexAuxiliaryClient
            assert isinstance(client, expected)
        elif wire == "messages":
            expected = aux.AsyncAnthropicAuxiliaryClient if async_mode else aux.AnthropicAuxiliaryClient
            assert isinstance(client, expected)
        elif wire == "chat":
            from openai import AsyncOpenAI, OpenAI
            assert isinstance(client, AsyncOpenAI if async_mode else OpenAI)
        assert main_runtime["model"] == main_model
    finally:
        if async_mode and wire in {"messages", "responses"}:
            client._real_client.close()
        elif async_mode:
            asyncio.run(client.close())
        else:
            client.close()
