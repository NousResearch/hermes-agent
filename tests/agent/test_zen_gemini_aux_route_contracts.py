"""End-to-end contracts for automatic auxiliary routing on mixed-wire relays."""

import asyncio
import json
import time

import httpx
import pytest

from agent import auxiliary_client as aux
from agent.gemini_native_adapter import AsyncGeminiNativeClient, GeminiNativeClient
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_constants import get_hermes_home
from run_agent import AIAgent


@pytest.fixture
def routing_config(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "configured-key")
    config = {
        "agent": {"reasoning_effort": False},
        "model": {"provider": "auto", "default": ""},
        "custom_providers": [
            {"name": "zen-pinned", "base_url": "https://opencode.ai/zen/v1", "api_key": "stale-key",
             "api_mode": "chat_completions"},
            {"name": "zen-bridge", "base_url": "https://opencode.ai/zen/v1", "api_key": "stale-key"},
        ],
    }
    write_config(config)
    aux.clear_runtime_main()
    yield config
    aux.clear_runtime_main()
    with aux._client_cache_lock:
        entries = list(aux._client_cache.values())
        aux._client_cache.clear()
    for client, _, _ in entries:
        close_client(client)


def write_config(config):
    import hermes_yaml as yaml
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def capture_http(monkeypatch):
    requests = []

    def response(request):
        requests.append(request)
        if "/models/" in request.url.path:
            payload = {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
        else:
            payload = {"id": "chat-test", "object": "chat.completion", "model": "test-model",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                    "finish_reason": "stop"}]}
        if request.url.path.endswith("streamGenerateContent") or json.loads(request.content).get("stream"):
            if "/models/" not in request.url.path:
                payload["choices"][0]["delta"] = payload["choices"][0].pop("message")
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, request=request, json=payload)

    def send(_client, request, **_kwargs):
        return response(request)

    async def async_send(_client, request, **_kwargs):
        return response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    return requests


def close_client(client):
    if isinstance(client, (aux.AsyncCodexAuxiliaryClient, aux.AsyncAnthropicAuxiliaryClient, AsyncGeminiNativeClient)):
        client._real_client.close()
    else:
        result = client.close()
        if asyncio.iscoroutine(result):
            asyncio.run(result)


def named_runtime(name, callable_key=False):
    runtime = resolve_runtime_provider(requested=name, target_model="gemini-4-argon")
    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == name
    runtime.update(model="gemini-4-argon", base_url="https://opencode.ai/zen/v1?live=1",
                   api_key=(lambda: "live-key") if callable_key else "live-key")
    return runtime


@pytest.mark.parametrize("name,native", [("custom:zen-pinned", False), ("custom:zen-bridge", True)])
@pytest.mark.parametrize("callable_key", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_named_auto_aux_preserves_live_route(routing_config, monkeypatch, name, native, callable_key, async_mode):
    requests = capture_http(monkeypatch)
    runtime = named_runtime(name, callable_key)

    async def call():
        client, model = aux.resolve_provider_client("auto", main_runtime=runtime, async_mode=async_mode)
        try:
            assert isinstance(client, AsyncGeminiNativeClient if async_mode else GeminiNativeClient) is native
            kwargs = {"model": model, "messages": [{"role": "user", "content": "OK"}]}
            if async_mode:
                await client.chat.completions.create(**kwargs)
                await aux.async_call_llm(task="compression", main_runtime=runtime, messages=kwargs["messages"])
                await aux.async_call_llm(task="compression", main_runtime=runtime, messages=kwargs["messages"])
            else:
                client.chat.completions.create(**kwargs)
                aux.call_llm(task="compression", main_runtime=runtime, messages=kwargs["messages"])
                aux.call_llm(task="compression", main_runtime=runtime, messages=kwargs["messages"])
            assert len(requests) == 3
            assert all(("/models/" in request.url.path) is native for request in requests)
            assert all(request.url.params["live"] == "1" for request in requests)
            assert all(request.headers.get("x-goog-api-key") == "live-key" if native else
                       request.headers.get("authorization") == "Bearer live-key" for request in requests)
        finally:
            if async_mode:
                await client.close()
            else:
                client.close()

    asyncio.run(call())


@pytest.mark.parametrize("async_mode", [False, True])
def test_auto_cache_tracks_named_pin_identity(routing_config, monkeypatch, async_mode):
    requests = capture_http(monkeypatch)

    async def call():
        for name in ("custom:zen-pinned", "custom:zen-bridge", "custom:zen-pinned"):
            kwargs = {"task": "compression", "main_runtime": named_runtime(name),
                      "messages": [{"role": "user", "content": "OK"}]}
            if async_mode:
                await aux.async_call_llm(**kwargs)
            else:
                aux.call_llm(**kwargs)
        assert ["/models/" in request.url.path for request in requests] == [False, True, False]

    asyncio.run(call())


@pytest.mark.parametrize("name,native", [("custom:zen-pinned", False), ("custom:zen-bridge", True)])
@pytest.mark.parametrize("callable_key", [False, True])
def test_named_main_compressor_retains_scoped_identity(routing_config, monkeypatch, name, native, callable_key):
    requests = capture_http(monkeypatch)
    runtime = named_runtime(name, callable_key)
    agent = AIAgent(**{key: runtime[key] for key in (
        "model", "provider", "requested_provider", "api_mode", "api_key", "base_url"
    )}, quiet_mode=True, skip_context_files=True, skip_memory=True)
    try:
        with aux.scoped_runtime_main(runtime):
            assert agent.context_compressor._call_summary_llm("summarize", time.monotonic()) == "OK"
        assert ("/models/" in requests[-1].url.path) is native
        assert requests[-1].url.params["live"] == "1"
        assert (requests[-1].headers.get("x-goog-api-key") == "live-key" if native else
                requests[-1].headers.get("authorization") == "Bearer live-key")
    finally:
        agent.close()


@pytest.mark.parametrize("main_model,override,wire", [
    ("gemini-4-argon", "glm-4.6v", "chat"),
    ("glm-5.3", "gemini-4-argon", "gemini"),
    ("gemini-4-argon", "gpt-5.4", "responses"),
    ("gemini-4-argon", "claude-sonnet-4-6", "messages"),
])
@pytest.mark.parametrize("async_mode", [False, True])
def test_vision_auto_selects_effective_model_before_wire(routing_config, monkeypatch, main_model, override, wire, async_mode):
    requests = capture_http(monkeypatch)
    runtime = resolve_runtime_provider(requested="opencode-zen", target_model=main_model)
    runtime["model"] = main_model
    routing_config["auxiliary"] = {"vision": {"model": override}}
    write_config(routing_config)

    async def call():
        provider, client, model = aux.resolve_vision_provider_client(main_runtime=runtime, async_mode=async_mode)
        try:
            assert provider == "opencode-zen"
            assert model == override
            native_type = AsyncGeminiNativeClient if async_mode else GeminiNativeClient
            assert isinstance(client, native_type) is (wire == "gemini")
            if wire == "responses":
                assert isinstance(client, aux.AsyncCodexAuxiliaryClient if async_mode else aux.CodexAuxiliaryClient)
            elif wire == "messages":
                assert isinstance(client, aux.AsyncAnthropicAuxiliaryClient if async_mode else aux.AnthropicAuxiliaryClient)
            else:
                kwargs = {"model": model, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ]}]}
                if async_mode:
                    await client.chat.completions.create(**kwargs)
                else:
                    client.chat.completions.create(**kwargs)
                assert ("/models/" in requests[-1].url.path) is (wire == "gemini")
                if wire == "chat":
                    assert json.loads(requests[-1].content)["model"] == override
        finally:
            if async_mode and wire in {"chat", "gemini"}:
                await client.close()
            else:
                close_client(client)

    asyncio.run(call())


@pytest.mark.parametrize("name,native", [("custom:zen-pinned", False), ("custom:zen-bridge", True)])
@pytest.mark.parametrize("async_mode", [False, True])
def test_vision_auto_keeps_named_pin(routing_config, monkeypatch, name, native, async_mode):
    runtime = named_runtime(name, callable_key=True)
    provider, client, model = aux.resolve_vision_provider_client(main_runtime=runtime, async_mode=async_mode)
    try:
        assert provider == name
        assert model == "gemini-4-argon"
        assert isinstance(client, AsyncGeminiNativeClient if async_mode else GeminiNativeClient) is native
    finally:
        close_client(client)


@pytest.mark.parametrize("entrypoint", ["resolve", "text"])
def test_discovery_openrouter_keeps_explicit_model(routing_config, monkeypatch, entrypoint):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    requests = capture_http(monkeypatch)
    override = "google/gemini-4-argon"
    routing_config["auxiliary"] = {"compression": {"model": override}}
    write_config(routing_config)
    if entrypoint == "resolve":
        client, model = aux.resolve_provider_client("auto", model=override, main_runtime={"provider": "auto"})
    else:
        client, model = aux.get_text_auxiliary_client("compression", main_runtime={"provider": "auto"})
    try:
        assert model == override
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "OK"}])
        assert requests[-1].url.host == "openrouter.ai"
        assert json.loads(requests[-1].content)["model"] == override
    finally:
        close_client(client)


@pytest.mark.parametrize("override,native", [("glm-5.3", False), ("gemini-4-argon", True)])
@pytest.mark.parametrize("entrypoint", ["resolve", "text"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_discovery_zen_override_selects_client_wire(routing_config, monkeypatch, override, native, entrypoint, async_mode):
    requests = capture_http(monkeypatch)
    # There is no selected main endpoint: only Zen's configured credential can win discovery.
    routing_config["auxiliary"] = {"compression": {"model": override}}
    write_config(routing_config)

    async def call():
        if entrypoint == "text":
            client, model = aux.get_text_auxiliary_client("compression", main_runtime={"provider": "auto"})
        else:
            client, model = aux.resolve_provider_client("auto", model=override, main_runtime={"provider": "auto"},
                                                       async_mode=async_mode)
        try:
            assert model == override
            assert isinstance(client, AsyncGeminiNativeClient if async_mode and entrypoint == "resolve" else GeminiNativeClient) is native
            kwargs = {"model": model, "messages": [{"role": "user", "content": "OK"}]}
            if async_mode and entrypoint == "resolve":
                await client.chat.completions.create(**kwargs)
            else:
                client.chat.completions.create(**kwargs)
            assert ("/models/" in requests[-1].url.path) is native
        finally:
            if async_mode and entrypoint == "resolve":
                await client.close()
            else:
                close_client(client)

    asyncio.run(call())


@pytest.mark.parametrize("is_vision", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_active_noncustom_route_ignores_old_custom_request(routing_config, monkeypatch, is_vision, async_mode):
    runtime = resolve_runtime_provider(requested="opencode-zen", target_model="gemini-4-argon")
    runtime.update(model="gemini-4-argon", requested_provider="custom:zen-pinned")
    if is_vision:
        provider, client, model = aux.resolve_vision_provider_client(main_runtime=runtime, async_mode=async_mode)
        assert provider == "opencode-zen"
    else:
        client, model = aux.resolve_provider_client("auto", main_runtime=runtime, async_mode=async_mode)
    try:
        assert model == "gemini-4-argon"
        assert isinstance(client, AsyncGeminiNativeClient if async_mode else GeminiNativeClient)
    finally:
        close_client(client)


def test_discovery_anthropic_keeps_compatible_model(routing_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    client, model = aux.resolve_provider_client("auto", model="claude-sonnet-4-6", main_runtime={"provider": "auto"})
    try:
        assert isinstance(client, aux.AnthropicAuxiliaryClient)
        assert model == "claude-sonnet-4-6"
    finally:
        close_client(client)


def test_discovery_discards_incompatible_aggregator_model(routing_config, monkeypatch):
    capture_http(monkeypatch)
    client, model = aux.resolve_provider_client("auto", model="google/gemini-4-argon", main_runtime={"provider": "auto"})
    try:
        assert model != "google/gemini-4-argon"
        assert "/" not in model
    finally:
        close_client(client)
