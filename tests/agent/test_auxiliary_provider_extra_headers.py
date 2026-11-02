"""Provider-scoped headers must survive auxiliary routing without crossing endpoints."""

import json
import socket

import httpx
import pytest
import yaml


@pytest.fixture
def provider_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GATEWAY_API_KEY", "offline-gateway-key")
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom:gateway", "default": "test-model",
                  "default_headers": {"X-Keep": "global", "X-Project": "global"}},
        "providers": {
            "gateway": {"api": "https://gateway.example/v1", "key_env": "GATEWAY_API_KEY",
                        "extra_headers": {"X-Project": "gateway-project", "X-Gateway-Token": "gateway-secret"}},
            "fallback": {"api": "https://fallback.example/v1", "api_key": "offline-fallback-key"},
        },
    }), encoding="utf-8")

    def deny_network(*args, **kwargs):
        raise AssertionError("test must not connect to a real endpoint")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    # A fresh disk catalog avoids registry fetches while keeping real config/runtime imports.
    (tmp_path / "models_dev_cache.json").write_text(json.dumps({
        "fixture": {"id": "fixture", "name": "Fixture", "env": [], "models": {}}
    }), encoding="utf-8")
    from agent import models_dev
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    return tmp_path


def capture_requests(monkeypatch):
    captured = []

    def response(request):
        captured.append(request)
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })

    def handle_request(self, request):
        return response(request)

    async def handle_async_request(self, request):
        return response(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request)
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("global_override", [False, True])
@pytest.mark.parametrize("explicit_base_url", [None, "https://GATEWAY.example:443/v1/"])
async def test_provider_headers_reach_auxiliary_wire(
    provider_config, monkeypatch, async_mode, global_override, explicit_base_url,
):
    from agent.auxiliary_client import resolve_provider_client

    if not global_override:
        cfg_path = provider_config / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        del cfg["model"]["default_headers"]["X-Project"]
        cfg_path.write_text(yaml.safe_dump(cfg))
    captured = capture_requests(monkeypatch)
    client, model = resolve_provider_client(
        "custom:gateway", "test-model", async_mode=async_mode,
        explicit_base_url=explicit_base_url,
    )
    try:
        result = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hello"}])
        if async_mode:
            result = await result
        assert result.choices[0].message.content == "ok"
        request, = captured
        assert str(request.url) == "https://gateway.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer offline-gateway-key"
        assert request.headers["X-Project"] == "gateway-project"
        assert request.headers["X-Gateway-Token"] == "gateway-secret"
        assert request.headers["X-Keep"] == "global"
    finally:
        if async_mode:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider, destination", [
    ("custom:fallback", "https://fallback.example/v1"),
    ("custom:gateway", "https://gateway.example.attacker.test/v1"),
    ("custom:gateway", "https://gateway.example/other"),
])
async def test_provider_headers_do_not_follow_auxiliary_override(
    provider_config, monkeypatch, async_mode, provider, destination,
):
    from agent.auxiliary_client import resolve_provider_client

    captured = capture_requests(monkeypatch)
    from agent.auxiliary_client import (
        _resolve_fallback_entry, _to_async_client,
        _call_fallback_candidate_sync, _call_fallback_candidate_async,
    )
    client, model = _resolve_fallback_entry({
        "provider": provider, "model": "test-model", "base_url": destination,
        "api_key": "offline-destination-key",
    })
    if async_mode:
        client, model = _to_async_client(client, model)
    try:
        kwargs = dict(
            task="compression", messages=[{"role": "user", "content": "hello"}],
            temperature=None, max_tokens=None, tools=None, effective_timeout=10,
            effective_extra_body={}, reasoning_config=None,
        )
        if async_mode:
            result = await _call_fallback_candidate_async(client, model, provider, **kwargs)
        else:
            result = _call_fallback_candidate_sync(client, model, provider, **kwargs)
        assert result.choices[0].message.content == "ok"
        request, = captured
        assert str(request.url) == destination + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer offline-destination-key"
        assert "X-Gateway-Token" not in request.headers
        assert request.headers["X-Project"] == "global"
        assert request.headers["X-Keep"] == "global"
    finally:
        if async_mode:
            await client.close()
        else:
            client.close()
