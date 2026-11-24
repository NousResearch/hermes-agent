"""Model HTTP observers survive auxiliary client reconstruction; other facades are unchanged."""

from types import SimpleNamespace

import httpx
import pytest

from agent import auxiliary_client
from agent.agent_runtime_helpers import create_openai_client
from hermes_cli import plugins
from providers import register_provider
from providers.base import ProviderProfile


@pytest.mark.asyncio
async def test_auxiliary_sync_and_rebuilt_async_clients_observe_responses(monkeypatch):
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    context = plugins.PluginContext(plugins.PluginManifest(name="observer"), manager)
    observed = []
    context.register_hook("llm_http_response", lambda status_code: observed.append(status_code))
    manager._publish_staged_registrations(context)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        "id": "completion", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }))
    try:
        async with httpx.AsyncClient(transport=transport) as async_http:
            monkeypatch.setattr(auxiliary_client, "_openai_http_client_kwargs",
                                lambda _url, *, async_mode=False: {"http_client": async_http} if async_mode else {})
            with httpx.Client(transport=transport) as sync_http, auxiliary_client._create_openai_client(
                api_key="test-key", base_url="https://api.example.com/v1", http_client=sync_http,
            ) as sync:
                asynchronous, model = auxiliary_client._to_async_client(sync, "test")
                async with asynchronous:
                    messages = [{"role": "user", "content": "hi"}]
                    sync.chat.completions.create(model=model, messages=messages)
                    await asynchronous.chat.completions.create(model=model, messages=messages)
                    assert asynchronous.api_key == sync.api_key == "test-key"
                    assert asynchronous.base_url == sync.base_url
        assert observed == [200, 200]
    finally:
        manager.unload()


def test_custom_non_http_facades_remain_unchanged(monkeypatch):
    import providers

    providers._discover_providers()
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    facade = SimpleNamespace(HERMES_SKIP_ASYNC_WRAP=True, _client={})

    class CustomProfile(ProviderProfile):
        def create_client(self, **_kwargs):
            return facade

    register_provider(CustomProfile(name="response-hook-test", base_url="https://api.example.com/v1"))
    agent = SimpleNamespace(provider="response-hook-test", _client_log_context=lambda: "")
    kwargs = {"api_key": "test-key", "base_url": "https://api.example.com/v1"}
    assert create_openai_client(agent, kwargs, reason="test", shared=False) is facade
    assert auxiliary_client._api_key_profile_supplied_client("response-hook-test", **kwargs) is facade
    assert auxiliary_client._to_async_client(facade, "test") == (facade, "test")
