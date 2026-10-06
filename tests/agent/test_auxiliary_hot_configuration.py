"""Real auto auxiliary calls must use the current task and named-provider configuration."""
import json
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux


@pytest.fixture(autouse=True)
def _clean_aux_state():
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    aux._reset_aux_unhealthy_cache()
    yield
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    aux._reset_aux_unhealthy_cache()


@pytest.fixture
def hot_config(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    endpoints = ("https://cache-a.invalid/v1", "https://cache-b.invalid/v1")
    clients = []

    class Client:
        def __init__(self, base_url):
            self.base_url = base_url
            self.calls = []
            self.is_closed = False
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="synthetic reply"), finish_reason="stop")])

        def close(self):
            self.is_closed = True

    def factory(*, api_key, base_url, **unused):
        assert api_key == "synthetic-key"
        assert base_url in endpoints
        client = Client(base_url)
        clients.append(client)
        return client

    monkeypatch.setattr(aux, "_create_openai_client", factory)
    main_runtime = {
        "provider": "custom", "model": "test-model",
        "base_url": "https://main.invalid/v1", "api_key": "synthetic-key",
        "api_mode": "chat_completions",
    }
    aux._mark_provider_unhealthy("custom", base_url=main_runtime["base_url"])

    def write(provider="cache-a", endpoint=endpoints[0]):
        config = {
            "custom_providers": [
                {"name": name, "base_url": url, "api_key": "synthetic-key",
                 "api_mode": "chat_completions"}
                for name, url in [("cache-a", endpoint), ("cache-b", endpoints[1])]
            ],
            "auxiliary": {"title_generation": {
                "fallback_chain": [{"provider": provider, "model": "test-model"}]}}
        }
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    def call():
        before = {id(client): len(client.calls) for client in clients}
        aux.call_llm(
            "title_generation", provider="auto", model="test-model",
            main_runtime=main_runtime,
            messages=[{"role": "user", "content": "synthetic request"}],
        )
        used = [client for client in clients
                if len(client.calls) > before.get(id(client), 0)]
        assert len(used) == 1
        return used[0].base_url

    return write, call, endpoints


def test_auto_cache_tracks_hot_named_destination_a_b_a(hot_config):
    write, call, endpoints = hot_config
    observed = []
    for endpoint in (endpoints[0], endpoints[1], endpoints[0]):
        write(endpoint=endpoint)
        observed.append(call())
    assert observed == [endpoints[0], endpoints[1], endpoints[0]]


def test_auto_cache_tracks_hot_task_chain_a_b_a(hot_config):
    write, call, endpoints = hot_config
    observed = []
    for provider in ("cache-a", "cache-b", "cache-a"):
        write(provider=provider)
        observed.append(call())
    assert observed == [endpoints[0], endpoints[1], endpoints[0]]
