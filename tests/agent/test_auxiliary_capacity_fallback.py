"""Exercise config loading and both public calls through actual SDK clients."""
import json

import httpx
import pytest
from openai import OpenAI, AsyncOpenAI

from agent import auxiliary_client as ac
from agent.auxiliary_task_config import _get_auxiliary_task_config


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", [402, 408, 503, 504])
@pytest.mark.parametrize("key", ["fallback_chain", "fallback_providers"])
@pytest.mark.asyncio
async def test_configured_capacity_fallback(tmp_path, monkeypatch, async_mode, status, key):
    (tmp_path / "config.yaml").write_text(json.dumps({"auxiliary": {
        "transient_retries": 0,
        "title_generation": {"provider": "custom", "model": "primary",
            "base_url": "https://primary.example/v1", "api_key": "primary-key",
            key: [{"provider": "custom", "model": "backup",
                "base_url": "https://backup.example/v1", "api_key": "backup-key"}]}}}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ac._reset_aux_unhealthy_cache()
    requests = []
    clients = []

    def respond(request):
        requests.append(request)
        if request.url.host == "primary.example":
            return httpx.Response(status, json={"error": {"message": "capacity unavailable"}})
        return httpx.Response(200, json={"id": "ok", "object": "chat.completion", "created": 0,
            "model": "backup", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "recovered"}}]})

    def make_client(url, token, model):
        cls, http_cls = (AsyncOpenAI, httpx.AsyncClient) if async_mode else (OpenAI, httpx.Client)
        client = cls(api_key=token, base_url=url, max_retries=0,
            http_client=http_cls(transport=httpx.MockTransport(respond)))
        clients.append(client)
        return client, model

    def cached(*args, **kwargs):
        return make_client("https://primary.example/v1", "primary-key", "primary")

    def resolve(provider, model=None, explicit_base_url=None, explicit_api_key=None, **kwargs):
        assert provider == "custom"
        return make_client(explicit_base_url, explicit_api_key, model)

    monkeypatch.setattr(ac, "_get_cached_client", cached)
    monkeypatch.setattr(ac, "resolve_provider_client", resolve)
    monkeypatch.setattr(ac, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
    if async_mode:
        monkeypatch.setattr(ac, "_to_async_client", lambda c, m, **kw: (c, m))
    try:
        kwargs = dict(task="title_generation", messages=[{"role": "user", "content": "hi"}])
        result = await ac.async_call_llm(**kwargs) if async_mode else ac.call_llm(**kwargs)
        assert result.choices[0].message.content == "recovered"
        assert requests[-1].url.host == "backup.example"
        assert requests[-1].headers["authorization"] == "Bearer backup-key"
        assert json.loads(requests[-1].content)["model"] == "backup"
        assert sum(r.url.host == "primary.example" for r in requests) == (2 if async_mode and status in (408, 503, 504) else 1)
    finally:
        for client in clients:
            if async_mode:
                await client.close()
            else:
                client.close()
        ac._reset_aux_unhealthy_cache()


def test_existing_chain_wins_when_alias_also_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"auxiliary": {"title_generation": {
        "fallback_chain": [], "fallback_providers": [{"provider": "custom", "model": "other"}]}}}))
    assert _get_auxiliary_task_config("title_generation")["fallback_chain"] == []
