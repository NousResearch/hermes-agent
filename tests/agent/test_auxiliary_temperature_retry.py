from types import SimpleNamespace

import pytest

from agent import auxiliary_client


class _SyncCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise RuntimeError(
                "HTTP 400: Error code: 400 - {'detail': 'Unsupported parameter: temperature'}"
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class _SyncClient:
    def __init__(self):
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.chat = SimpleNamespace(completions=_SyncCompletions())


class _AsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise RuntimeError(
                "HTTP 400: Error code: 400 - {'detail': 'Unsupported parameter: temperature'}"
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class _AsyncClient:
    def __init__(self):
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.chat = SimpleNamespace(completions=_AsyncCompletions())


def _patch_resolution(monkeypatch, client):
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        lambda task, provider, model, base_url, api_key: (
            "openai-codex",
            "gpt-5.5",
            "https://chatgpt.com/backend-api/codex",
            "test-key",
            "chat_completions",
        ),
    )
    monkeypatch.setattr(auxiliary_client, "_get_task_extra_body", lambda task: {})
    monkeypatch.setattr(auxiliary_client, "_get_task_timeout", lambda task: 30.0)
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *args, **kwargs: (client, "gpt-5.5"),
    )


def test_call_llm_retries_without_unsupported_temperature(monkeypatch):
    client = _SyncClient()
    _patch_resolution(monkeypatch, client)

    response = auxiliary_client.call_llm(
        task="flush_memories",
        messages=[{"role": "user", "content": "remember this"}],
        temperature=0.3,
        max_tokens=128,
    )

    assert response.choices[0].message.content == "ok"
    assert len(client.chat.completions.calls) == 2
    assert client.chat.completions.calls[0]["temperature"] == 0.3
    assert "temperature" not in client.chat.completions.calls[1]
    assert client.chat.completions.calls[1]["max_tokens"] == 128


@pytest.mark.asyncio
async def test_async_call_llm_retries_without_unsupported_temperature(monkeypatch):
    client = _AsyncClient()
    _patch_resolution(monkeypatch, client)

    response = await auxiliary_client.async_call_llm(
        task="flush_memories",
        messages=[{"role": "user", "content": "remember this"}],
        temperature=0.3,
        max_tokens=128,
    )

    assert response.choices[0].message.content == "ok"
    assert len(client.chat.completions.calls) == 2
    assert client.chat.completions.calls[0]["temperature"] == 0.3
    assert "temperature" not in client.chat.completions.calls[1]
    assert client.chat.completions.calls[1]["max_tokens"] == 128
