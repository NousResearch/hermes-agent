"""Private Anthropic reasoning must follow the selected auxiliary transport."""
import asyncio
import json

import httpx
import pytest

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("base_url", ["https://api.minimax.io/v1", "https://relay.example/v1",
                                      "https://relay.example/anthropic"])
def test_public_minimax_chat_route_accepts_reasoning(monkeypatch, base_url, async_mode):
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {
        "provider": "minimax", "api_mode": "chat_completions"})
    requests = []

    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            "id": "fixture", "object": "chat.completion", "created": 0,
            "model": "MiniMax-M3", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "Fixture title"},
                "finish_reason": "stop"}],
        })

    monkeypatch.setattr(httpx.Client, "send", lambda self, request, **kw: response(request))

    async def async_send(self, request, **kw):
        return response(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    options = dict(task="title_generation", provider="minimax", model="MiniMax-M3",
                   base_url=base_url, api_key="fixture-only",
                   messages=[{"role": "user", "content": "Make a title"}],
                   reasoning_config={"enabled": False})
    error = None
    try:
        result = asyncio.run(aux.async_call_llm(**options)) if async_mode else aux.call_llm(**options)
    except TypeError as exc:
        error = exc
    assert error is None, f"Plain OpenAI SDK rejected auxiliary request: {error}"
    assert result.choices[0].message.content == "Fixture title"
    assert len(requests) == 1
    assert "_reasoning_config" not in requests[0]


def _messages_client(async_mode=False):
    from types import SimpleNamespace

    captured = []

    class Messages:
        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Fixture title")],
                stop_reason="end_turn", usage=None,
            )

    client = aux.AnthropicAuxiliaryClient(
        SimpleNamespace(messages=Messages()), "claude-sonnet-4-6", "fixture-only",
        "https://relay.example/v1",
    )
    return (aux.AsyncAnthropicAuxiliaryClient(client) if async_mode else client), captured


@pytest.mark.parametrize("async_mode", [False, True])
def test_messages_adapter_preserves_explicit_reasoning_on_unmarked_url(monkeypatch, async_mode):
    client, captured = _messages_client(async_mode)
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (client, "claude-sonnet-4-6"))
    options = dict(task="title_generation", provider="custom", model="claude-sonnet-4-6",
                   base_url=client.base_url, api_key="fixture-only",
                   messages=[{"role": "user", "content": "Title"}],
                   reasoning_config={"enabled": False},
                   extra_body={"reasoning": {"enabled": True, "effort": "high"}})
    result = asyncio.run(aux.async_call_llm(**options)) if async_mode else aux.call_llm(**options)
    assert result.choices[0].message.content == "Fixture title"
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert options["reasoning_config"] == {"enabled": False}
    assert options["extra_body"]["reasoning"]["enabled"] is True


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("reasoning", [None, {}, {"enabled": False}])
def test_builder_private_control_owned_by_client(async_mode, reasoning):
    client, _ = _messages_client(async_mode)
    for destination in [client, object(), None]:
        kwargs = aux._build_call_kwargs(
            "minimax", "MiniMax-M3", [{"role": "user", "content": "Title"}],
            reasoning_config=reasoning, base_url="https://api.minimax.io/anthropic",
            client=destination,
        )
        assert ("_reasoning_config" in kwargs) == (destination is client and bool(reasoning))


@pytest.mark.parametrize("async_mode", [False, True])
def test_retry_and_fallback_rebuild_use_new_client(monkeypatch, async_mode):
    from types import SimpleNamespace

    messages_client, _ = _messages_client(async_mode)
    plain_client = SimpleNamespace(base_url="https://api.minimax.io/v1")
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    request = dict(messages=[{"role": "user", "content": "Title"}], tools=None,
                   temperature=None, max_tokens=32, effective_extra_body={},
                   reasoning_config={"enabled": False})
    for initial, rebuilt in [(messages_client, plain_client), (plain_client, messages_client)]:
        _, first, rebuild = aux._plan_fallback_candidate(
            initial, "MiniMax-M3", "minimax", task="title_generation",
            effective_timeout=12, apply_fast_lane=False, **request,
        )
        _, second = rebuild("minimax", rebuilt, "MiniMax-M3")
        assert ("_reasoning_config" in first) == (initial is messages_client)
        assert ("_reasoning_config" in second) == (rebuilt is messages_client)
        assert second["timeout"] == 12
        monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (rebuilt, "MiniMax-M3"))
        actual, retry = aux._prepare_same_provider_retry(
            task="title_generation", resolved_provider="minimax", resolved_model="MiniMax-M3",
            resolved_base_url=rebuilt.base_url, resolved_api_key="fixture-only",
            resolved_api_mode="chat_completions", main_runtime={}, final_model="MiniMax-M3",
            effective_timeout=12, async_mode=async_mode, **request,
        )
        assert actual is rebuilt
        assert ("_reasoning_config" in retry) == (rebuilt is messages_client)
