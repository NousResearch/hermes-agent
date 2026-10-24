"""Configured HTTP outage recovery through real YAML routing and SDK requests."""

import json
from collections import Counter

import httpx
import pytest

from agent import auxiliary_client as ac


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", [408, 502, 503, 400, "exhausted"])
async def test_configured_http_failover_preserves_original_payload(
    tmp_path, monkeypatch, async_mode, failure
):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: custom:test-gateway\n  default: vision-primary\n"
        "custom_providers:\n  - name: test-gateway\n"
        "    base_url: https://example.invalid/v1\n    api_key: test-key\n"
        "auxiliary:\n  transient_retries: 1\n  vision:\n"
        "    provider: custom:test-gateway\n    model: vision-primary\n"
        "    fallback_chain:\n"
        "      - {provider: 'custom:test-gateway', model: vision-middle}\n"
        "      - {provider: 'custom:test-gateway', model: vision-final}\n",
        encoding="utf-8",
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
    ]}]
    requests = []
    transports = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        model = payload["model"]
        assert payload["messages"] == messages
        if model == "vision-final" and failure != "exhausted":
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 1, "model": model,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "final vision result"}}],
            })
        status = (502 if failure == "exhausted" else failure) if model == "vision-primary" else 503
        return httpx.Response(status, json={"error": {"message": f"outage on {model}"}})

    def http_client(_base_url, *, async_mode=False):
        client_type = httpx.AsyncClient if async_mode else httpx.Client
        client = client_type(transport=httpx.MockTransport(respond))
        transports.append(client)
        return {"http_client": client}

    monkeypatch.setattr(ac, "_openai_http_client_kwargs", http_client)
    monkeypatch.setattr(ac, "_TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
    ac._reset_aux_unhealthy_cache()
    ac.shutdown_cached_clients()
    route = {}
    try:
        async def call():
            if async_mode:
                return await ac.async_call_llm(task="vision", messages=messages, route_info=route)
            return ac.call_llm(task="vision", messages=messages, route_info=route)

        if failure in (400, "exhausted"):
            with pytest.raises(Exception, match="outage on vision-primary"):
                await call()
        else:
            result = await call()
            assert result.choices[0].message.content == "final vision result"
            assert route["model"] == "vision-final"
        expected = {"vision-primary": 1} if failure == 400 else {
            "vision-primary": 2, "vision-middle": 1, "vision-final": 1,
        }
        assert Counter(row["model"] for row in requests) == expected
        assert not ac._is_provider_unhealthy("custom:test-gateway", "https://example.invalid/v1")
    finally:
        ac.shutdown_cached_clients()
        ac._reset_aux_unhealthy_cache()
        for client in transports:
            if isinstance(client, httpx.AsyncClient):
                await client.aclose()
            else:
                client.close()


def test_configured_walk_is_bounded_when_selector_labels_keep_changing(monkeypatch):
    from types import SimpleNamespace

    selections = []

    def changing_selector(*args, **kwargs):
        selections.append(None)
        if len(selections) > 3:
            pytest.fail("configured selector did not stop at its attempt bound")
        return SimpleNamespace(base_url="https://example.invalid/v1"), "model", f"lane-{len(selections)}"

    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda _task: {
        "fallback_chain": [{"provider": "custom:test", "model": "fallback"}],
    })
    monkeypatch.setattr(ac, "_try_configured_fallback_chain", changing_selector)
    monkeypatch.setattr(ac, "_try_main_agent_model_fallback", lambda *args, **kwargs: (None, None, ""))
    monkeypatch.setattr(ac, "_try_payment_fallback", lambda *args, **kwargs: (None, None, ""))
    route = ac._LadderRoute(**{
        **dict.fromkeys(ac._LadderRoute._fields), "task": "vision", "tag": "",
        "resolved_provider": "custom:test", "base_info": "https://example.invalid/v1",
    })
    error = ConnectionError("upstream connection refused")
    ladder = ac._ladder_provider_fallback(error, route)
    assert ac._drive_ladder(ladder, lambda step: None) is None
    assert len(selections) == 2
