"""agent.served_model — per-response served-model capture behind routing proxies (#54864)."""

from __future__ import annotations

import httpx
import openai

from agent.served_model import install_served_model_capture, result_model_fields


class _Agent:
    model = "hermes-router"
    _fallback_activated = False
    _primary_runtime: dict = {}


def test_httpx_hook_captures_litellm_header_and_clears_when_absent():
    served = {"value": "gpt-4o-2024-11-20"}

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"x-litellm-model-id": served["value"]} if served["value"] else {}
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "hermes-router",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }, headers=headers)

    client = openai.OpenAI(api_key="x", base_url="http://proxy.test/v1",
                           http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    agent = _Agent()
    install_served_model_capture(agent, client)
    install_served_model_capture(agent, client)  # idempotent: one hook per client
    assert len(client._client.event_hooks["response"]) == 1

    client.chat.completions.create(model="hermes-router", messages=[{"role": "user", "content": "hi"}])
    assert result_model_fields(agent) == {"requested_model": "hermes-router", "served_model": "gpt-4o-2024-11-20"}

    served["value"] = ""  # next response has no routing header: the stale value must not survive
    client.chat.completions.create(model="hermes-router", messages=[{"role": "user", "content": "hi"}])
    assert result_model_fields(agent) == {"requested_model": "hermes-router", "served_model": None}

    # Hermes' own fallback route surfaces the same way when no proxy header is present.
    agent._fallback_activated = True
    agent._primary_runtime = {"model": "gpt-5.6-sol"}
    agent.model = "qwen/qwen3.8-max"
    assert result_model_fields(agent) == {"requested_model": "gpt-5.6-sol", "served_model": "qwen/qwen3.8-max"}
