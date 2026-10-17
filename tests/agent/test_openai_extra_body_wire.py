"""Wire-shape coverage for OpenAI SDK ``extra_body`` handling."""

import json

import httpx
from openai import OpenAI

from agent.auxiliary_client import _build_call_kwargs
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


def _capture_openai_chat_body(api_kwargs: dict) -> dict:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": api_kwargs["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(
        api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
        http_client=http_client,
    )
    try:
        client.chat.completions.create(**api_kwargs)
    finally:
        client.close()

    assert len(captured) == 1
    return captured[0]


def test_transport_extra_body_is_flattened_on_openai_wire():
    transport = ChatCompletionsTransport()
    profile = get_provider_profile("nvidia")

    kwargs = transport.build_kwargs(
        model="minimaxai/minimax-m3",
        messages=[{"role": "user", "content": "hi"}],
        provider_profile=profile,
        max_tokens_param_fn=lambda n: {"max_tokens": n},
        request_overrides={
            "extra_body": {
                "chat_template_kwargs": {"thinking_mode": "enabled"},
                "reasoning_budget": 8192,
            }
        },
    )

    assert kwargs["extra_body"]["chat_template_kwargs"] == {
        "thinking_mode": "enabled"
    }
    assert "chat_template_kwargs" not in kwargs

    body = _capture_openai_chat_body(kwargs)

    assert body["chat_template_kwargs"] == {"thinking_mode": "enabled"}
    assert body["reasoning_budget"] == 8192
    assert "extra_body" not in body


def test_auxiliary_extra_body_is_flattened_on_openai_wire():
    kwargs = _build_call_kwargs(
        provider="nvidia",
        model="minimaxai/minimax-m3",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={
            "chat_template_kwargs": {"thinking_mode": "enabled"},
            "reasoning_budget": 8192,
        },
        base_url="https://integrate.api.nvidia.com/v1",
    )

    assert kwargs["extra_body"]["chat_template_kwargs"] == {
        "thinking_mode": "enabled"
    }
    assert "chat_template_kwargs" not in kwargs

    body = _capture_openai_chat_body(kwargs)

    assert body["chat_template_kwargs"] == {"thinking_mode": "enabled"}
    assert body["reasoning_budget"] == 8192
    assert "extra_body" not in body
