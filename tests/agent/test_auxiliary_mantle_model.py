"""Auxiliary title/vision requests retain Mantle IDs and bearer authentication."""
import asyncio
import json

import httpx
import pytest

from agent import auxiliary_client as aux

MANTLE_BASE = "https://bedrock-mantle.us-east-1.api.aws/anthropic"


@pytest.mark.parametrize("async_mode", [False, True], ids=["title", "vision"])
@pytest.mark.parametrize("base_url, model, expected", [
    (MANTLE_BASE, "anthropic.claude-opus-5", "anthropic.claude-opus-5"),
    ("https://api.anthropic.com", "claude-sonnet-4.6", "claude-sonnet-4-6"),
])
def test_auxiliary_model_and_auth_on_wire(monkeypatch, base_url, model, expected, async_mode):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-native-test")
    monkeypatch.delenv("BEDROCK_MANTLE_WORKSPACE_ID", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    monkeypatch.setattr("hermes_cli.config.get_custom_provider_extra_headers", lambda *a: {})
    monkeypatch.setattr(aux, "_select_pool_entry", lambda *a: (False, None))
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        message = {"id": "msg_test", "type": "message", "role": "assistant",
                   "model": expected, "content": [], "stop_reason": None,
                   "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}
        events = [
            {"type": "message_start", "message": message},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Test title"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=body, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    client, resolved = aux.resolve_provider_client(
        "anthropic", model=model, explicit_base_url=base_url,
        explicit_api_key="sk-ant-api03-native-test", async_mode=async_mode, is_vision=async_mode,
    )
    try:
        result = client.chat.completions.create(
            model=resolved, messages=[{"role": "user", "content": "Describe this session"}], max_tokens=32,
        )
        if async_mode:
            result = asyncio.run(result)
        assert result.choices[0].message.content == "Test title"
        request, = requests
        assert json.loads(request.content)["model"] == expected
        if base_url == MANTLE_BASE:
            assert request.headers["authorization"] == "Bearer bedrock-test-key"
            assert "x-api-key" not in request.headers
        else:
            assert request.headers["x-api-key"] == "sk-ant-api03-native-test"
    finally:
        client._real_client.close()
