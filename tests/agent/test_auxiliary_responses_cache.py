"""Exercise xAI auxiliary cache routing through the real OpenAI SDK."""

import json

import httpx
import pytest
from openai import OpenAI

from agent.auxiliary_client import (
    _CodexCompletionsAdapter,
    reset_runtime_main,
    set_runtime_main,
)
from agent.transports.codex import ResponsesApiTransport, _bounded_prompt_cache_key


@pytest.fixture
def auxiliary_wire():
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append((payload, dict(request.headers)))
        message = {
            "id": "msg_test", "type": "message", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": "OK", "annotations": []}],
        }
        response = {
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": payload["model"], "status": "completed", "output": [message],
            "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
        }
        events = [
            {"type": "response.created", "response": dict(response, status="in_progress", output=[])},
            {"type": "response.output_item.done", "output_index": 0, "item": message},
            {"type": "response.completed", "response": response},
        ]
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    with OpenAI(
        api_key="test-only", base_url="https://api.x.ai/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        yield client, requests


@pytest.mark.parametrize("model", ["grok-4.5", "grok-4.6"])
def test_xai_auxiliary_cache_key_survives_rotation_and_isolates_prefixes(auxiliary_wire, model):
    client, requests = auxiliary_wire
    adapter = _CodexCompletionsAdapter(client, model)
    scopes = ["conversation-A", "conversation-A", "conversation-B", "conversation-A"]
    for index, scope in enumerate(scopes):
        token = set_runtime_main("xai-oauth", model, session_id=f"session-{index}", cache_scope=scope)
        try:
            result = adapter.create(messages=[
                {"role": "system", "content": "Summarize this conversation."},
                {"role": "user", "content": f"Transcript revision {index}"},
            ], extra_headers={"x-test-observer": "preserved"})
            assert result.choices[0].message.content == "OK"
        finally:
            reset_runtime_main(token)
    keys = [body["prompt_cache_key"] for body, _ in requests]
    assert keys[0] == keys[1] == keys[3]
    assert keys[2] != keys[0]
    main = ResponsesApiTransport().build_kwargs(
        model, [{"role": "user", "content": "Hello"}],
        instructions="Main agent instructions.", is_xai_responses=True,
        session_id="session-main", cache_scope_id=scopes[0],
    )
    assert keys[0] != main["extra_body"]["prompt_cache_key"]
    assert all(headers["x-test-observer"] == "preserved" for _, headers in requests)
    assert all("prompt_cache_retention" not in body for body, _ in requests)
    token = set_runtime_main("xai-oauth", model)
    try:
        adapter.create(messages=[{"role": "user", "content": "Unscoped call."}])
    finally:
        reset_runtime_main(token)
    assert "prompt_cache_key" not in requests[-1][0]


@pytest.mark.parametrize("key", ["explicit-auxiliary-key", "long-key-" * 12, ""])
def test_xai_auxiliary_preserves_explicit_cache_routing(auxiliary_wire, key):
    client, requests = auxiliary_wire
    extra_body = {"prompt_cache_key": key, "reasoning": {"enabled": False}}
    token = set_runtime_main("xai-oauth", "grok-4.6", session_id="session-test")
    try:
        _CodexCompletionsAdapter(client, "grok-4.6").create(
            messages=[{"role": "user", "content": "Summarize."}], extra_body=extra_body,
        )
    finally:
        reset_runtime_main(token)
    body, _ = requests[0]
    assert body.get("prompt_cache_key") == _bounded_prompt_cache_key(key)
    assert extra_body == {"prompt_cache_key": key, "reasoning": {"enabled": False}}
