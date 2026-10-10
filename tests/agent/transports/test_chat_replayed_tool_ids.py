"""Lossless Chat Completions replay of recurring tool IDs (regression for #135145)."""

from copy import deepcopy
import json

import httpx
from openai import OpenAI
import pytest

from agent.agent_runtime_helpers import sanitize_api_messages
from agent.transports.chat_completions import ChatCompletionsTransport


def _history(ids, *, bridge=False):
    messages = [{"role": "system", "content": "Use the tools to answer."}]
    for index, call_id in enumerate(ids):
        name = "lookup" if index % 2 == 0 else "terminal"
        call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({"index": index})},
            "extra_content": {"google": {"thought_signature": f"signed-{index}"}},
        }
        result_id = call_id
        if bridge:
            call.update(id=f"item_{index}", call_id=call_id, response_item_id=f"item_{index}")
            result_id = f"{call_id}|item_{index}"
        messages.extend([
            {"role": "user", "content": f"Request {index}"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": result_id, "name": name, "content": f"result-{index}"},
            {"role": "assistant", "content": f"Answer {index}"},
        ])
    return messages


@pytest.mark.parametrize("bridge", [False, True])
@pytest.mark.parametrize("ids", [
    ["call_0"] * 50,
    ["call_0", "call_0", "call_0_d2", "call_0", "call_0_d2"],
    ["call_0_d2", "call_0", "call_0", "call_0_d3", "call_0"],
])
def test_replayed_pairs_reach_strict_chat_endpoint_without_loss(ids, bridge):
    history = _history(ids, bridge=bridge)
    original = deepcopy(history)
    received = []

    def endpoint(request):
        messages = json.loads(request.content)["messages"]
        seen, pending = set(), {}
        for message in messages:
            for call in message.get("tool_calls", []):
                cid = call["id"]
                if cid in seen:
                    return httpx.Response(400, json={"error": {"message": "Duplicate tool ID", "code": "INVALID_ARGUMENT"}})
                seen.add(cid)
                pending[cid] = json.loads(call["function"]["arguments"])["index"]
            if message["role"] == "tool":
                cid = message["tool_call_id"]
                if cid not in pending or message["content"] != f"result-{pending.pop(cid)}":
                    return httpx.Response(400, json={"error": {"message": "Unpaired tool result"}})
                received.append(message["content"])
        assert not pending
        return httpx.Response(200, json={
            "id": "reply", "object": "chat.completion", "created": 0, "model": "gemini-test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Done"}}],
        })

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="gemini-test", messages=sanitize_api_messages(history), tools=None,
    )
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http_client:
        with OpenAI(api_key="test", base_url="https://example.test/v1", http_client=http_client, max_retries=0) as client:
            response = client.chat.completions.create(**kwargs)
    assert response.choices[0].message.content == "Done"
    assert received == [f"result-{index}" for index in range(len(ids))]
    assert history == original
    calls = [call for row in kwargs["messages"] for call in row.get("tool_calls", [])]
    assert [call["extra_content"]["google"]["thought_signature"] for call in calls] == [f"signed-{index}" for index in range(len(ids))]


@pytest.mark.parametrize("bridge", [False, True])
def test_wire_repair_is_idempotent_and_keeps_append_only_prefix(bridge):
    transport = ChatCompletionsTransport()
    history = _history(["call_0", "call_0"], bridge=bridge)
    original = deepcopy(history)
    wire = transport.convert_messages(history, model="gemini-test")
    longer = _history(["call_0", "call_0", "call_0_d2", "call_0"], bridge=bridge)
    extended = transport.convert_messages(longer, model="gemini-test")
    assert extended[:len(wire)] == wire
    assert transport.convert_messages(wire, model="gemini-test") == wire
    calls = [tc["id"] for row in wire for tc in row.get("tool_calls", [])]
    assert len(set(calls)) == len(calls)
    assert history == original
    clean = [{"role": "user", "content": "Hello"}]
    assert transport.convert_messages(clean) is clean
