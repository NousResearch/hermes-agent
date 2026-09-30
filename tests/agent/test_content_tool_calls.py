"""Opt-in content calls must execute through the normal, cache-safe tool round."""

import json
from unittest.mock import patch

import httpx
import pytest

from hermes_constants import get_hermes_home
from run_agent import AIAgent


@pytest.fixture
def make_agent(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
    clients = []

    def create(opt_in, handler):
        # Exercise the real readonly YAML loader and AIAgent initialization.
        (get_hermes_home() / "config.yaml").write_text(
            "model:\n  context_length: 128000\n"
            f"  content_tool_calls_from_content: {json.dumps(opt_in)}\n",
            encoding="utf-8",
        )

        def http_client(*args, **kwargs):
            client = httpx.Client(transport=httpx.MockTransport(handler))
            clients.append(client)
            return client

        monkeypatch.setattr(
            "agent.process_bootstrap.build_keepalive_http_client", http_client
        )
        with (
            patch("model_tools.get_tool_definitions", return_value=[tool]),
            patch("model_tools.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(
                model="test-model",
                api_key="offline-fixture",
                base_url="https://content-tools.invalid/v1",
                api_mode="chat_completions",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                save_trajectories=False,
                max_iterations=3,
            )
        agent._cached_system_prompt = "Keep this system prompt unchanged."
        agent.compression_enabled = False
        return agent

    yield create
    for client in clients:
        client.close()


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize(
    "opt_in,change,native,execute",
    [
        (True, None, False, True),
        (False, None, False, False),
        ("true", None, False, False),
        (True, "model", False, False),
        (True, "base_url", False, False),
        (False, None, True, True),
        (True, None, True, True),
    ],
)
def test_config_and_scope_control_real_tool_round(
    make_agent, opt_in, change, native, execute, stream
):
    content = json.dumps({"name": "search", "arguments": {"query": "local fixture"}})
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        first = len(requests) == 1
        message = {"role": "assistant", "content": content if first else "Finished."}
        if first and native:
            message["tool_calls"] = [
                {
                    "id": "native-call",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "native"}',
                    },
                }
            ]
        finish_reason = "tool_calls" if first and native else "stop"
        payload = {
            "id": "offline-response",
            "object": "chat.completion.chunk"
            if body.get("stream")
            else "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
        }
        if body.get("stream"):
            for index, call in enumerate(message.get("tool_calls", [])):
                call["index"] = index
            payload["choices"][0] = {
                "index": 0,
                "delta": message,
                "finish_reason": finish_reason,
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode(),
            )
        return httpx.Response(200, json=payload)

    agent = make_agent(opt_in, respond)
    agent._disable_streaming = not stream
    if change:
        setattr(agent, change, getattr(agent, change) + "-changed")
    with patch(
        "model_tools.handle_function_call", return_value="fixture result"
    ) as dispatch:
        result = agent.run_conversation("Search the fixture")

    assert result["final_response"] == ("Finished." if execute else content)
    assert dispatch.call_count == int(execute)
    assert len(requests) == 1 + int(execute)
    if not execute:
        return
    first, second = (request["messages"] for request in requests)
    assert second[: len(first)] == first  # the complete cached prefix stays byte-stable
    assert requests[0]["tools"] == requests[1]["tools"]
    assert [m["role"] for m in second[len(first) :]] == ["assistant", "tool"]
    call = second[-2]["tool_calls"][0]
    assert call["function"]["name"] == "web_search"
    assert second[-1]["tool_call_id"] == call["id"]
    assert dispatch.call_args.kwargs["tool_call_id"] == call["id"]
    assert json.loads(call["function"]["arguments"])["query"] == (
        "native" if native else "local fixture"
    )
    if native:
        assert call["id"] == "native-call"


@pytest.mark.parametrize(
    "content",
    [
        'Here is a call: {"name":"web_search","arguments":{}}',
        '```json\n{"name":"web_search","arguments":{}}\n```',
        '{"name":"web_search","arguments":{},"extra":true}',
        '{"name":"unknown_tool","arguments":{}}',
        '[{"name":"web_search","arguments":{}},{"name":"unknown_tool","arguments":{}}]',
        '{"name":"web_search","arguments":"{}"}',
        '{"name":"web_search","arguments":{"x":NaN}}',
        '{"name":"web_search","arguments":{"x":Infinity}}',
        '{"name":"web_search","arguments":{"x":1,"x":2}}',
        '{"name":"web_search","arguments":{"x":"\\ud800"}}',
        '{"name":"web_search","arguments":{"x":' + "[" * 30 + "0" + "]" * 30 + "}}",
        json.dumps({"name": "web_search", "arguments": {"x": [0] * 1001}}),
        json.dumps({"name": "web_search", "arguments": {"x": "a" * 65536}}),
        json.dumps([{"name": "web_search", "arguments": {}}] * 17),
        "[]",
    ],
    ids=[
        "prose",
        "fence",
        "extra-key",
        "unknown",
        "mixed",
        "string-args",
        "nan",
        "infinite",
        "duplicate",
        "surrogate",
        "depth",
        "nodes",
        "bytes",
        "count",
        "empty",
    ],
)
def test_rejection_is_atomic_and_valid_call_ids_are_stable(make_agent, content):
    from agent.content_tool_calls import _extract_content_tool_calls

    def unexpected_request(request):
        pytest.fail(f"parser made an HTTP request: {request.url}")

    agent = make_agent(True, unexpected_request)
    assert _extract_content_tool_calls(agent, content) is None
    calls = [{"name": "web_search", "arguments": {"query": "ok"}}] * 2
    first = _extract_content_tool_calls(agent, json.dumps(calls))
    second = _extract_content_tool_calls(agent, json.dumps(calls, indent=2))
    assert len(first) == 2
    assert first == second
    assert first[0].id != first[1].id
    agent.api_mode = "codex_responses"
    assert _extract_content_tool_calls(agent, json.dumps(calls)) is None
