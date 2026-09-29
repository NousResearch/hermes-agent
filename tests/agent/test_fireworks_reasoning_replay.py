"""Fireworks GLM reasoning must survive tool results and resumed history."""
from copy import deepcopy
import json

import httpx
from openai import OpenAI
import pytest

from agent.message_sanitization import stale_thinking_reaches_wire
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.turn_context import build_api_messages
from providers import get_provider_profile
from run_agent import AIAgent


MODEL = "accounts/fireworks/models/glm-5p3-flash"
BASE_URL = "https://api.fireworks.ai/inference/v1"


def _agent(provider, base_url=BASE_URL):
    agent = object.__new__(AIAgent)
    agent.provider, agent.model, agent.base_url = provider, MODEL, base_url
    agent.verbose_logging = False
    agent.api_mode = "chat_completions"
    agent._current_turn_timestamp = 0
    agent.ephemeral_system_prompt = ""
    agent.reasoning_callback = agent.stream_delta_callback = agent._stream_callback = None
    return agent


@pytest.mark.parametrize("provider", ["fireworks", "fireworks-ai", "fw", "custom"])
@pytest.mark.parametrize("effort", ["medium", "none"])
def test_glm_tool_round_trip_keeps_reasoning_on_wire(provider, effort):
    agent = _agent(provider)
    transport = ChatCompletionsTransport()
    captured = []
    thought = "I need the tool result before finishing the calculation."

    def serve(request):
        captured.append(json.loads(request.content))
        message = {"role": "assistant", "content": "42"}
        finish = "stop"
        if len(captured) == 1:
            message.update(content=None, reasoning_content=thought, tool_calls=[{
                "id": "call_calc", "type": "function",
                "function": {"name": "calculator", "arguments": '{"a":20,"b":22}'},
            }])
            finish = "tool_calls"
        return httpx.Response(200, json={"id": "completion", "object": "chat.completion",
            "created": 0, "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

    history = [{"role": "user", "content": "Calculate 20 + 22 using the tool."}]
    profile = get_provider_profile(provider)
    assert profile is not None
    config = {"enabled": effort != "none", "effort": effort}
    with OpenAI(api_key="test-only", base_url=BASE_URL,
                http_client=httpx.Client(transport=httpx.MockTransport(serve))) as client:
        first = client.chat.completions.create(model=MODEL, messages=history)
        stored = agent._build_assistant_message(first.choices[0].message, "tool_calls")
        stored = json.loads(json.dumps(stored))
        unchanged = deepcopy(stored)
        history += [stored, {"role": "tool", "tool_call_id": "call_calc", "content": "42"}]
        replay, _ = build_api_messages(agent, history, current_turn_user_idx=0,
            ext_prefetch_cache=None, plugin_user_context=None, moa_config=None, active_system_prompt="")
        kwargs = transport.build_kwargs(model=MODEL, messages=replay, provider_profile=profile,
            base_url=BASE_URL, reasoning_config=config, supports_reasoning=True)
        client.chat.completions.create(**kwargs)

    sent = captured[1]
    assert sent["messages"][1]["reasoning_content"] == thought
    assert sent["messages"][1]["tool_calls"][0]["id"] == sent["messages"][2]["tool_call_id"]
    assert "reasoning" not in sent["messages"][1]
    assert sent["reasoning_effort"] == effort
    assert "reasoning" not in sent and "thinking" not in sent
    assert stored == unchanged


def test_fireworks_replay_and_context_accounting_follow_active_route():
    source = {"role": "assistant", "content": None, "reasoning_content": "Use the result.",
              "tool_calls": [{"id": "call_calc", "type": "function",
                              "function": {"name": "calculator", "arguments": "{}"}}]}
    agent = _agent("custom")
    original = deepcopy(source)
    for provider, url, expected in [
        ("custom", BASE_URL, True),
        ("groq", "https://api.groq.com/openai/v1", False),
        ("custom", "https://api.fireworks.ai.example.com/v1", False),
        ("custom", "https://example.com/api.fireworks.ai/v1", False),
        ("fireworks", BASE_URL, True),
    ]:
        agent.provider, agent.base_url = provider, url
        replay = [dict(source)]
        agent._reapply_reasoning_echo_for_provider(replay)
        assert (replay[0].get("reasoning_content") == source["reasoning_content"]) is expected
        assert stale_thinking_reaches_wire("chat_completions", provider, MODEL, url) is expected
        assert agent._reapply_reasoning_echo_for_provider(replay) == 0
    assert source == original
