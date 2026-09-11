"""Live DeepSeek V4 thinking-mode tool-call replay smoke test.

Opt-in only:
    HERMES_LIVE_TESTS=1 pytest tests/run_agent/test_deepseek_v4_thinking_live.py -q

Requires DEEPSEEK_API_KEY in the process environment. The key is captured at
module import time because tests/conftest.py intentionally removes credential
environment variables before each test body runs.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest


LIVE = os.environ.get("HERMES_LIVE_TESTS") == "1"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LIVE_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
LIVE_BASE_URL = "https://api.deepseek.com"

pytestmark = [
    pytest.mark.skipif(not LIVE, reason="live-only: set HERMES_LIVE_TESTS=1"),
    pytest.mark.skipif(not DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not configured"),
]

TOOL_NAME = "lookup_ticket_status"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return the status for a test ticket id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket id to look up.",
                    },
                },
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    }
]


def _thinking_kwargs() -> dict:
    return {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _print_trace(label: str, value: Any) -> None:
    sys.__stdout__.write(f"\n--- {label} ---\n")
    sys.__stdout__.write(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
    )
    sys.__stdout__.write("\n")
    sys.__stdout__.flush()


def _message_snapshot(message) -> dict:
    return {
        "content": getattr(message, "content", None),
        "reasoning": getattr(message, "reasoning", None),
        "reasoning_content": _raw_reasoning_content(message),
        "model_extra": getattr(message, "model_extra", None),
        "tool_calls": _jsonable(getattr(message, "tool_calls", None)),
    }


def _make_live_client():
    from openai import OpenAI

    return OpenAI(api_key=DEEPSEEK_KEY, base_url=LIVE_BASE_URL)


def _make_agent_for_message_building(model: str):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider = "deepseek"
    agent.model = model
    agent.base_url = LIVE_BASE_URL
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def _raw_reasoning_content(message):
    direct = getattr(message, "reasoning_content", None)
    if direct is not None:
        return direct
    model_extra = getattr(message, "model_extra", None) or {}
    if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
        return model_extra["reasoning_content"]
    return None


@pytest.mark.parametrize("live_model", LIVE_MODELS)
def test_deepseek_v4_thinking_tool_call_replay_round_trip(live_model: str):
    """Hit DeepSeek twice and replay the assistant tool-call turn.

    The first request forces a tool call with thinking enabled. The second
    request replays that assistant message with content, reasoning_content,
    and tool_calls, then appends the tool result. DeepSeek accepting the
    second request is the live guardrail for the V4 thinking replay contract.
    """

    client = _make_live_client()
    agent = _make_agent_for_message_building(live_model)

    first_request = {
        "model": live_model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You must use the provided lookup_ticket_status tool "
                    "exactly once with ticket_id 'DS-4242'. Do not answer "
                    "directly."
                ),
            }
        ],
        "tools": TOOLS,
        "max_tokens": 1024,
        "timeout": 90,
        **_thinking_kwargs(),
    }
    _print_trace(f"{live_model} first request", first_request)
    first = client.chat.completions.create(**first_request)
    _print_trace(f"{live_model} first raw response", first)

    first_choice = first.choices[0]
    first_message = first_choice.message
    _print_trace(
        f"{live_model} first assistant message",
        {
            "finish_reason": first_choice.finish_reason,
            **_message_snapshot(first_message),
        },
    )
    assert first_message.tool_calls, "DeepSeek did not return a tool call"
    first_tool_call = first_message.tool_calls[0]
    assert first_tool_call.function.name == TOOL_NAME
    assert isinstance(json.loads(first_tool_call.function.arguments or "{}"), dict)

    raw_reasoning_content = _raw_reasoning_content(first_message)
    assert raw_reasoning_content is not None, (
        "DeepSeek did not return reasoning_content; the thinking payload may "
        "not have been honored"
    )

    stored_assistant = agent._build_assistant_message(
        first_message,
        first_choice.finish_reason or "tool_calls",
    )
    _print_trace(f"{live_model} stored assistant message", stored_assistant)
    assert stored_assistant["reasoning_content"] == raw_reasoning_content

    replay_assistant = {
        "role": "assistant",
        "content": stored_assistant.get("content") or "",
        "tool_calls": stored_assistant["tool_calls"],
    }
    agent._copy_reasoning_content_for_api(stored_assistant, replay_assistant)
    _print_trace(f"{live_model} replay assistant message", replay_assistant)

    tool_call_id = stored_assistant["tool_calls"][0]["id"]
    messages = [
        {
            "role": "user",
            "content": (
                "You must use the provided lookup_ticket_status tool "
                "exactly once with ticket_id 'DS-4242'. Do not answer "
                "directly."
            ),
        },
        replay_assistant,
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                {"ticket_id": "DS-4242", "status": "green", "source": "live-test"},
                separators=(",", ":"),
            ),
        },
    ]

    from agent.transports.chat_completions import ChatCompletionsTransport

    api_messages = ChatCompletionsTransport().convert_messages(messages)
    _print_trace(
        f"{live_model} second request messages after transport conversion",
        api_messages,
    )
    assert api_messages[1]["reasoning_content"] == raw_reasoning_content
    assert "call_id" not in api_messages[1]["tool_calls"][0]
    assert "response_item_id" not in api_messages[1]["tool_calls"][0]

    second_request = {
        "model": live_model,
        "messages": api_messages,
        "max_tokens": 1024,
        "timeout": 90,
        **_thinking_kwargs(),
    }
    _print_trace(f"{live_model} second request", second_request)
    second = client.chat.completions.create(**second_request)
    _print_trace(f"{live_model} second raw response", second)
    _print_trace(
        f"{live_model} second assistant message",
        {
            "finish_reason": second.choices[0].finish_reason,
            **_message_snapshot(second.choices[0].message),
        },
    )

    second_message = second.choices[0].message
    final_content = second_message.content or ""
    final_reasoning = _raw_reasoning_content(second_message) or ""
    assert second.choices[0].finish_reason == "stop"
    assert final_content.strip() or final_reasoning.strip(), (
        "DeepSeek returned neither visible content nor reasoning_content"
    )
