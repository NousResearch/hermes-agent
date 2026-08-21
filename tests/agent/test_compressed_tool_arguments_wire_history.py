"""Wire-history regressions for compressed historical tool arguments."""

from __future__ import annotations

import copy
import json

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.transports.chat_completions import ChatCompletionsTransport


MARKER = json.dumps(
    {
        "__hermes_incomplete_tool_arguments__": {
            "arguments_omitted": True,
            "original_chars": 12345,
            "reason": "context_compression",
            "replayable": False,
            "sha256": "a" * 64,
            "version": 1,
        }
    }
)


def _mixed_history():
    return [
        {"role": "user", "content": "inspect both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_incomplete",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": MARKER},
                },
                {
                    "id": "call_complete",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_incomplete",
            "content": '{"error_type":"incomplete_historical_tool_arguments"}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_complete",
            "content": "README contents",
        },
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
    ]


def _assert_mixed_pairing_is_safe(payload):
    serialized = json.dumps(payload)
    assert "__hermes_incomplete_tool_arguments__" not in serialized
    assert "call_incomplete" not in serialized
    assert "call_complete" in serialized
    assert "README contents" in serialized
    assert "compressed historical tool call" in serialized.lower()


def test_chat_request_copy_neutralizes_completed_marker_call_and_preserves_source():
    history = _mixed_history()
    original = copy.deepcopy(history)
    transport = ChatCompletionsTransport()

    first = transport.convert_messages(history, model="gpt-5.6")
    second = transport.convert_messages(history, model="gpt-5.6")

    assert first == second  # retries are deterministic
    assert history == original  # persisted/resumed transcript remains canonical
    _assert_mixed_pairing_is_safe(first)
    complete_calls = first[1]["tool_calls"]
    assert [call["id"] for call in complete_calls] == ["call_complete"]
    assert [m.get("tool_call_id") for m in first if m.get("role") == "tool"] == [
        "call_complete"
    ]


def test_codex_response_items_neutralize_completed_marker_call_and_keep_pairing():
    history = _mixed_history()
    original = copy.deepcopy(history)

    first = _chat_messages_to_responses_input(history)
    second = _chat_messages_to_responses_input(history)

    assert first == second
    assert history == original
    _assert_mixed_pairing_is_safe(first)
    function_calls = [item for item in first if item.get("type") == "function_call"]
    outputs = [item for item in first if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in function_calls] == ["call_complete"]
    assert [item["call_id"] for item in outputs] == ["call_complete"]


def test_marker_call_without_completed_result_remains_for_fail_closed_execution_guard():
    history = _mixed_history()[:2]

    chat = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")
    codex = _chat_messages_to_responses_input(history)

    assert "__hermes_incomplete_tool_arguments__" in json.dumps(chat)
    assert "__hermes_incomplete_tool_arguments__" in json.dumps(codex)
