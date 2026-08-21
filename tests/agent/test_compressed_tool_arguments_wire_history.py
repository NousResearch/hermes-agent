"""Wire-history regressions for compressed historical tool arguments."""

from __future__ import annotations

import copy
import json

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.anthropic import AnthropicTransport


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


def _all_compressed_history(content="I will inspect it.", call_count=1):
    calls = [
        {
            "id": f"call_incomplete_{index}",
            "type": "function",
            "function": {"name": "terminal", "arguments": MARKER},
        }
        for index in range(call_count)
    ]
    results = [
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "completed",
        }
        for call in calls
    ]
    return [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": calls,
        },
        *results,
        {"role": "assistant", "content": "Inspection finished."},
        {"role": "user", "content": "continue"},
    ]


def test_all_compressed_chat_history_preserves_role_sequence_and_visible_text():
    history = _all_compressed_history()
    original = copy.deepcopy(history)
    converted = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")

    assert history == original
    assert "__hermes_incomplete_tool_arguments__" not in json.dumps(converted)
    assert all(
        left.get("role") != "assistant" or right.get("role") != "assistant"
        for left, right in zip(converted, converted[1:])
    )
    assistant_text = "\n".join(
        str(message.get("content", ""))
        for message in converted
        if message.get("role") == "assistant"
    )
    assert "I will inspect it." in assistant_text
    assert "Inspection finished." in assistant_text


def test_all_compressed_anthropic_and_bedrock_history_preserves_role_sequence():
    _system, converted = AnthropicTransport().convert_messages(
        _all_compressed_history()
    )
    assert "__hermes_incomplete_tool_arguments__" not in json.dumps(converted)
    assert all(
        left.get("role") != right.get("role")
        for left, right in zip(converted, converted[1:])
    )


def test_all_compressed_codex_history_has_no_callable_marker():
    converted = _chat_messages_to_responses_input(_all_compressed_history())
    serialized = json.dumps(converted)
    assert "__hermes_incomplete_tool_arguments__" not in serialized
    assert "I will inspect it." in serialized
    assert "Inspection finished." in serialized


def test_all_compressed_multi_call_history_is_provider_valid():
    history = _all_compressed_history(call_count=2)
    chat = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")
    _system, anthropic_bedrock = AnthropicTransport().convert_messages(history)
    codex = _chat_messages_to_responses_input(history)

    for converted in (chat, anthropic_bedrock, codex):
        assert "__hermes_incomplete_tool_arguments__" not in json.dumps(converted)
    assert all(
        left.get("role") != "assistant" or right.get("role") != "assistant"
        for left, right in zip(chat, chat[1:])
    )
    assert all(
        left.get("role") != right.get("role")
        for left, right in zip(anthropic_bedrock, anthropic_bedrock[1:])
    )
