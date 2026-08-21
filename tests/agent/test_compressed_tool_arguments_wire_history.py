"""Wire-history regressions for compressed historical tool arguments."""

from __future__ import annotations

import copy
import json

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.anthropic import AnthropicTransport
from agent.transports.bedrock import BedrockTransport
from agent.tool_argument_integrity import neutralize_completed_incomplete_tool_calls


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
            "anthropic_content_blocks": [{"type": "text", "text": content}],
            "codex_message_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            ],
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


def test_malformed_non_dict_history_is_dropped_from_every_provider_request():
    history = _all_compressed_history()[:3] + [None]

    chat = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")
    _system, anthropic = AnthropicTransport().convert_messages(history)
    bedrock = BedrockTransport().build_kwargs(model="anthropic.claude", messages=history)

    assert None not in chat
    assert all(isinstance(message, dict) for message in anthropic)
    assert all(isinstance(message, dict) for message in bedrock["messages"])


def test_affected_turn_discards_malformed_and_stale_anthropic_sidecars():
    history = _all_compressed_history()
    affected = history[1]
    affected["anthropic_content_blocks"] = [
        {
            "type": "tool_use",
            "id": [],
            "name": "terminal",
            "input": {"command": "must never survive"},
        },
        {
            "type": "tool_use",
            "id": "stale-unmatched",
            "name": "terminal",
            "input": {"command": "also must never survive"},
        },
        {"type": "thinking", "thinking": "signed after tool", "signature": "sig"},
    ]

    projected = neutralize_completed_incomplete_tool_calls(history)
    _system, anthropic = AnthropicTransport().convert_messages(history)

    assert "anthropic_content_blocks" not in projected[1]
    serialized = json.dumps(anthropic)
    assert "must never survive" not in serialized
    assert "also must never survive" not in serialized
    assert "signed after tool" not in serialized
    assert '"signature": "sig"' not in serialized


def test_affected_turn_discards_all_provider_native_sidecars():
    history = _all_compressed_history()
    affected = history[1]
    affected["reasoning_details"] = [
        {"type": "thinking", "thinking": "provider native", "signature": "sig"}
    ]
    affected["codex_reasoning_items"] = [{"type": "reasoning", "id": "r1"}]
    affected["codex_message_items"] = [
        {"type": "message", "role": "assistant", "content": "provider native"}
    ]

    projected = neutralize_completed_incomplete_tool_calls(history)

    for key in (
        "anthropic_content_blocks",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
    ):
        assert key not in projected[1]


def test_direct_bedrock_build_kwargs_neutralizes_completed_marker_call():
    history = _mixed_history()
    original = copy.deepcopy(history)

    kwargs = BedrockTransport().build_kwargs(
        model="anthropic.claude-3-5-sonnet", messages=history
    )

    assert history == original
    serialized = json.dumps(kwargs)
    assert "__hermes_incomplete_tool_arguments__" not in serialized
    assert "call_incomplete" not in serialized
    assert "call_complete" in serialized
    assert "README contents" in serialized
