"""Tests for assistant-response validation control helpers."""

import json

from hermes_cli.middleware import AssistantResponseMiddlewareResult

from agent.assistant_response_control import (
    build_validator_tool_response,
    make_validator_retry_messages,
    sanitize_validator_text,
    should_disable_streaming_for_turn,
)


def test_stream_policy_can_disable_streaming_for_one_turn():
    assert should_disable_streaming_for_turn({"stream_policy": "buffer_until_validated"}) is True
    assert should_disable_streaming_for_turn({"disable_streaming_for_turn": True}) is True
    assert should_disable_streaming_for_turn({}) is False


def test_build_validator_tool_response_preserves_provenance_and_json_args():
    decision = AssistantResponseMiddlewareResult(
        action="require_tool",
        feedback="Read README before reversing.",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "README.md"},
                "reason": "verify source file",
                "read_only": True,
            }
        ],
    )

    response = build_validator_tool_response(
        decision,
        valid_tool_names={"read_file"},
        validation_attempt=0,
    )

    assert response.content.startswith("Judgment Integrity validator requested")
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "validator_read_file_0_0"
    assert call.name == "read_file"
    assert json.loads(call.arguments) == {"path": "README.md"}
    assert call.provider_data["validator_requested"] is True
    assert call.provider_data["reason"] == "verify source file"


def test_build_validator_tool_response_rejects_missing_or_unknown_tools():
    missing = AssistantResponseMiddlewareResult(action="require_tool", tool_calls=[])
    assert build_validator_tool_response(missing, valid_tool_names={"read_file"}, validation_attempt=0) is None

    unknown = AssistantResponseMiddlewareResult(
        action="require_tool",
        tool_calls=[{"name": "rm_everything", "args": {}, "read_only": True}],
    )
    assert build_validator_tool_response(unknown, valid_tool_names={"read_file"}, validation_attempt=0) is None


def test_validator_retry_messages_preserve_role_alternation_and_mark_synthetic():
    messages = make_validator_retry_messages(
        draft="맞습니다. 제가 틀렸습니다. sk-secretsecretsecret",
        feedback="Do not reverse without evidence.",
        validation_attempt=0,
    )

    assert [m["role"] for m in messages] == ["assistant", "user"]
    assert messages[0]["_response_validation_synthetic"] is True
    assert messages[1]["_response_validation_synthetic"] is True
    assert "제가 틀렸습니다" not in messages[0]["content"]
    assert "sk-secretsecretsecret" not in messages[0]["content"]
    assert "Do not reverse without evidence." in messages[1]["content"]


def test_sanitize_validator_text_redacts_and_strips_reasoning():
    text = "<think>private reasoning</think>Use sk-secretsecretsecret safely\ud800"

    sanitized = sanitize_validator_text(
        text,
        strip_think_blocks=lambda value: value.replace("<think>private reasoning</think>", ""),
    )

    assert "private reasoning" not in sanitized
    assert "sk-secretsecretsecret" not in sanitized
    assert "\ud800" not in sanitized
    assert "Use" in sanitized
