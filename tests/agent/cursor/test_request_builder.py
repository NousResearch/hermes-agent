"""Tests for Cursor request builder behavior."""

from __future__ import annotations

import json

import pytest

from agent.cursor.request_builder import (
    build_conversation_turns,
    build_cursor_system_prompt_jsons,
    build_grpc_request,
    build_root_prompt_messages_json,
    create_blob_id,
    store_cursor_blob,
)
from agent.cursor.proto import agent_pb2


def _text_message(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def _tool_result(text: str) -> dict:
    return {
        "role": "tool",
        "content": text,
    }


def _decode_json_blobs(blob_ids: list[bytes], blob_store: dict[str, bytes]) -> list[dict]:
    return [json.loads(blob_store[blob_id.hex()].decode("utf-8")) for blob_id in blob_ids]


def _decode_turns(turn_blob_ids: list[bytes], blob_store: dict[str, bytes]) -> list[dict]:
    decoded = []
    for turn_blob_id in turn_blob_ids:
        turn = agent_pb2.ConversationTurnStructure()
        turn.ParseFromString(blob_store[turn_blob_id.hex()])

        user_message = agent_pb2.UserMessage()
        user_message.ParseFromString(blob_store[turn.agent_conversation_turn.user_message.hex()])

        steps = []
        for step_blob_id in turn.agent_conversation_turn.steps:
            step = agent_pb2.ConversationStep()
            step.ParseFromString(blob_store[step_blob_id.hex()])
            steps.append(step)

        decoded.append({"user_message": user_message, "steps": steps})
    return decoded


class TestBuildCursorSystemPromptJsons:
    def test_empty_uses_default_system_prompt(self):
        assert build_cursor_system_prompt_jsons([]) == [
            json.dumps({"role": "system", "content": "You are a helpful assistant."})
        ]

    def test_multi_preserves_order(self):
        assert build_cursor_system_prompt_jsons(["one", "two"]) == [
            json.dumps({"role": "system", "content": "one"}),
            json.dumps({"role": "system", "content": "two"}),
        ]


def test_single_turn_request_has_user_message_action():
    messages = [_text_message("user", "hello there")]

    request_bytes, conversation_state, blob_store = build_grpc_request(
        messages=messages,
        system_prompt=["system prompt"],
        tools=[],
        model_id="gpt-5.4",
        conversation_id="conv-1",
        cached_conversation_state=None,
    )

    client_message = agent_pb2.AgentClientMessage()
    client_message.ParseFromString(request_bytes)

    assert client_message.WhichOneof("message") == "run_request"
    run_request = client_message.run_request
    assert run_request.action.WhichOneof("action") == "user_message_action"
    assert run_request.action.user_message_action.user_message.text == "hello there"
    assert run_request.conversation_id == "conv-1"
    assert conversation_state.root_prompt_messages_json
    assert blob_store


def test_multi_turn_history_includes_prior_user_in_root_prompt_messages_json():
    blob_store: dict[str, bytes] = {}
    system_prompt_blob_ids = [
        store_cursor_blob(blob_store, item.encode("utf-8"))
        for item in build_cursor_system_prompt_jsons(["system prompt"])
    ]
    messages = [
        _text_message("user", "first user turn"),
        _text_message("assistant", "assistant reply"),
        _text_message("user", "current user turn"),
    ]

    root_blob_ids = build_root_prompt_messages_json(
        messages=messages,
        system_prompt_ids=system_prompt_blob_ids,
        blob_store=blob_store,
    )
    decoded = _decode_json_blobs(root_blob_ids, blob_store)

    assert decoded[0] == {"role": "system", "content": "system prompt"}
    assert decoded[1] == {
        "role": "user",
        "content": [{"type": "text", "text": "first user turn"}],
    }
    assert decoded[2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "assistant reply"}],
    }
    assert all(
        entry.get("content") != [{"type": "text", "text": "current user turn"}]
        for entry in decoded
    )


def test_build_conversation_turns_groups_prior_history_only():
    blob_store: dict[str, bytes] = {}
    messages = [
        _text_message("user", "first user turn"),
        _text_message("assistant", "assistant reply"),
        _tool_result("tool output"),
        _text_message("user", "current user turn"),
    ]

    turn_blob_ids = build_conversation_turns(messages=messages, blob_store=blob_store)
    decoded = _decode_turns(turn_blob_ids, blob_store)

    assert len(decoded) == 1
    assert decoded[0]["user_message"].text == "first user turn"
    assert [step.assistant_message.text for step in decoded[0]["steps"]] == [
        "assistant reply",
        "[Tool Result]\ntool output",
    ]


def test_cached_conversation_state_is_cloned_for_multi_turn_requests():
    first_messages = [
        _text_message("user", "first user turn"),
        _text_message("assistant", "assistant reply"),
    ]
    _, cached_state, first_blob_store = build_grpc_request(
        messages=first_messages,
        system_prompt=["system prompt"],
        tools=[],
        model_id="gpt-5.4",
        conversation_id="conv-1",
        cached_conversation_state=None,
    )
    original_root_prompt_messages = list(cached_state.root_prompt_messages_json)
    original_turns = list(cached_state.turns)

    second_messages = [
        _text_message("user", "first user turn"),
        _text_message("assistant", "assistant reply"),
        _text_message("user", "follow-up question"),
    ]
    _, next_state, second_blob_store = build_grpc_request(
        messages=second_messages,
        system_prompt=["system prompt"],
        tools=[],
        model_id="gpt-5.4",
        conversation_id="conv-1",
        cached_conversation_state=cached_state,
    )

    assert cached_state is not next_state
    assert list(cached_state.root_prompt_messages_json) == original_root_prompt_messages
    assert list(cached_state.turns) == original_turns

    decoded_root = _decode_json_blobs(list(next_state.root_prompt_messages_json), second_blob_store)
    assert decoded_root[0] == {"role": "system", "content": "system prompt"}
    assert decoded_root[1] == {
        "role": "user",
        "content": [{"type": "text", "text": "first user turn"}],
    }
    assert decoded_root[2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "assistant reply"}],
    }

    decoded_turns = _decode_turns(list(next_state.turns), second_blob_store)
    assert len(decoded_turns) == 1
    assert decoded_turns[0]["user_message"].text == "first user turn"
    assert [step.assistant_message.text for step in decoded_turns[0]["steps"]] == [
        "assistant reply",
    ]

    # The original blob store remains intact; repeated blobs may be deterministic.
    assert first_blob_store is not second_blob_store


def test_blob_ids_are_deterministic_for_same_content():
    data = b"same content"
    first = create_blob_id(data)
    second = create_blob_id(data)

    assert first == second
    assert len(first) == 32


def test_encode_tool_input_schema_uses_protobuf_value_bytes():
    from google.protobuf import json_format
    from google.protobuf import struct_pb2

    from agent.cursor.request_builder import encode_tool_input_schema

    schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    encoded = encode_tool_input_schema(schema)
    value = struct_pb2.Value()
    value.ParseFromString(encoded)
    round_trip = json_format.MessageToDict(value)
    assert round_trip["type"] == "object"
    assert round_trip["properties"]["command"]["type"] == "string"
    assert round_trip["required"] == ["command"]

