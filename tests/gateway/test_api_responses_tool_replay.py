"""Responses clients replay the previous turn's tool items as the next ``input``.

Codex (and the Responses SDKs) echo a prior response's ``output`` list back as ``input``.
Those items carry ``call_id``/``output`` instead of ``content``. Parsing them as
``role="user"`` produced an empty user turn, so every request that followed a tool call
failed with ``No user message found in input``.
"""
from gateway.platforms.api_server import _content_has_visible_payload
from gateway.platforms.api_server_openai_routes import (
    _is_non_message_input_item, _parse_responses_input)
from gateway.platforms.api_server_response_admissions import _last_user_turn_index


TOOL_REPLAY_INPUT = [
    {"type": "message", "role": "developer",
     "content": [{"type": "input_text", "text": "rules"}]},
    {"type": "message", "role": "user",
     "content": [{"type": "input_text", "text": "列出 /tmp"}]},
    {"type": "message", "role": "assistant",
     "content": [{"type": "output_text", "text": "我看一下"}]},
    {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
]


def test_tool_items_are_not_parsed_as_messages():
    messages, error = _parse_responses_input(TOOL_REPLAY_INPUT)

    assert error is None
    assert [message["role"] for message in messages] == ["developer", "user", "assistant"]
    assert all(_content_has_visible_payload(message["content"]) for message in messages)


def test_item_reference_is_skipped():
    messages, error = _parse_responses_input([
        {"type": "item_reference", "id": "msg_1"},
        {"role": "user", "content": "hello"},
    ])

    assert error is None
    assert messages == [{"role": "user", "content": "hello"}]


def test_unknown_item_without_role_or_content_is_skipped():
    # Forward compatibility: an unfamiliar item can never be a chat message.
    assert _is_non_message_input_item({"type": "some_future_item", "id": "x"}) is True
    assert _is_non_message_input_item({"role": "user", "content": "hi"}) is False


def test_turn_after_a_tool_call_still_finds_its_user_message():
    messages, _ = _parse_responses_input(TOOL_REPLAY_INPUT)

    index = _last_user_turn_index(messages)

    assert index == 1
    assert _content_has_visible_payload(messages[index]["content"])


def test_trailing_assistant_message_does_not_become_the_turn():
    # The client replayed the assistant message it already received; the turn is still
    # the newest user message, not that assistant text.
    messages, _ = _parse_responses_input(TOOL_REPLAY_INPUT)

    assert messages[-1]["role"] == "assistant"
    assert _last_user_turn_index(messages) == 1


def test_a_request_without_any_user_message_is_still_rejected():
    messages, _ = _parse_responses_input([
        {"type": "function_call_output", "call_id": "c", "output": "x"},
    ])

    assert messages == []
    assert _last_user_turn_index(messages) is None


def test_an_empty_user_message_is_still_rejected():
    messages, _ = _parse_responses_input([{"role": "user", "content": "   "}])

    assert _last_user_turn_index(messages) is None
