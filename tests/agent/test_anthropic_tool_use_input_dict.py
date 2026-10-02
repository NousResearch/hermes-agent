"""Regression: Anthropic ``tool_use.input`` must always be a JSON object on the wire.

A weak model can emit tool-call arguments that are *valid JSON but not an object*
(``"code"``, ``[1,2]``, ``42``). The executor correctly rejects those ("Tool
arguments must be a valid JSON object; tool was not executed."), but the stored
assistant ``tool_calls`` keep the malformed arguments — and the Anthropic-format
replay passed them through verbatim as ``tool_use.input``. Object-strict
Anthropic-compatible endpoints (e.g. MiniMax ``/anthropic``) then 400 the whole
request with ``tool_use.input: Input should be a valid dictionary (2013)`` on
every turn, wedging the session.
"""
from agent.anthropic_message_convert import _tool_use_block, convert_messages_to_anthropic

_TOOL_ID = "call_function_yyyck3t8jjyr_1"


def _messages_with_args(arguments):
    return [
        {"role": "user", "content": "pull github files"},
        {"role": "assistant", "content": "rate limited, retrying", "tool_calls": [
            {"id": _TOOL_ID, "type": "function",
             "function": {"name": "execute_code", "arguments": arguments}}
        ]},
        {"role": "tool", "tool_call_id": _TOOL_ID,
         "content": '{"error": "Invalid tool arguments"}'},
    ]


def _tool_use_inputs(messages):
    for m in messages:
        for b in m.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                yield b


def test_scalar_json_arguments_never_reach_the_wire():
    # The production wedge: arguments emitted as a bare JSON string scalar.
    _, result = convert_messages_to_anthropic(_messages_with_args('"code"'))
    blocks = list(_tool_use_inputs(result))
    assert blocks and all(isinstance(b["input"], dict) for b in blocks)


def test_list_and_number_arguments_are_coerced():
    for raw in ("[1, 2]", "42"):
        _, result = convert_messages_to_anthropic(_messages_with_args(raw))
        assert all(isinstance(b["input"], dict) for b in _tool_use_inputs(result))


def test_none_and_non_string_arguments_are_coerced():
    for raw in (None, ["not", "json"], 7):
        _, result = convert_messages_to_anthropic(_messages_with_args(raw))
        assert all(isinstance(b["input"], dict) for b in _tool_use_inputs(result))


def test_real_object_arguments_are_untouched():
    _, result = convert_messages_to_anthropic(
        _messages_with_args('{"code": "print(1)", "lang": "py"}')
    )
    inputs = [b["input"] for b in _tool_use_inputs(result)]
    assert inputs == [{"code": "print(1)", "lang": "py"}]


def test_stored_replay_block_with_scalar_input_is_coerced():
    # anthropic_content_blocks replay: the stored block itself carries a non-dict
    # input (captured verbatim from a malformed stream).
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "anthropic_content_blocks": [
            {"type": "text", "text": "x"},
            {"type": "tool_use", "id": _TOOL_ID, "name": "execute_code", "input": "code"},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": _TOOL_ID, "content": "err"}]},
    ]
    _, result = convert_messages_to_anthropic(messages)
    blocks = list(_tool_use_inputs(result))
    assert blocks and all(isinstance(b["input"], dict) for b in blocks)


def test_tool_use_block_helper_coerces_directly():
    assert _tool_use_block("id-1", "t", "code")["input"] == {}
    assert _tool_use_block("id-1", "t", None)["input"] == {}
    assert _tool_use_block("id-1", "t", '{"a": 1}')["input"] == {"a": 1}
    assert _tool_use_block("id-1", "t", {"a": 1})["input"] == {"a": 1}
