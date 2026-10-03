"""Regression: a malformed ``tool_use.input`` stored in history must not kill the next request.

The executor's argument validator rejects a ``tool_use`` whose ``input`` is not a
JSON object (the model occasionally wraps params as a one-element array when
copying a prior call) and answers with an error ``tool_result`` — but the
malformed assistant message is already appended to history. Replayed verbatim,
Anthropic rejects the whole request with HTTP 400 "Input should be a valid
dictionary", a non-retryable error that ends the cron/agent loop.

Three storage shapes reach the wire without ever seeing the executor's
validator: native ``anthropic_content_blocks`` replay, a pre-shaped assistant
content block list, and OpenAI-style ``tool_calls`` whose ``arguments`` were
already parsed into a list (``_parse_tool_args`` only coerces bad JSON
*strings*). ``_coerce_non_dict_tool_use_inputs`` is the request-build backstop
for all three, mirroring ``_parse_tool_args``' bad-JSON -> {}.
Ref #118240.
"""
from agent.anthropic_message_convert import convert_messages_to_anthropic


MALFORMED_INPUT = [{"goal": "research improvements"}]  # params wrapped as a one-element array


def _tool_use_blocks(messages):
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    yield b


def test_native_block_replay_coerces_list_input():
    # The issue's exact shape: the assistant turn was stored as native Anthropic
    # blocks; replay rebuilds tool_use with the stored (list) input verbatim.
    messages = [
        {"role": "user", "content": "run the weekly scan"},
        {"role": "assistant", "content": [], "tool_calls": [], "anthropic_content_blocks": [
            {"type": "text", "text": "starting the scan"},
            {"type": "tool_use", "id": "call_1", "name": "delegate_task", "input": MALFORMED_INPUT},
        ]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": '{"error": "Invalid tool arguments", "message": "Tool arguments must be a valid JSON object"}'},
    ]
    _, result = convert_messages_to_anthropic(messages)
    blocks = [b for b in _tool_use_blocks(result) if b["id"] == "call_1"]
    assert blocks, "tool_use block was dropped instead of repaired"
    assert blocks[0]["input"] == {}


def test_pre_shaped_content_block_list_coerces_list_input():
    # A content list that already carries a tool_use block (e.g. imported or
    # compressed history) is passed through via dict(part), never re-validated.
    messages = [
        {"role": "user", "content": "run the weekly scan"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "starting the scan"},
            {"type": "tool_use", "id": "call_2", "name": "delegate_task", "input": MALFORMED_INPUT},
        ]},
        {"role": "tool", "tool_call_id": "call_2", "content": "Invalid tool arguments"},
    ]
    _, result = convert_messages_to_anthropic(messages)
    blocks = [b for b in _tool_use_blocks(result) if b["id"] == "call_2"]
    assert blocks, "tool_use block was dropped instead of repaired"
    assert blocks[0]["input"] == {}


def test_tool_call_arguments_already_a_list_is_coerced():
    # _parse_tool_args passes non-strings through untouched, so arguments that
    # arrived pre-parsed as a list flow into the block as-is.
    messages = [
        {"role": "user", "content": "run the weekly scan"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_3", "type": "function",
             "function": {"name": "delegate_task", "arguments": MALFORMED_INPUT}},
        ]},
        {"role": "tool", "tool_call_id": "call_3", "content": "Invalid tool arguments"},
    ]
    _, result = convert_messages_to_anthropic(messages)
    blocks = [b for b in _tool_use_blocks(result) if b["id"] == "call_3"]
    assert blocks, "tool_use block was dropped instead of repaired"
    assert blocks[0]["input"] == {}


def test_none_input_is_coerced():
    messages = [
        {"role": "user", "content": "run the weekly scan"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "starting"},
            {"type": "tool_use", "id": "call_4", "name": "delegate_task", "input": None},
        ]},
        {"role": "tool", "tool_call_id": "call_4", "content": "Invalid tool arguments"},
    ]
    _, result = convert_messages_to_anthropic(messages)
    blocks = [b for b in _tool_use_blocks(result) if b["id"] == "call_4"]
    assert blocks and blocks[0]["input"] == {}


def test_valid_dict_input_is_left_untouched():
    messages = [
        {"role": "user", "content": "run the weekly scan"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_5", "type": "function",
             "function": {"name": "web_search", "arguments": {"query": "signals"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_5", "content": "found 3 results"},
    ]
    _, result = convert_messages_to_anthropic(messages)
    blocks = [b for b in _tool_use_blocks(result) if b["id"] == "call_5"]
    assert blocks and blocks[0]["input"] == {"query": "signals"}
