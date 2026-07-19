"""Regression tests for Bedrock Converse whitespace-only text-block rejection.

Bedrock's Converse API rejects the ENTIRE request with
    ValidationException: messages: text content blocks must contain
    non-whitespace text
if any text content block anywhere in the payload is empty OR whitespace-only.

An earlier fix (issue #9486) substituted a single space " " for empty text, but
Bedrock later tightened the rule from "non-empty" to "non-whitespace", so the
space placeholder became the bug. These tests pin the invariant across EVERY
conversion path: string content, list content, tool results, assistant
tool-only turns, system prompts, and the first/last-user boundary inserts.

Invariant asserted: after convert_messages_to_converse(), NO text block in the
system prompt or any message (including nested toolResult content) is empty or
whitespace-only.
"""
from agent.bedrock_adapter import (
    convert_messages_to_converse,
    _convert_content_to_converse,
    _sanitize_text_blocks,
    _EMPTY_TEXT_PLACEHOLDER,
)


def _iter_text_blocks(system_blocks, converse_msgs):
    """Yield every text string that will be sent to Bedrock."""
    for b in (system_blocks or []):
        if "text" in b:
            yield b["text"]
    for m in converse_msgs:
        for b in m.get("content", []):
            if "text" in b:
                yield b["text"]
            elif "toolResult" in b:
                for inner in b["toolResult"].get("content", []):
                    if "text" in inner:
                        yield inner["text"]


def _assert_no_whitespace_only(system_blocks, converse_msgs):
    for text in _iter_text_blocks(system_blocks, converse_msgs):
        assert isinstance(text, str) and text.strip(), (
            f"whitespace-only/empty text block would be sent to Bedrock: {text!r}"
        )


def test_placeholder_is_non_whitespace():
    assert _EMPTY_TEXT_PLACEHOLDER.strip(), "sentinel must be non-whitespace"


def test_whitespace_only_string_content():
    """A user message whose content is only whitespace must be sanitized."""
    msgs = [{"role": "user", "content": "   \n\t  "}]
    system, converse = convert_messages_to_converse(msgs)
    _assert_no_whitespace_only(system, converse)


def test_whitespace_only_text_part_in_list():
    """The list-branch bug: a whitespace-only text part was passed through."""
    blocks = _convert_content_to_converse([{"type": "text", "text": "\n\n"}])
    assert blocks[0]["text"].strip(), "list-branch whitespace text not sanitized"


def test_none_text_part_in_list_does_not_crash():
    """A malformed part ``{"type": "text", "text": None}`` must NOT raise.

    ``dict.get("text", "")`` returns ``None`` (not the default) when the key is
    present with a ``None`` value, so an unguarded ``.strip()`` would throw
    AttributeError and kill the whole conversion. It must instead collapse to
    the non-whitespace sentinel — matching main's tolerance of falsy text.
    """
    blocks = _convert_content_to_converse([{"type": "text", "text": None}])
    assert len(blocks) == 1
    assert blocks[0]["text"].strip(), "None text part not sanitized"


def test_non_string_text_part_in_list_does_not_crash():
    """Any non-string ``text`` value (e.g. an int) collapses to the sentinel
    rather than crashing on ``.strip()``."""
    blocks = _convert_content_to_converse([{"type": "text", "text": 123}])
    assert len(blocks) == 1
    assert blocks[0]["text"].strip(), "non-string text part not sanitized"


def test_empty_tool_result():
    """A tool that produced no output (empty stdout) must still be valid."""
    msgs = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "sh", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "   "},
    ]
    system, converse = convert_messages_to_converse(msgs)
    _assert_no_whitespace_only(system, converse)


def test_assistant_tool_only_turn():
    """Assistant turn with no text (only tool calls) → no whitespace block."""
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "   ", "tool_calls": [
            {"id": "t1", "function": {"name": "sh", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]
    system, converse = convert_messages_to_converse(msgs)
    _assert_no_whitespace_only(system, converse)


def test_all_empty_content_array():
    """A content array of only whitespace text parts collapses safely."""
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": ""},
        {"type": "text", "text": "  "},
    ]}]
    system, converse = convert_messages_to_converse(msgs)
    _assert_no_whitespace_only(system, converse)


def test_boundary_user_insert_is_non_whitespace():
    """The synthetic first/last user inserts must not be a bare space."""
    # Assistant-first sequence forces a leading synthetic user message.
    msgs = [{"role": "assistant", "content": "hi"}]
    system, converse = convert_messages_to_converse(msgs)
    _assert_no_whitespace_only(system, converse)
    assert converse[0]["role"] == "user"


def test_sanitize_recurses_into_tool_result():
    """The defensive sweep reaches nested toolResult content."""
    blocks = [{"toolResult": {"toolUseId": "x", "content": [{"text": "  "}]}}]
    _sanitize_text_blocks(blocks)
    assert blocks[0]["toolResult"]["content"][0]["text"].strip()


def test_normal_content_untouched():
    """Non-empty text must pass through byte-for-byte."""
    msgs = [{"role": "user", "content": "hello world"}]
    system, converse = convert_messages_to_converse(msgs)
    texts = list(_iter_text_blocks(system, converse))
    assert "hello world" in texts

