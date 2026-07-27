"""Regression tests for Bedrock Converse empty/whitespace text block rejection.

Bedrock's Converse API raises::

    ValidationException: The model returned the following errors: messages:
    text content blocks must contain non-whitespace text

for ANY text content block that is empty OR whitespace-only. Anthropic's native
API tolerates these; Bedrock does not. Such blocks most often appear after
context compression rewrites an assistant/tool turn into a blank string — once
in history the block is re-sent on every call and fails deterministically
(all retries fail identically). Ref: issue #9486.

These tests assert convert_messages_to_converse never emits a blank text block
(including inside toolResult content) and never uses a whitespace-only
placeholder (a lone space would be rejected by the same validation).
"""
import pytest

from agent.bedrock_adapter import (
    convert_messages_to_converse,
    _convert_content_to_converse,
    _safe_text,
    _EMPTY_TEXT_PLACEHOLDER,
)


def _iter_text_blocks(msgs):
    """Yield every text string that will be sent to Bedrock, incl. toolResult."""
    for m in msgs:
        for b in m["content"]:
            if "text" in b:
                yield b["text"]
            if "toolResult" in b:
                for tb in b["toolResult"]["content"]:
                    if "text" in tb:
                        yield tb["text"]


def test_placeholder_is_non_whitespace():
    # The core lesson of #9486: a space is whitespace and is itself rejected.
    assert _EMPTY_TEXT_PLACEHOLDER.strip(), "placeholder must be non-whitespace"


@pytest.mark.parametrize("value", ["", "   ", "\n\n", "\t", None])
def test_safe_text_blank_inputs_become_non_whitespace(value):
    assert _safe_text(value).strip()


def test_safe_text_preserves_real_content():
    assert _safe_text("hello") == "hello"
    assert _safe_text("  padded  ") == "  padded  "  # inner content kept verbatim


def test_no_blank_blocks_reach_bedrock():
    """The exact failing history: blank system/assistant/tool/user turns."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "system", "content": [{"type": "text", "text": "   "}]},
        {"role": "user", "content": "search for foo"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "tc1",
                         "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": ""},        # empty tool output
        {"role": "assistant", "content": "   \n\n  "},                 # whitespace-only (compaction)
        {"role": "user", "content": [{"type": "text", "text": ""}]},
        {"role": "assistant", "content": None},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    for text in _iter_text_blocks(msgs):
        assert text.strip(), f"blank text block would be rejected by Bedrock: {text!r}"


def test_empty_tool_result_gets_placeholder():
    """A tool that returns no output must not produce a blank toolResult block."""
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "function": {"name": "sh", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "   "},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    tool_msg = next(m for m in msgs
                    if any("toolResult" in b for b in m["content"]))
    block = next(b for b in tool_msg["content"] if "toolResult" in b)
    text = block["toolResult"]["content"][0]["text"]
    assert text.strip()


def test_real_content_is_preserved_alongside_blank_siblings():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "  "},
            {"type": "text", "text": "real question"},
        ]},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    texts = list(_iter_text_blocks(msgs))
    assert "real question" in texts
    assert all(t.strip() for t in texts)


def test_blank_system_blocks_dropped_not_blanked():
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "keep me"},
            {"type": "text", "text": "   "},
        ]},
        {"role": "user", "content": "hi"},
    ]
    system, _msgs = convert_messages_to_converse(messages)
    assert system is not None
    for block in system:
        assert block["text"].strip()
    assert any(b["text"] == "keep me" for b in system)
