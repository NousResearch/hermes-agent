"""Regression tests for length-continuation interim assistant messages."""

from __future__ import annotations


def test_length_continuation_interim_message_gets_placeholder_content():
    """Strict OpenAI-compatible APIs reject assistant messages with no content/tools.

    When a response is truncated after spending all visible output tokens on
    non-text provider-side work, the continuation path appends an interim
    assistant message before asking the model to continue.  That interim
    message must contain either visible content or tool calls; otherwise the
    next API request can be rejected before the model ever sees the
    continuation prompt.
    """
    from agent.conversation_loop import _ensure_length_continuation_message_is_non_empty

    msg = {"role": "assistant", "content": None}

    _ensure_length_continuation_message_is_non_empty(msg)

    assert msg == {
        "role": "assistant",
        "content": "[Response interrupted by length limit]",
    }


def test_length_continuation_interim_message_preserves_existing_content():
    from agent.conversation_loop import _ensure_length_continuation_message_is_non_empty

    msg = {"role": "assistant", "content": "partial text"}

    _ensure_length_continuation_message_is_non_empty(msg)

    assert msg == {"role": "assistant", "content": "partial text"}


def test_length_continuation_interim_message_preserves_tool_calls_without_content():
    from agent.conversation_loop import _ensure_length_continuation_message_is_non_empty

    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
    msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}

    _ensure_length_continuation_message_is_non_empty(msg)

    assert msg == {"role": "assistant", "content": None, "tool_calls": tool_calls}
