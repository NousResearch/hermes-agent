"""Regression tests for #63200 — DeepSeek API rejects assistant messages
with empty content + tool_calls.

Hermes stores assistant turns with `content=""` and `tool_calls=[...]`
in two paths:
  1. Codex Responses streaming (codex_responses_adapter.py:486 emits
     `{"role": "assistant", "content": ""}` as the required following
     item for a reasoning-only turn).
  2. Reasoning-only tool-call turns where the model produces no
     visible content (build_assistant_message in chat_completion_helpers.py).

Some providers (#63200 — DeepSeek, and any strict OpenAI-compatible API)
reject these messages with HTTP 400 "An assistant message with 'tool_calls'
must be followed by tool messages responding to each 'tool_call_id'."

The fix is a filter that runs on the outgoing `api_messages` copy just
before the HTTP request: drop assistant messages whose content is
empty AND whose tool_calls is non-empty. The internal `messages` list
is untouched (session persistence and resume keep the full transcript).
"""

import pytest

from agent.message_sanitization import strip_empty_content_assistant_tool_calls


# A realistic tool_calls payload — the shape matters because the
# filter requires non-empty list.
_REALISTIC_TOOL_CALLS = [
    {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path": "/etc/hostname"}',
        },
    },
]


def test_strips_empty_content_assistant_with_tool_calls():
    """#63200: the assistant message that's the bug-shape must be dropped."""
    messages = [
        {"role": "user", "content": "Read /etc/hostname"},
        {"role": "assistant", "content": "", "tool_calls": _REALISTIC_TOOL_CALLS},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "myhost"},
        {"role": "assistant", "content": "The hostname is myhost."},
    ]
    # The function returns a NEW list — assign back. The input list
    # reference is NEVER mutated (the safer contract for a pre-API filter).
    result = strip_empty_content_assistant_tool_calls(messages)
    original_input_ref = messages  # save reference

    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "tool"
    assert result[2]["role"] == "assistant"
    assert result[2]["content"] == "The hostname is myhost."
    # The original input list is not mutated.
    assert messages is original_input_ref
    assert len(messages) == 4, "input list must not be mutated"


def test_strips_when_content_is_none():
    """Same shape but content is None (not empty string). Some
    codex_responses_adapter paths set content=None instead of ""."""
    messages = [
        {"role": "assistant", "content": None, "tool_calls": _REALISTIC_TOOL_CALLS},
    ]
    result = strip_empty_content_assistant_tool_calls(messages)
    assert result == []
    # Input not mutated.
    assert len(messages) == 1


def test_keeps_assistant_with_content_and_no_tool_calls():
    """Regression guard: a normal assistant turn must NOT be dropped."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ]
    result = strip_empty_content_assistant_tool_calls(messages)
    assert result is not messages  # returns a new list, not the same one
    assert len(result) == 2


def test_keeps_assistant_with_content_and_tool_calls():
    """An assistant turn that has BOTH content and tool_calls must NOT be
    dropped. Empty content + non-empty tool_calls is the bug; non-empty
    content is fine on its own."""
    messages = [
        {"role": "user", "content": "Read file"},
        {"role": "assistant", "content": "Reading...", "tool_calls": _REALISTIC_TOOL_CALLS},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "data"},
    ]
    result = strip_empty_content_assistant_tool_calls(messages)
    assert len(result) == 3


def test_keeps_assistant_with_empty_content_and_no_tool_calls():
    """An assistant turn with empty content but NO tool_calls must NOT be
    dropped — it's the closing turn of a turn where the model streamed
    reasoning only, and the closing message is needed for role-alternation."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": ""},  # empty but no tool_calls
    ]
    result = strip_empty_content_assistant_tool_calls(messages)
    assert len(result) == 2


def test_keeps_assistant_with_empty_content_and_empty_tool_calls_list():
    """Defense: empty content + empty tool_calls list ([]) must NOT be dropped.
    Only non-empty tool_calls combined with empty content is the bug."""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": []},
    ]
    result = strip_empty_content_assistant_tool_calls(messages)
    assert len(result) == 1


def test_handles_multiple_consecutive_empty_tool_call_messages():
    """Edge case: more than one empty-content+tool_calls message in a row
    (e.g. multiple reasoning-only streaming turns). All get dropped."""
    messages = [
        {"role": "user", "content": "Do X then Y"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "ok"},
        {"role": "assistant", "content": "Done."},
    ]
    result = strip_empty_content_assistant_tool_calls(messages)
    assert len(result) == 4
    # Order is preserved.
    assert [m["role"] for m in result] == ["user", "tool", "tool", "assistant"]


def test_handles_non_list_input_gracefully():
    """Belt-and-suspenders: the filter is robust against malformed input."""
    assert strip_empty_content_assistant_tool_calls(None) == []
    assert strip_empty_content_assistant_tool_calls("not a list") == []
    assert strip_empty_content_assistant_tool_calls([]) == []


def test_does_not_mutate_non_matching_messages():
    """Belt-and-suspenders: messages that don't match the bug shape are
    passed through with their full structure intact."""
    messages = [
        {
            "role": "assistant",
            "content": "Here you go.",
            "tool_calls": _REALISTIC_TOOL_CALLS,
            "reasoning": "the user asked for X",
            "reasoning_content": "the user asked for X",
            "finish_reason": "tool_calls",
        },
    ]
    original = messages.copy()
    result = strip_empty_content_assistant_tool_calls(messages)
    assert len(result) == 1
    assert result == original
    # Input unchanged.
    assert messages == original


def test_returns_new_list_not_same_object():
    """The function must return a NEW list object (not the input).
    This is the contract that makes the wire-up safe — caller does
    ``api_messages = strip_...(api_messages)`` and the input reference
    stays valid."""
    messages = [
        {"role": "user", "content": "Hi"},
    ]
    result = strip_empty_content_assistant_tool_calls(messages)
    assert result is not messages
