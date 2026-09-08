"""Regression tests for Bedrock Converse reasoning-only assistant turn rejection.

Bedrock's Converse API raises on Sonnet 5 / Fable 5.1::

    ValidationException: This model does not support assistant message prefill.
    The conversation must end with a user message.

when an assistant turn in the conversation history contains ONLY ``reasoningContent``
blocks (redacted thinking, no text, no toolUse) — e.g. a reasoning-only turn produced
by claude-sonnet-5 or claude-fable-5-1 with extended thinking. Older models
(claude-sonnet-4-6) tolerate it. Ref: issue #105780.

``convert_messages_to_converse`` drops such turns: Bedrock does not need its own prior
reasoning replayed (it stays in the trajectory's ``reasoning_details`` sidecar), and the
following user turn (or the tail pad) remains the last message Bedrock sees. Populated
turns (reasoning + text, reasoning + toolUse) are kept.
"""
from agent.bedrock_adapter import convert_messages_to_converse

# "decryptedreasoning" base64 — _decode_redacted accepts strict base64 and returns bytes.
_REASONING_B64 = "ZGVjcnlwdGVkcmVhc29uaW5n"


def _reasoning_only_assistant():
    """An assistant turn whose only payload is redacted reasoning (no text/toolUse)."""
    return {
        "role": "assistant",
        "content": None,
        "reasoning_details": [{"type": "redacted_thinking", "data": _REASONING_B64}],
    }


def test_reasoning_only_assistant_turn_is_dropped():
    messages = [
        {"role": "user", "content": "what is 2+2?"},
        _reasoning_only_assistant(),
        {"role": "user", "content": "and 3+3?"},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    # No assistant turn survives — the two user turns merge into one (same-role merge).
    assert [m["role"] for m in msgs] == ["user"]
    # No reasoningContent block reaches Bedrock.
    assert all("reasoningContent" not in b for m in msgs for b in m["content"])


def test_reasoning_only_assistant_at_tail_pads_user_last():
    messages = [
        {"role": "user", "content": "q"},
        _reasoning_only_assistant(),
    ]
    _system, msgs = convert_messages_to_converse(messages)
    # The reasoning-only assistant is dropped, so the tail pad adds a user turn.
    assert msgs[-1]["role"] == "user"
    assert all("reasoningContent" not in b for m in msgs for b in m["content"])


def test_reasoning_only_assistant_at_head_pads_user_first():
    messages = [
        _reasoning_only_assistant(),
        {"role": "user", "content": "q"},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "user"
    assert all("reasoningContent" not in b for m in msgs for b in m["content"])


def test_reasoning_plus_text_assistant_turn_is_kept():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_details": [{"type": "redacted_thinking", "data": _REASONING_B64}],
        },
        {"role": "user", "content": "next"},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    # The assistant turn has text, so it survives — alternation user/assistant/user.
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assistant_blocks = msgs[1]["content"]
    assert any("text" in b for b in assistant_blocks)
    assert any("reasoningContent" in b for b in assistant_blocks)


def test_reasoning_plus_tooluse_assistant_turn_is_kept():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_details": [{"type": "redacted_thinking", "data": _REASONING_B64}],
            "tool_calls": [{"id": "call_1", "function": {"name": "calc", "arguments": '{"x":1}'}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        {"role": "user", "content": "next"},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    # The assistant turn has a toolUse, so it survives (reasoning + toolUse).
    assert "assistant" in [m["role"] for m in msgs]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert any("toolUse" in b for b in assistant["content"])
    assert any("reasoningContent" in b for b in assistant["content"])


def test_bedrock_content_blocks_sidecar_reasoning_only_is_dropped():
    """The ordered bedrock_content_blocks sidecar path also drops reasoning-only turns."""
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "bedrock_content_blocks": [{"reasoningContent": {"text": "internal reasoning"}}],
        },
        {"role": "user", "content": "next"},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    assert [m["role"] for m in msgs] == ["user"]
    assert all("reasoningContent" not in b for m in msgs for b in m["content"])


def test_text_only_assistant_turn_is_kept():
    """Sanity: a normal text-only assistant turn is unaffected by the reasoning-only guard."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
    ]
    _system, msgs = convert_messages_to_converse(messages)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == [{"text": "answer"}]
