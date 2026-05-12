"""Regression coverage for MiniMax Anthropic-compatible thinking replay."""

from __future__ import annotations

import pytest


MINIMAX_ANTHROPIC_ENDPOINTS = (
    "https://api.minimax.io/anthropic",
    "https://api.minimaxi.com/anthropic",
)


@pytest.mark.parametrize("base_url", MINIMAX_ANTHROPIC_ENDPOINTS)
def test_minimax_preserves_unsigned_thinking_on_replay(base_url: str) -> None:
    """Both MiniMax hosts require reasoning_content-derived blocks to round-trip."""
    from agent.anthropic_adapter import convert_messages_to_anthropic

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "reasoning_content": "planning the tool call",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "skill_view", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    _system, converted = convert_messages_to_anthropic(
        messages,
        base_url=base_url,
    )

    assistant_msg = next(m for m in converted if m["role"] == "assistant")
    thinking_blocks = [
        block
        for block in assistant_msg["content"]
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]
    assert thinking_blocks == [
        {"type": "thinking", "thinking": "planning the tool call"}
    ]


@pytest.mark.parametrize("base_url", MINIMAX_ANTHROPIC_ENDPOINTS)
def test_minimax_strips_anthropic_signed_thinking(base_url: str) -> None:
    """MiniMax cannot validate Anthropic-proprietary thinking signatures."""
    from agent.anthropic_adapter import convert_messages_to_anthropic

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "anthropic-signed payload",
                    "signature": "anthropic-sig-xyz",
                },
                {"type": "text", "text": "hello"},
            ],
        },
        {"role": "user", "content": "again"},
    ]

    _system, converted = convert_messages_to_anthropic(
        messages,
        base_url=base_url,
    )

    assistant_msg = next(m for m in converted if m["role"] == "assistant")
    assert assistant_msg["content"] == [{"type": "text", "text": "hello"}]
