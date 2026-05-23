"""Tests for Cursor SDK usage mapping."""

from __future__ import annotations

from agent.transports.cursor_usage import cursor_turn_usage_to_hermes


def test_cursor_turn_usage_maps_input_tokens_to_prompt():
    usage = {
        "inputTokens": 10686,
        "outputTokens": 33,
        "cacheReadTokens": 1883,
        "cacheWriteTokens": 0,
    }
    mapped = cursor_turn_usage_to_hermes(usage)
    assert mapped is not None
    assert mapped["prompt_tokens"] == 10686
    assert mapped["completion_tokens"] == 33
    assert mapped["total_tokens"] == 10719
    assert mapped["cache_read_tokens"] == 1883
