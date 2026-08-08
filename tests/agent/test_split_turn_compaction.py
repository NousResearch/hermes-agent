"""Regression for #80449 — an oversized in-progress turn must stay compressible.

When one turn (opening user message + many individually small tool groups) grows past
the protected-tail soft ceiling, anchoring the cut back to the turn-opening request kept
the whole turn verbatim: compaction re-fired with an empty summarizable window and the
session sat over threshold. The cut must instead land on a tool-group-aligned mid-turn
boundary, and the active request must survive the handoff.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    _estimate_msg_budget_tokens,
)


_ACTIVE_REQUEST = "Inspect every shard and preserve the active request exactly."
_TOKEN_BUDGET = 250


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=0,
            protect_last_n=3,
            quiet_mode=True,
        )
        _ = instance.context_length
    instance.tail_token_budget = _TOKEN_BUDGET
    return instance


def _tool_group(index: int) -> list[dict]:
    call_id = f"call_{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "inspect_shard",
                        # Large enough for the complete turn to exceed the
                        # ceiling, but below the phase-1 argument-prune limit.
                        "arguments": "x" * 440,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            # Keep each result below the phase-1 result-prune floor. The bug is
            # aggregate turn size, not one individually oversized result.
            "content": f"result-{index}:" + "r" * 110,
        },
    ]


def _oversized_active_turn() -> list[dict]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older request completed"},
        {"role": "user", "content": _ACTIVE_REQUEST},
    ]
    for index in range(10):
        messages.extend(_tool_group(index))
    return messages


def _assert_tool_pairs_are_complete(messages: list[dict]) -> None:
    call_ids = {
        call["id"]
        for message in messages
        for call in message.get("tool_calls") or []
    }
    result_ids = {
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    }
    assert call_ids == result_ids


def test_oversized_active_turn_uses_a_mid_turn_tool_boundary(
    compressor: ContextCompressor,
) -> None:
    messages = _oversized_active_turn()
    head_end = compressor._protect_head_size(messages)

    cut = compressor._find_tail_cut_by_tokens(
        messages,
        head_end,
        token_budget=_TOKEN_BUDGET,
    )

    active_user_idx = next(
        index
        for index, message in enumerate(messages)
        if message.get("content") == _ACTIVE_REQUEST
    )
    tail_tokens = sum(_estimate_msg_budget_tokens(msg) for msg in messages[cut:])

    assert cut > active_user_idx
    # Tool-group alignment may retain one additional indivisible group beyond
    # the scalar ceiling. It must not retain the whole active turn.
    max_group_tokens = max(
        sum(_estimate_msg_budget_tokens(msg) for msg in _tool_group(index))
        for index in range(10)
    )
    assert tail_tokens <= int(_TOKEN_BUDGET * 1.5) + max_group_tokens
    assert tail_tokens < sum(
        _estimate_msg_budget_tokens(msg) for msg in messages[active_user_idx:]
    )
    assert messages[cut]["role"] == "assistant"
    _assert_tool_pairs_are_complete(messages[head_end:cut])
    _assert_tool_pairs_are_complete(messages[cut:])


def test_full_compaction_preserves_active_request_and_tool_pairs(
    compressor: ContextCompressor,
) -> None:
    messages = _oversized_active_turn()

    # Exercise the deterministic handoff too: even when the summary model is
    # unavailable, splitting the turn must not lose the opening request.
    with patch.object(compressor, "_generate_summary", return_value=None):
        compressed = compressor.compress(messages, current_tokens=90_000)

    summary_rows = [
        message
        for message in compressed
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(summary_rows) == 1
    assert _ACTIVE_REQUEST in str(summary_rows[0].get("content"))
    assert sum(
        _ACTIVE_REQUEST in str(message.get("content"))
        for message in compressed
    ) == 1
    assert len(compressed) < len(messages)
    _assert_tool_pairs_are_complete(compressed)