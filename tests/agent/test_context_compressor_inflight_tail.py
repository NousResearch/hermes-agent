"""Behavioral coverage for compaction inside a long, unfinished tool loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
    _SUMMARY_END_MARKER,
)


TASK = "Implement the reservation-aware stock summary and verify it."


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        compressor = ContextCompressor(
            model="test",
            quiet_mode=True,
            protect_first_n=3,
            protect_last_n=2,
        )
    compressor.compression_count = 1
    compressor.tail_token_budget = 500
    return compressor


def _tool_pairs(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        call_id = f"call-{index}"
        rows.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": "terminal",
                                "arguments": '{"cmd":"read-only probe"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": ("synthetic output " * 200) + str(index),
                },
            ]
        )
    return rows


def _resumed_inflight_transcript() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": SUMMARY_PREFIX + "\nEarlier work.\n" + _SUMMARY_END_MARKER,
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "assistant", "content": "Continuing."},
        {"role": "user", "content": TASK},
        *_tool_pairs(30),
    ]


def test_tail_cut_can_cross_inflight_user_after_a_prior_handoff():
    compressor = _compressor()
    messages = _resumed_inflight_transcript()
    head_end = compressor._protect_head_size(messages)

    cut = compressor._find_tail_cut_by_tokens(messages, head_end)

    assert cut > 2, (
        "the latest in-flight user turn anchored the entire tool loop, leaving "
        "nothing useful for compression"
    )
    assert messages[cut]["role"] == "assistant"
    assert messages[cut].get("tool_calls"), "the retained tool group must stay intact"


def test_compress_rolls_up_old_inflight_steps_and_replays_exact_task():
    compressor = _compressor()
    messages = _resumed_inflight_transcript()
    summary = SUMMARY_PREFIX + "\n## Summary\nEarlier synthetic steps completed."

    with patch.object(compressor, "_generate_summary", return_value=summary):
        result = compressor.compress(messages, current_tokens=30_000, force=True)

    assert len(result) < len(messages) // 2
    handoff_index = next(
        index
        for index, message in enumerate(result)
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    )
    after_handoff = result[handoff_index:]
    assert any(
        TASK in str(message.get("content")) for message in after_handoff
    ), "the exact unfinished task must remain actionable after the handoff"

    retained_calls = {
        call["id"]
        for message in result
        for call in message.get("tool_calls", [])
    }
    retained_results = {
        str(message.get("tool_call_id"))
        for message in result
        if message.get("role") == "tool"
    }
    assert retained_calls == retained_results
    assert "call-29" in retained_calls


def test_completed_turn_keeps_the_existing_user_anchor():
    compressor = _compressor()
    messages = [
        *_resumed_inflight_transcript(),
        {"role": "assistant", "content": "Completed and verified."},
    ]
    head_end = compressor._protect_head_size(messages)

    cut = compressor._find_tail_cut_by_tokens(messages, head_end)

    assert cut <= 2
    assert TASK in [str(message.get("content")) for message in messages[cut:]]
