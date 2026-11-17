"""A compaction summary merged into an assistant row must drop that row's signed thinking (#152513).

``_merge_summary_into_tail_row`` rewrites the carrier row's text into a handoff summary. The row's
pre-merge reasoning stayed behind, and the Anthropic converter replays it as a signed ``thinking``
block ahead of the summary text — the pairing the provider classifies as reasoning extraction and
refuses, which poisons every later turn of the session because retries resend the same history.
A rewrite invalidates the signature (it was minted for the pre-merge text), so the merged row's
reasoning fields go with the rewritten content.

Scope: only the compaction merge opts in (``drop_reasoning``); every other content rewrite keeps the
row's reasoning exactly as it was.
"""

from __future__ import annotations

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _SUMMARY_END_MARKER,
    ContextCompressor,
)
from agent.turn_context import drop_stale_api_content

SUMMARY_BODY = "## Active Task\nsummarized prior work"
SIGNATURE = "c2lnbmF0dXJl"
REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details")


def _signed_assistant_row() -> dict:
    return {
        "role": "assistant",
        "content": "the pre-merge answer",
        "reasoning": "why the old answer was given",
        "reasoning_content": "why the old answer was given",
        "reasoning_details": [
            {"type": "thinking", "thinking": "why the old answer was given", "signature": SIGNATURE}
        ],
    }


def _merge(row: dict) -> None:
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor._summary_has_user_turn = False
    compressor._merge_summary_into_tail_row(row, SUMMARY_BODY, "assistant", False)


def test_merge_into_an_assistant_row_drops_its_reasoning():
    row = _signed_assistant_row()

    _merge(row)

    assert [key for key in REASONING_KEYS if key in row] == []
    # The merge layout itself is unchanged: prior text, then the summary, then the end marker.
    assert row["content"].startswith(_MERGED_PRIOR_CONTEXT_HEADER + "\n")
    assert "the pre-merge answer" in row["content"]
    assert SUMMARY_BODY in row["content"]
    assert row["content"].rstrip().endswith(_SUMMARY_END_MARKER.strip())
    assert row[COMPRESSED_SUMMARY_METADATA_KEY] is True


def test_the_merged_row_reaches_anthropic_without_a_thinking_block():
    row = _signed_assistant_row()

    _merge(row)

    _system, wire = convert_messages_to_anthropic([row], model="claude-opus-5-5")
    assistant = next(message for message in wire if message["role"] == "assistant")

    assert [block["type"] for block in assistant["content"]] == ["text"]
    assert SIGNATURE not in str(assistant["content"])
    assert SUMMARY_BODY in assistant["content"][0]["text"]


def test_other_rewrite_sites_still_replay_reasoning():
    """The drop is opt-in: a rewrite that leaves the row's text alone keeps its reasoning."""
    row = _signed_assistant_row()

    drop_stale_api_content(row)

    assert row["reasoning_details"] == _signed_assistant_row()["reasoning_details"]
    assert row["reasoning_content"] == "why the old answer was given"
