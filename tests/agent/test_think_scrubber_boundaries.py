"""Chunk-boundary invariants for the production streaming think scrubber."""

from __future__ import annotations

import pytest

from agent.think_scrubber import StreamingThinkScrubber


def _scrub(chunks: list[str]) -> str:
    scrubber = StreamingThinkScrubber()
    return "".join([*(scrubber.feed(chunk) for chunk in chunks), scrubber.flush()])


@pytest.mark.parametrize(
    ("prefix", "close_tag", "separator", "suffix"),
    [
        ("Visible", "</think>", " ", "answer"),
        ("Visible", "</reasoning>", "\n\t", "answer"),
    ],
)
def test_orphan_close_whitespace_is_independent_of_chunk_boundaries(
    prefix: str, close_tag: str, separator: str, suffix: str
) -> None:
    """Transport chunking cannot change the user-visible assistant text."""
    complete = prefix + close_tag + separator + suffix

    assert _scrub([prefix + close_tag, separator, suffix]) == _scrub([complete])


def test_orphan_close_state_does_not_cross_a_following_reasoning_pair() -> None:
    """Every three-chunk split must match unchanged complete-string semantics."""
    complete = "Visible</think>\n<think>hidden</think>  answer"
    expected = "Visible  answer"

    assert _scrub([complete]) == expected
    for first in range(1, len(complete)):
        for second in range(first + 1, len(complete)):
            assert _scrub(
                [complete[:first], complete[first:second], complete[second:]]
            ) == expected
