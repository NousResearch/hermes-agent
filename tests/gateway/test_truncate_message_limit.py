"""``truncate_message`` never returns a chunk longer than ``max_length``.

Platforms post each chunk as one API call against a hard cap (Slack: 40,000 characters, 39,000
configured), so a chunk a few characters over is a rejected message, not a cosmetic issue.
"""

from __future__ import annotations

import random

import pytest

from gateway.platforms.base import BasePlatformAdapter, utf16_len

truncate = BasePlatformAdapter.truncate_message


def _fence_heavy(rng: random.Random) -> str:
    pieces = ["```py\n", "```\n", "a" * rng.randint(1, 80), " ", "\n", "x" * rng.randint(1, 300)]
    return "".join(rng.choice(pieces) for _ in range(rng.randint(1, 30)))


@pytest.mark.parametrize("seed", range(300))
def test_no_chunk_exceeds_the_limit_when_a_reopened_fence_is_closed(seed):
    rng = random.Random(seed)
    limit = rng.randint(30, 200)
    chunks = truncate(_fence_heavy(rng), limit)
    assert max(len(chunk) for chunk in chunks) <= limit


def test_final_chunk_that_closes_a_carried_fence_stays_within_the_limit():
    limit = 60
    text = "```\n" + "a" * 40 + "\n" + "b" * 45
    chunks = truncate(text, limit)
    assert len(chunks) > 1
    assert [len(c) for c in chunks if len(c) > limit] == []
    assert chunks[-1].rstrip().endswith("```" + f" ({len(chunks)}/{len(chunks)})")


def test_four_digit_chunk_counts_still_fit():
    limit = 40
    chunks = truncate(" ".join(["word"] * 9000), limit)
    assert len(chunks) >= 1000
    assert max(len(chunk) for chunk in chunks) <= limit
    assert chunks[-1].endswith(f"({len(chunks)}/{len(chunks)})")


def test_custom_length_function_is_respected():
    chunks = truncate("😀" * 500, 101, len_fn=utf16_len)
    assert max(utf16_len(chunk) for chunk in chunks) <= 101


def test_text_within_the_limit_is_returned_unchanged():
    assert truncate("", 39000) == [""]
    assert truncate("short", 39000) == ["short"]
    assert truncate("x" * 39000, 39000) == ["x" * 39000]


@pytest.mark.parametrize("limit", range(40, 80))
def test_an_escaped_entity_is_never_cut_across_chunks(limit):
    """Whatever offset the hard split lands on, ``&amp;`` stays whole (a cut right after the
    ``&`` included): the next chunk must not open on ``amp;``."""
    chunks = truncate("&amp;" * 200, limit)
    bodies = [chunk.rsplit(" (", 1)[0] for chunk in chunks]
    assert all(body.startswith("&amp;") and body.endswith("&amp;") for body in bodies)
    assert sum(body.count("&amp;") for body in bodies) == 200


@pytest.mark.parametrize("text, limit", [
    ("x" * 60 + " R&D " + "y" * 60, 80),
    ("x" * 60 + " AT&amp;T " + "y" * 60, 80),
    (("word " * 300 + "if x < 5 then " + "word " * 700) * 3, 2000),
], ids=["ampersand-word", "escaped-word", "comparison"])
def test_plain_ampersands_and_comparisons_split_where_neutral_text_does(text, limit):
    """Raw text for other platforms: ``R&amp;D`` or ``x < 5`` must not pull a natural split back."""
    neutral = text.replace("&", "+").replace("<", "=")
    assert [len(c) for c in truncate(text, limit)] == [len(c) for c in truncate(neutral, limit)]


def test_a_slack_link_label_is_never_cut_across_chunks():
    chunks = truncate(("word " * 9 + "<https://example.com/a|see the docs> ") * 60, 120)
    bodies = [chunk.rsplit(" (", 1)[0] for chunk in chunks]
    assert len(bodies) > 1
    assert all(body.count("<") == body.count(">") for body in bodies)
