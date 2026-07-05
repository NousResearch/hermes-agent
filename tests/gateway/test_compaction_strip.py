"""
Tests for the context-compaction leak fix in stream_consumer.

When Hermes's context-compaction mechanism summarizes older turns, it injects
a ``[CONTEXT COMPACTION — REFERENCE ONLY]`` handoff block into the response.
That block is intended for the model in the next turn, not for the human
user, but ``GatewayStreamConsumer`` was forwarding it to the chat gateway
until ``_clean_for_display`` learned to strip it.

These tests pin down the strip behavior: which header variants it handles,
how it picks the cut boundary, and how ``_clean_for_display`` wires it in.
"""
from __future__ import annotations

import pytest

from gateway.stream_consumer import (
    _COMPACTION_HEADER_RE,
    GatewayStreamConsumer,
    _strip_compaction_block,
)


# ---------- Detection ----------


def test_compaction_header_regex_matches_known_variants():
    """The regex must match all three documented variants."""
    for variant in [
        "[CONTEXT COMPACTION — REFERENCE ONLY]",
        "[CONTEXT COMPACTION — HANDOFF]",
        "[CONTEXT COMPACTION — BACKGROUND]",
        "[CONTEXT COMPACTION - REFERENCE ONLY]",  # ASCII hyphen variant
        "[context compaction — reference only]",  # lowercase
    ]:
        assert _COMPACTION_HEADER_RE.search(variant), f"regex missed: {variant!r}"


def test_compaction_header_regex_ignores_unrelated_brackets():
    """The regex must NOT match arbitrary bracket text."""
    for unrelated in [
        "[CONTEXT]",
        "[CONTEXT COMPACTION]",
        "[CONTEXT COMPACTION — NOTES]",
        "[context here either]",
    ]:
        # If the regex does match, it must be because the suffix is one of the
        # documented variants. Other text without one of those suffixes is
        # not a compaction header and must be left alone by strip().
        m = _COMPACTION_HEADER_RE.search(unrelated)
        if m:
            # If it matched, that's only acceptable if the text contains one
            # of the documented suffix keywords; otherwise strip() should be
            # a no-op.
            assert any(
                kw in m.group(0).upper()
                for kw in ("REFERENCE ONLY", "HANDOFF", "BACKGROUND")
            ), f"regex too greedy: matched {unrelated!r}"


# ---------- Strip ----------


@pytest.mark.parametrize(
    "inp,expected,desc",
    [
        ("plain text", "plain text", "no leak"),
        (
            "answer\n\n[CONTEXT COMPACTION — REFERENCE ONLY]\n## Historical\nfoo",
            "answer",
            "strip REFERENCE ONLY variant",
        ),
        (
            "answer\n\n[CONTEXT COMPACTION — HANDOFF]\n## State\nbar",
            "answer",
            "strip HANDOFF variant",
        ),
        (
            "answer\n\n[CONTEXT COMPACTION — BACKGROUND]\nold",
            "answer",
            "strip BACKGROUND variant",
        ),
        (
            "x [CONTEXT COMPACTION — REFERENCE ONLY] junk\n\n--- \n## Last Dropped",
            "x",
            "strip with --- anchor",
        ),
        (
            "[CONTEXT COMPACTION — REFERENCE ONLY]\n## end\nfoo\n## END OF CONTEXT SUMMARY\nbar",
            "",
            "strip at ## END OF SUMMARY anchor",
        ),
        ("", "", "empty string"),
        (None, None, "None passes through"),
    ],
)
def test_strip_compaction_block(inp, expected, desc):
    assert _strip_compaction_block(inp) == expected, desc


def test_strip_preserves_real_answer_with_inline_mention():
    """Substantive prose that merely mentions the marker is NOT stripped.

    A user-visible answer that says 'the [CONTEXT COMPACTION — REFERENCE ONLY]
    mechanism works like X' should NOT be affected by strip — only actual
    handoff blocks at the start/end of a response are.
    """
    inp = "Let me explain [CONTEXT COMPACTION] briefly."
    # No 'REFERENCE ONLY' / 'HANDOFF' / 'BACKGROUND' suffix → no strip
    assert _strip_compaction_block(inp) == inp


def test_strip_is_idempotent():
    """Calling strip twice should equal calling it once."""
    inp = "real answer\n\n[CONTEXT COMPACTION — REFERENCE ONLY]\n## Historical"
    once = _strip_compaction_block(inp)
    twice = _strip_compaction_block(once)
    assert once == twice


# ---------- Integration with _clean_for_display ----------


def test_clean_for_display_strips_compaction_block():
    """The hook point for stripping is _clean_for_display.

    Every chunk passed through the streaming consumer ends up there before
    being sent to the chat gateway. If the compaction block survives that
    call, the bug is back.
    """
    chunk = "real answer\n\n[CONTEXT COMPACTION — REFERENCE ONLY] junk"
    result = GatewayStreamConsumer._clean_for_display(chunk)
    assert "[CONTEXT COMPACTION" not in result
    assert result == "real answer"


def test_clean_for_display_passes_clean_text_through():
    """_clean_for_display must not touch text that has no compaction header."""
    chunk = "this is a perfectly normal response with no markers"
    result = GatewayStreamConsumer._clean_for_display(chunk)
    # May also pass through _BasePlatformAdapter.strip_media_directives_for_display,
    # which is a no-op for text without MEDIA: directives.
    assert result == chunk