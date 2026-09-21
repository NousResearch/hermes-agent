"""A recalled bullet is stated once per ``<memory-context>`` block.

Providers merge several stores, and the manager merges several providers, so one prefetch routinely
surfaces the same fact more than once. The composed block is stamped into the user row's
``api_content`` sidecar and replayed verbatim on every later request, so a byte-identical repeat is
paid once per turn for the life of the row — while telling the model nothing the block has not
already said.

Structure is not touched: headings, prose, blank lines and ``---`` rules survive as written, so a
provider's sections still read the way it wrote them.
"""
from __future__ import annotations

from agent.memory_manager import build_memory_context_block


def _body(block: str) -> str:
    """The provider's own content, without the wrapper and the system note that precedes it."""
    return block.split("]\n\n", 1)[1].rsplit("\n</memory-context>", 1)[0]


def test_a_repeated_bullet_is_kept_once_in_first_position():
    raw = "- alpha\n- beta\n- alpha\n- gamma\n"

    body = _body(build_memory_context_block(raw))

    # First-occurrence order, not a re-sort, and nothing else moved.
    assert body.splitlines() == ["- alpha", "- beta", "- gamma"]


def test_a_bullet_with_continuation_lines_is_never_touched():
    """Two entries can share a headline and differ underneath it. Dropping one would re-parent its
    provenance under the other and invent a record neither provider reported."""
    raw = ("- prefers draft PRs\n  (logged 12 Jan, source: supermemory)\n"
           "- prefers draft PRs\n  (logged 3 Feb, source: builtin)\n")

    assert _body(build_memory_context_block(raw)) == raw


def test_a_bold_heading_is_not_a_bullet():
    """``**Preferences**`` starts with ``*``; without the whitespace test it entered the rule and a
    repeated section heading was silently deleted."""
    raw = "**Preferences**\n- a\n\n**Preferences**\n- b\n"

    assert _body(build_memory_context_block(raw)) == raw


def test_emphasis_and_numbered_items_are_left_as_written():
    for raw in ("*Important:*\n- x\n*Important:*\n- y\n",
                "1. ships on Fridays\n2. reviews in the morning\n1. ships on Fridays\n"):
        assert _body(build_memory_context_block(raw)) == raw


def test_bullets_repeated_across_merged_provider_sections_collapse():
    raw = ("## Store A\n- the operator prefers draft PRs\n- ships on Fridays\n\n"
           "## Store B\n- the operator prefers draft PRs\n- reviews in the morning\n")

    body = _body(build_memory_context_block(raw))

    assert body.count("- the operator prefers draft PRs") == 1
    # Both headings survive: only the duplicate bullet went.
    assert "## Store A" in body and "## Store B" in body
    assert "- ships on Fridays" in body and "- reviews in the morning" in body


def test_indented_duplicates_at_the_same_depth_still_collapse():
    raw = "  - fact\n  - other\n  - fact\n"

    body = _body(build_memory_context_block(raw))

    assert body.splitlines() == ["  - fact", "  - other"]


def test_a_nested_bullet_is_a_child_and_is_left_alone():
    """``- fact`` followed by a deeper ``- fact`` is a parent and its child, not a repeat."""
    raw = "- fact\n  - fact\n"

    assert _body(build_memory_context_block(raw)) == raw


def test_structure_and_prose_are_left_exactly_as_written():
    raw = ("# Recall\n\n---\n\n- a\n\n---\n\nThe same sentence.\nThe same sentence.\n\n---\n"
           "-\n-\n")

    body = _body(build_memory_context_block(raw))

    assert body.count("---") == 3, "separator rules are structure, never duplicates"
    assert body.count("The same sentence.") == 2, "prose lines are not list items"
    assert body.count("\n-\n") >= 1, "a bare dash carries no content to duplicate"


def test_a_block_without_repeats_is_passed_through_byte_for_byte():
    raw = "## Memory\n- one\n- two\n- three\n"

    assert _body(build_memory_context_block(raw)) == raw


def test_the_pre_wrapped_warning_still_tracks_sanitization_not_dedupe(caplog):
    """A deduped bullet is routine; only a provider returning wrapped context is a fault."""
    with caplog.at_level("WARNING"):
        build_memory_context_block("- dup\n- dup\n")
    assert not [r for r in caplog.records if "pre-wrapped" in r.message]

    with caplog.at_level("WARNING"):
        build_memory_context_block("<memory-context>\n- x\n</memory-context>")
    assert [r for r in caplog.records if "pre-wrapped" in r.message]


def test_an_empty_or_blank_prefetch_still_yields_no_block():
    assert build_memory_context_block("") == ""
    assert build_memory_context_block("   \n\t\n") == ""
