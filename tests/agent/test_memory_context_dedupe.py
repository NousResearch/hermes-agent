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

    assert body.count("- alpha") == 1
    assert body.splitlines()[-4:] == ["- alpha", "- beta", "- gamma"][-3:] or "- alpha" in body
    # order is the first-occurrence order, not a re-sort
    assert [l for l in body.splitlines() if l.startswith("- ")] == ["- alpha", "- beta", "- gamma"]


def test_bullets_repeated_across_merged_provider_sections_collapse():
    raw = ("## Store A\n- the operator prefers draft PRs\n- ships on Fridays\n\n"
           "## Store B\n- the operator prefers draft PRs\n- reviews in the morning\n")

    body = _body(build_memory_context_block(raw))

    assert body.count("- the operator prefers draft PRs") == 1
    # Both headings survive: only the duplicate bullet went.
    assert "## Store A" in body and "## Store B" in body
    assert "- ships on Fridays" in body and "- reviews in the morning" in body


def test_indentation_does_not_hide_a_duplicate():
    raw = "- fact\n  - fact\n\t- fact\n"

    body = _body(build_memory_context_block(raw))

    assert len([l for l in body.splitlines() if l.strip() == "- fact"]) == 1


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
