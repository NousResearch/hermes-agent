"""Compatibility seam tests for the conversation text helper."""

from agent import conversation_loop, conversation_text


def test_join_truncated_parts_is_identity_preserving_reexport():
    assert (
        conversation_loop._join_truncated_parts
        is conversation_text._join_truncated_parts
    )


def test_join_truncated_parts_preserves_existing_whitespace():
    assert conversation_loop._join_truncated_parts(["one ", "two", " three"]) == (
        "one two three"
    )


def test_join_truncated_parts_adds_newline_only_when_fragments_would_glue():
    assert conversation_loop._join_truncated_parts(["one", "two"]) == "one\ntwo"
    assert conversation_loop._join_truncated_parts(["", "two", ""]) == "two"


def test_join_truncated_parts_handles_empty_input():
    assert conversation_loop._join_truncated_parts([]) == ""
