"""Unit tests for the whitespace-aware join of truncated response parts.

When a provider stream is truncated and continued, each continuation pass
appends its partial text to ``truncated_response_parts``.  A plain
``"".join`` of those parts can glue the end of one chunk directly onto the
start of the next (e.g. ``"index.html"`` + ``"Review the 5 changes"`` →
``"index.htmlReview the 5 changes"``).  ``_join_truncated_parts`` inserts a
newline at each join point where neither boundary already provides
whitespace (#78577).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def join_parts():
    from agent.conversation_loop import _join_truncated_parts

    return _join_truncated_parts


class TestJoinTruncatedParts:
    def test_parts_that_would_glue_get_newline(self, join_parts):
        # Exact symptom from #78577: no whitespace on either boundary.
        assert join_parts(["index.html", "Review the 5 changes"]) == (
            "index.html\nReview the 5 changes"
        )

    def test_word_boundary_glue_gets_newline(self, join_parts):
        assert join_parts(["The final answer is", "42."]) == (
            "The final answer is\n42."
        )

    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (["first part ", "second part"], "first part second part"),
            (["first part", " second part"], "first part second part"),
            (["first part\n", "second part"], "first part\nsecond part"),
            (["first part\t", "second part"], "first part\tsecond part"),
        ],
    )
    def test_existing_whitespace_is_preserved(self, join_parts, parts, expected):
        assert join_parts(parts) == expected

    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (["first", "", "second"], "first\nsecond"),
            (["", "only"], "only"),
            (["first", ""], "first"),
            (["only"], "only"),
            ([], ""),
        ],
    )
    def test_empty_parts_are_skipped(self, join_parts, parts, expected):
        assert join_parts(parts) == expected

    def test_trailing_suffix_boundary_gets_newline(self, join_parts):
        assert join_parts(["index.html"], trailing="Review the 5 changes") == (
            "index.html\nReview the 5 changes"
        )

    def test_trailing_suffix_existing_whitespace_preserved(self, join_parts):
        assert join_parts(["index.html"], trailing=" Review the 5 changes") == (
            "index.html Review the 5 changes"
        )
        assert join_parts(["index.html "], trailing="Review") == "index.html Review"

    def test_trailing_suffix_alone(self, join_parts):
        assert join_parts([], trailing="solo") == "solo"
