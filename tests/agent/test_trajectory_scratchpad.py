"""Regression tests for agent.trajectory scratchpad boundary detection (#11194)."""

import pytest

from agent.trajectory import has_incomplete_scratchpad

OPEN = "<REASONING_SCRATCHPAD>"
CLOSE = "</REASONING_SCRATCHPAD>"


class TestHasIncompleteScratchpad:
    def test_empty_not_incomplete(self):
        assert has_incomplete_scratchpad("") is False

    def test_plain_text_not_incomplete(self):
        assert has_incomplete_scratchpad("hello") is False

    def test_single_closed_pair_not_incomplete(self):
        assert has_incomplete_scratchpad(f"{OPEN}thought{CLOSE}") is False

    def test_adjacent_empty_pair_not_incomplete(self):
        assert has_incomplete_scratchpad(f"{OPEN}{CLOSE}") is False

    def test_two_closed_pairs_not_incomplete(self):
        body = f"{OPEN}a{CLOSE} mid {OPEN}b{CLOSE}"
        assert has_incomplete_scratchpad(body) is False

    def test_second_block_unclosed_is_incomplete(self):
        """Global '</...>' check wrongly returns False when an earlier pair closed."""
        body = f"{OPEN}a{CLOSE}{OPEN}still streaming"
        assert has_incomplete_scratchpad(body) is True

    def test_orphan_close_then_open_unclosed_is_incomplete(self):
        """Closing tag elsewhere must not mask a later unclosed block."""
        body = f"{CLOSE}{OPEN}orphan prefix then real block"
        assert has_incomplete_scratchpad(body) is True

    def test_only_open_tag_is_incomplete(self):
        assert has_incomplete_scratchpad(f"{OPEN}no close") is True
