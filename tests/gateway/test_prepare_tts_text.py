"""Regression tests for BasePlatformAdapter.prepare_tts_text().

Verifies that markdown emphasis stripping does NOT corrupt literal asterisks
in cron expressions, math, or other non-markdown content.

See: https://github.com/NousResearch/hermes-agent/pull/28010
"""

import re

import pytest


class TestPrepareTtsText:
    """Test the prepare_tts_text asterisk-aware sanitization."""

    def _call(self, text: str) -> str:
        """Import and call prepare_tts_text without instantiating the ABC."""
        from gateway.platforms.base import BasePlatformAdapter

        return BasePlatformAdapter.prepare_tts_text(None, text)

    # --- Must strip: valid markdown emphasis ---

    def test_bold_stripped(self):
        assert self._call("**bold**") == "bold"

    def test_italic_stripped(self):
        assert self._call("*italic*") == "italic"

    def test_triple_bold_italic_stripped(self):
        assert self._call("***bold italic***") == "bold italic"

    def test_bold_with_spaces(self):
        assert self._call("some **bold** text") == "some bold text"

    def test_italic_with_spaces(self):
        assert self._call("some *italic* text") == "some italic text"

    # --- Must NOT strip: literal asterisks ---

    def test_cron_expression_preserved(self):
        assert self._call("cron: 30 8 * * 1-5") == "cron: 30 8 * * 1-5"

    def test_cron_every_minute_preserved(self):
        assert self._call("* * * * *") == "* * * * *"

    def test_math_multiplication_preserved(self):
        assert self._call("5 * 3 * 2 = 30") == "5 * 3 * 2 = 30"

    def test_single_multiplication_preserved(self):
        assert self._call("5 * 3 = 15") == "5 * 3 = 15"

    def test_asterisks_with_only_whitespace_inside(self):
        assert self._call("* * *") == "* * *"

    def test_mixed_markdown_and_literal(self):
        result = self._call("**bold** and *italic* vs 5 * 3 = 15")
        assert result == "bold and italic vs 5 * 3 = 15"

    def test_literal_asterisk_between_words(self):
        assert self._call("foo * bar * baz") == "foo * bar * baz"

    def test_empty_emphasis_not_stripped(self):
        assert self._call("**") == "**"

    def test_single_word_emphasis(self):
        assert self._call("*word*") == "word"

    def test_multichar_emphasis(self):
        assert self._call("*hello world*") == "hello world"

    # --- Truncation ---

    def test_truncation_at_4000(self):
        text = "x" * 5000
        assert len(self._call(text)) == 4000
