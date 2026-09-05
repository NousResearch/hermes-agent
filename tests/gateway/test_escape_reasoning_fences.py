"""
Tests for escape_code_fences_for_display.

B1: Escape triple-backtick markers inside reasoning text before wrapping
    in an outer ``` fence, so inner ``` doesn't break the outer block.
"""

import pytest
from gateway.stream_consumer import escape_code_fences_for_display


class TestEscapeCodeFencesForDisplay:
    """escape_code_fences_for_display prevents inner ``` from breaking
    the outer code block used to render reasoning."""


    def test_single_fence_escaped(self):
        text = "model used ```python\nx = 1\n``` in its thinking"
        result = escape_code_fences_for_display(text)
        assert "```" not in result
        assert "\\`\\`\\`" in result

    def test_multiple_fences_all_escaped(self):
        text = "```\nblock1\n``` and ```python\nblock2\n```"
        result = escape_code_fences_for_display(text)
        assert result.count("```") == 0
        assert result.count("\\`\\`\\`") == 4


