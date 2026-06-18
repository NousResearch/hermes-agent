"""
Tests for reasoning max_lines config.

B2: Replace hardcoded 15-line (gateway) / 5-line (CLI) reasoning truncation
with config option display.reasoning_max_lines.
"""


def max_lines_for_reasoning(text: str, max_lines: int = 0) -> str:
    """Truncate reasoning to max_lines.  max_lines=0 means no limit."""
    stripped = text.strip()
    if max_lines <= 0 or not stripped:
        return stripped
    lines = stripped.splitlines()
    if len(lines) <= max_lines:
        return stripped
    shown = "\n".join(lines[:max_lines])
    remainder = len(lines) - max_lines
    return f"{shown}\n_... ({remainder} more lines)_"


class TestMaxLinesForReasoning:
    """max_lines_for_reasoning truncates with ... more lines indicator."""

    def test_no_limit_returns_full(self):
        text = "line1\nline2\nline3"
        assert max_lines_for_reasoning(text, 0) == text

    def test_under_limit_passthrough(self):
        text = "line1\nline2"
        assert max_lines_for_reasoning(text, 5) == text.strip()

    def test_over_limit_truncated(self):
        text = "line1\nline2\nline3\nline4\nline5\nline6"
        result = max_lines_for_reasoning(text, 3)
        assert result == "line1\nline2\nline3\n_... (3 more lines)_"

    def test_exactly_at_limit(self):
        text = "line1\nline2\nline3"
        result = max_lines_for_reasoning(text, 3)
        assert result == "line1\nline2\nline3"
        assert "_..." not in result

    def test_empty_string(self):
        assert max_lines_for_reasoning("", 5) == ""

    def test_negative_limit_treated_as_no_limit(self):
        text = "line1\nline2"
        assert max_lines_for_reasoning(text, -1) == text

    def test_single_line(self):
        assert max_lines_for_reasoning("hello", 1) == "hello"

    def test_trailing_newlines_stripped(self):
        text = "line1\nline2\n\n\n"
        assert max_lines_for_reasoning(text, 5) == "line1\nline2"

    def test_strips_leading_trailing_whitespace(self):
        text = "  \nline1\nline2\n  "
        assert max_lines_for_reasoning(text, 5) == "line1\nline2"
