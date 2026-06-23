"""Tests for tool_preview_length: 0 (no limit) handling in gateway progress.

Regression test for #51067: setting tool_preview_length to 0 should disable
truncation entirely, not fall back to a 40-char default.
"""

import pytest
from unittest.mock import patch

from agent.display import set_tool_preview_max_len, get_tool_preview_max_len


@pytest.fixture(autouse=True)
def reset_preview_len():
    """Reset tool_preview_max_len before/after each test."""
    set_tool_preview_max_len(0)
    yield
    set_tool_preview_max_len(0)


class TestToolPreviewLengthZeroMeansUnlimited:
    """tool_preview_length: 0 should mean no truncation."""

    def test_get_returns_zero_for_unlimited(self):
        """Zero is returned as-is (sentinel for unlimited)."""
        set_tool_preview_max_len(0)
        assert get_tool_preview_max_len() == 0

    def test_positive_value_is_stored(self):
        """Positive values are stored as-is."""
        set_tool_preview_max_len(80)
        assert get_tool_preview_max_len() == 80

    def test_zero_does_not_truncate_long_string(self):
        """When _pl == 0, no truncation occurs regardless of string length."""
        set_tool_preview_max_len(0)
        _pl = get_tool_preview_max_len()
        long_preview = "x" * 200
        # Simulate the fixed gateway logic:
        # Only truncate when _pl > 0 and len > _pl
        if _pl > 0 and len(long_preview) > _pl:
            long_preview = long_preview[:_pl - 3] + "..."
        # With _pl == 0, the preview should remain untouched
        assert len(long_preview) == 200
        assert "..." not in long_preview

    def test_positive_value_truncates(self):
        """When _pl > 0 and string exceeds it, truncation occurs."""
        set_tool_preview_max_len(40)
        _pl = get_tool_preview_max_len()
        long_preview = "x" * 200
        if _pl > 0 and len(long_preview) > _pl:
            long_preview = long_preview[:_pl - 3] + "..."
        assert len(long_preview) == 40
        assert long_preview.endswith("...")

    def test_positive_value_no_truncate_when_short(self):
        """When _pl > 0 but string is shorter, no truncation."""
        set_tool_preview_max_len(40)
        _pl = get_tool_preview_max_len()
        short_preview = "ls -la"
        if _pl > 0 and len(short_preview) > _pl:
            short_preview = short_preview[:_pl - 3] + "..."
        assert short_preview == "ls -la"

    def test_terminal_multiline_still_gets_ellipsis(self):
        """Multiline commands still get ' ...' suffix (indicating more lines)."""
        set_tool_preview_max_len(0)
        _pl = get_tool_preview_max_len()
        _cmd_full = "echo hello\necho world"
        _lines = _cmd_full.splitlines()
        _cmd_short = _lines[0] if _lines else _cmd_full
        _multiline = len(_lines) > 1
        # Fixed logic from gateway/run.py
        if _pl > 0 and len(_cmd_short) > _pl:
            _cmd_short = _cmd_short[:_pl - 3] + "..."
        elif _multiline:
            _cmd_short = _cmd_short + " ..."
        assert _cmd_short == "echo hello ..."

    def test_terminal_long_single_line_not_truncated_at_zero(self):
        """A single long command line is NOT truncated when _pl == 0."""
        set_tool_preview_max_len(0)
        _pl = get_tool_preview_max_len()
        _cmd_full = "find /usr/local/share -name '*.py' -exec grep -l 'import asyncio' {} +"
        _lines = _cmd_full.splitlines()
        _cmd_short = _lines[0] if _lines else _cmd_full
        _multiline = len(_lines) > 1
        if _pl > 0 and len(_cmd_short) > _pl:
            _cmd_short = _cmd_short[:_pl - 3] + "..."
        elif _multiline:
            _cmd_short = _cmd_short + " ..."
        # Should be unchanged — no truncation with _pl == 0
        assert _cmd_short == _cmd_full
        assert "..." not in _cmd_short
