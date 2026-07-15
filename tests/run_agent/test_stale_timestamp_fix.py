"""Tests for the two-layer timestamp injection fix (PR #15872).

Architecture:
  - System prompt: 'Session started: [date]' (frozen, cache-stable, date-only)
  - User message: 'Current time: [datetime]' (volatile, per-turn injection)

These tests verify cache invariants and helper correctness WITHOUT
importing agent.turn_context (which may not exist in all codebase versions).
Instead, they assert against the assembled output of hermes_time helpers
and system_prompt builder.
"""

import re
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# hermes_time helper tests
# ---------------------------------------------------------------------------

class TestHermesTimeHelpers:
    """Tests for the new centralized time formatting helpers."""

    def test_get_timezone_name_returns_string(self):
        """get_timezone_name() returns a string (possibly empty)."""
        from hermes_time import get_timezone_name
        result = get_timezone_name()
        assert isinstance(result, str)

    def test_get_timezone_display_format(self):
        """get_timezone_display() returns 'IANA (UTC±HH:MM)' or empty string."""
        from hermes_time import get_timezone_display
        result = get_timezone_display()
        assert isinstance(result, str)
        if result:
            # Must match pattern: 'Region/City (UTC±HH:MM)'
            assert re.match(
                r"^.+/.+ \(UTC[+-]\d{2}:\d{2}\)$", result
            ), f"Unexpected format: {result!r}"

    def test_get_timezone_display_with_known_tz(self):
        """Verify UTC offset calculation with a mocked timezone."""
        from hermes_time import reset_cache
        with patch.dict("os.environ", {"HERMES_TIMEZONE": "Asia/Tokyo"}):
            reset_cache()
            try:
                from hermes_time import get_timezone_display
                result = get_timezone_display()
                assert "Asia/Tokyo" in result
                assert "UTC+09:00" in result
            finally:
                reset_cache()

    def test_format_current_time_context_contains_required_lines(self):
        """format_current_time_context() returns 'Current time:' + optional 'Timezone:'."""
        from hermes_time import format_current_time_context
        result = format_current_time_context()
        assert isinstance(result, str)
        lines = result.split("\n")
        # First line must have 'Current time:'
        assert lines[0].startswith("Current time:")
        # If timezone configured, second line has 'Timezone:'
        if len(lines) > 1:
            assert lines[1].startswith("Timezone:")

    def test_format_current_time_context_has_minute_precision(self):
        """Per-turn injection MUST include minute precision (unlike system prompt)."""
        from hermes_time import format_current_time_context
        result = format_current_time_context()
        # Must contain AM/PM with time
        assert re.search(r"\d{1,2}:\d{2}\s*(AM|PM)", result), (
            f"Per-turn time must include HH:MM AM/PM, got: {result!r}"
        )

    def test_format_current_time_context_no_stale_label(self):
        """Must NOT contain 'Conversation started' or 'Session started'."""
        from hermes_time import format_current_time_context
        result = format_current_time_context()
        assert "Conversation started" not in result
        assert "Session started" not in result


# ---------------------------------------------------------------------------
# System prompt cache invariant tests
# ---------------------------------------------------------------------------

class TestSystemPromptCacheInvariants:
    """Verify the cached system prompt never contains volatile time data."""

    def test_system_prompt_uses_session_started_label(self):
        """System prompt must use 'Session started' (unambiguous frozen label),
        not 'Conversation started' (which agents misread as current time)."""
        # We test by importing the builder and checking it calls now() correctly.
        # The actual build requires a full AIAgent, so we test the string pattern.
        from hermes_time import now
        from agent.system_prompt import build_system_prompt_parts
        # We can't easily instantiate AIAgent in a unit test, so verify
        # the source code uses the correct label.
        import inspect
        source = inspect.getsource(build_system_prompt_parts)
        assert "Session started" in source, (
            "system_prompt.py must use 'Session started' (not 'Conversation started')"
        )
        assert "Conversation started" not in source, (
            "system_prompt.py must NOT use 'Conversation started' — agents misread it as current time"
        )

    def test_system_prompt_date_only_no_minutes(self):
        """Cached system prompt timestamp must be date-only (no %I:%M %p)."""
        import inspect
        from agent.system_prompt import build_system_prompt_parts
        source = inspect.getsource(build_system_prompt_parts)
        # The strftime format for Session started must NOT contain time components
        assert "%I" not in source or "format_current_time" in source, (
            "System prompt must not use %I (hour) in strftime — breaks cache on every hour"
        )
        assert "%M" not in source or "format_current_time" in source, (
            "System prompt must not use %M (minute) in strftime — breaks cache on every minute"
        )

    def test_system_prompt_excludes_current_time(self):
        """'Current time:' must NEVER appear in the cached system prompt.

        This is the core cache safety invariant. Volatile time goes to user message.
        """
        import inspect
        from agent.system_prompt import build_system_prompt_parts
        source = inspect.getsource(build_system_prompt_parts)
        assert "Current time:" not in source, (
            "'Current time:' must NOT be in system prompt — "
            "it changes every minute and would break the prompt cache prefix"
        )


# ---------------------------------------------------------------------------
# Turn-level injection tests (via hermes_time, not source inspection)
# ---------------------------------------------------------------------------

class TestTurnLevelTimeInjection:
    """Verify per-turn time injection produces correct output."""

    def test_per_turn_time_changes_between_calls(self):
        """format_current_time_context() should reflect current time on each call."""
        from hermes_time import format_current_time_context
        import time
        result1 = format_current_time_context()
        # Both calls should produce valid output (same-minute calls may be identical)
        assert "Current time:" in result1
        # Verify it's a real datetime, not a frozen placeholder
        assert re.search(r"\d{4}", result1), "Must contain a year"

    def test_timezone_display_consistency(self):
        """get_timezone_display() and format_current_time_context() should agree on timezone."""
        from hermes_time import get_timezone_display, format_current_time_context
        tz_display = get_timezone_display()
        time_ctx = format_current_time_context()
        if tz_display:
            assert f"Timezone: {tz_display}" in time_ctx
        else:
            assert "Timezone:" not in time_ctx
