"""Tests for tool_preview_length: 0 (no limit) handling in gateway progress.

Regression test for #51067: setting tool_preview_length to 0 (the documented
"no limit" value) must disable truncation entirely, not fall back to a 40-char
default via a falsy check (``_cap = _pl if _pl > 0 else 40``).

The behavioural regression guards live in tests/gateway/test_run_progress_topics.py,
where they exercise the REAL gateway progress-delivery path via
``_run_long_preview_helper`` and ``TerminalCommandAgent`` so they fail if
``gateway/run.py`` reintroduces the falsy-check fallback:

    - test_all_mode_zero_means_unlimited
        Quoted-preview path: explicit 0 preserves the full preview.
    - test_all_mode_respects_custom_preview_length
        Positive cap still truncates.
    - test_terminal_progress_long_single_line_not_truncated_at_zero
        Fenced code-block terminal path: explicit 0 preserves the full
        single-line command (no ellipsis).

This module keeps only lightweight unit coverage for the ``get/set`` accessor
sentinel — it does NOT re-implement the gateway truncation branch, so it cannot
mask a regression in gateway/run.py.
"""

from agent.display import get_tool_preview_max_len, set_tool_preview_max_len


class TestToolPreviewLengthAccessor:
    """The 0 sentinel round-trips through the display accessor unchanged."""

    def test_zero_round_trips_as_unlimited_sentinel(self):
        """Zero is stored and returned as-is (sentinel for 'no limit')."""
        try:
            set_tool_preview_max_len(0)
            assert get_tool_preview_max_len() == 0
        finally:
            set_tool_preview_max_len(0)

    def test_positive_value_round_trips(self):
        """Positive caps are stored and returned unchanged."""
        try:
            set_tool_preview_max_len(80)
            assert get_tool_preview_max_len() == 80
        finally:
            set_tool_preview_max_len(0)
