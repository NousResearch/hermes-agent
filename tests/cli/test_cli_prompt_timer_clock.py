"""Regression tests for the per-prompt status-bar timer formatter.

The per-prompt elapsed timer is now driven by time.monotonic() instead of
time.time(), so it no longer jumps when the wall clock is adjusted (NTP step,
manual change) or across a suspend/resume. _format_prompt_elapsed therefore
interprets a live prompt_start_time as a monotonic timestamp.
"""

import time

from cli import HermesCLI

fmt = HermesCLI._format_prompt_elapsed


class TestFormatPromptElapsed:
    def test_fresh_start_shows_zero(self):
        assert fmt(None, 0.0) == "⏲ 0s"

    def test_frozen_duration_seconds(self):
        assert "5s" in fmt(None, 5.0)

    def test_frozen_duration_minutes(self):
        out = fmt(None, 65.0)
        assert "1m" in out and "5s" in out

    def test_frozen_duration_hours(self):
        out = fmt(None, 3661.0)  # 1h 0m 1s
        assert "1h" in out

    def test_live_start_is_interpreted_as_monotonic(self):
        # A live prompt with a start time 2s in the monotonic past must render
        # ~2s elapsed. If the formatter still subtracted from time.time(), a
        # monotonic start (a much smaller number than epoch seconds) would
        # produce a huge bogus value.
        start = time.monotonic() - 2.0
        out = fmt(start, 0.0, live=True)
        assert "2s" in out
        # sanity: not an absurd wall-clock-vs-monotonic mismatch value
        assert "h" not in out and "d" not in out

    def test_negative_is_clamped_to_zero(self):
        # Start slightly in the future (clock-edge race) clamps to 0s, never
        # renders a negative timer.
        start = time.monotonic() + 5.0
        out = fmt(start, 0.0, live=True)
        assert "0s" in out
        assert "-" not in out
