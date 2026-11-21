"""Tests for the rework detector."""

from __future__ import annotations

import time

from agent.signal_sources.rework_detector import (
    ReworkEvent,
    count_recent,
    count_rework_retries,
    filter_window,
)


NOW = 1_700_000_000.0


def _e(task_id: str, seconds_ago: float) -> ReworkEvent:
    return ReworkEvent(task_id=task_id, timestamp=NOW - seconds_ago)


class TestFilterWindow:
    def test_keeps_recent(self):
        events = [_e("t1", 60), _e("t1", 600), _e("t1", 700)]
        kept = filter_window(events, now=NOW, window_sec=600)
        # 700 seconds ago is outside the 600s window.
        assert len(kept) == 2

    def test_empty(self):
        assert filter_window([], now=NOW, window_sec=600) == []


class TestCountRecent:
    def test_counts_target_in_window(self):
        events = [_e("t1", 30), _e("t1", 90), _e("t2", 30)]
        assert count_recent(events, "t1", now=NOW, window_sec=600) == 2

    def test_excludes_outside_window(self):
        events = [_e("t1", 30), _e("t1", 700)]
        assert count_recent(events, "t1", now=NOW, window_sec=600) == 1

    def test_no_match(self):
        events = [_e("t2", 30)]
        assert count_recent(events, "t1", now=NOW, window_sec=600) == 0

    def test_zero_window_disables(self):
        events = [_e("t1", 0.1)]
        assert count_recent(events, "t1", now=NOW, window_sec=0) == 0


class TestCountReworkRetries:
    def test_subtracts_current(self):
        """The current attempt is in the input list; retries = count - 1."""
        events = [_e("t1", 30), _e("t1", 90), _e("t1", 150)]
        # Three t1 events in window; current + 2 retries.
        assert count_rework_retries(events, "t1", now=NOW, window_sec=600) == 2

    def test_no_retries(self):
        events = [_e("t1", 30)]  # only the current attempt
        assert count_rework_retries(events, "t1", now=NOW, window_sec=600) == 0

    def test_clamped_at_zero(self):
        # Defensive: never return negative even if input is weird.
        events = []
        assert count_rework_retries(events, "t1", now=NOW, window_sec=600) == 0


class TestNowDefault:
    def test_uses_time_time_when_no_now(self, monkeypatch):
        # Patch time.time to return NOW.
        monkeypatch.setattr(time, "time", lambda: NOW)
        events = [_e("t1", 30)]
        assert count_recent(events, "t1", window_sec=600) == 1
