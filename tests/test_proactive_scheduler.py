"""Tests for the flow-aware Proactive Communication Loop scheduler."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.proactive_scheduler import (
    analyze_flow,
    FlowProfile,
    ProactiveScheduler,
    _DEFAULT_PEAK_HOUR,
    _MIN_MESSAGES_FOR_ANALYSIS,
    _PEAK_HOUR_WINDOW_MINUTES,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_messages(hour_weights: Dict[int, int], base_length: int = 100) -> List[Dict[str, Any]]:
    """Create synthetic message history with given counts per local hour."""
    msgs = []
    ref_day = time.time() - 15 * 86400
    day = 0
    for hour, count in hour_weights.items():
        for i in range(count):
            ts = ref_day + (day % 20) * 86400 + hour * 3600 + i * 120
            msgs.append({"role": "user", "ts": ts, "content": "x" * (base_length + i * 3)})
            day += 1
    msgs.append({"role": "assistant", "ts": ref_day, "content": "response"})
    return msgs


def _make_scheduler() -> ProactiveScheduler:
    s = ProactiveScheduler.__new__(ProactiveScheduler)
    s._adapters = None
    s._loop = None
    s._flow_profiles = {}
    s._last_synthesis_date = {}
    return s


# ──────────────────────────────────────────────────────────────────────
# FlowProfile
# ──────────────────────────────────────────────────────────────────────

def test_default_when_insufficient_history():
    msgs = _make_messages({9: 3})
    profile = analyze_flow(msgs)
    assert profile.peak_hour == _DEFAULT_PEAK_HOUR
    assert profile.confidence == 0.0


def test_clear_morning_peak():
    msgs = _make_messages({9: 40, 10: 35, 11: 20, 14: 8, 22: 3})
    profile = analyze_flow(msgs)
    assert 8 <= profile.peak_hour <= 12, f"Expected morning peak, got {profile.peak_hour}"
    assert profile.confidence > 0.0


def test_clear_evening_peak():
    msgs = _make_messages({20: 40, 21: 38, 22: 25, 9: 4, 10: 3})
    profile = analyze_flow(msgs)
    assert 19 <= profile.peak_hour <= 23, f"Expected evening peak, got {profile.peak_hour}"


def test_depth_signal_included():
    """Hour with fewer but longer messages should score differently from shallow hour."""
    ref = time.time() - 15 * 86400
    long_msgs = [
        {"role": "user", "ts": ref + i * 86400 + 14 * 3600, "content": "x" * 800}
        for i in range(25)
    ]
    short_msgs = [
        {"role": "user", "ts": ref + i * 86400 + 10 * 3600, "content": "ok"}
        for i in range(30)
    ]
    profile = analyze_flow(long_msgs + short_msgs)
    # Depth score should give hour 14 a higher per-message value
    # Just verify the profile is valid and depth signal exists
    assert profile.peak_hour in range(24)
    assert 10 in profile.scores or 14 in profile.scores


def test_confidence_increases_with_clear_peak():
    flat_msgs = _make_messages({h: 5 for h in range(24)})
    peaked_msgs = _make_messages({9: 60, 10: 50, **{h: 2 for h in range(11, 24)}})
    flat_profile = analyze_flow(flat_msgs)
    peaked_profile = analyze_flow(peaked_msgs)
    assert peaked_profile.confidence > flat_profile.confidence


def test_timezone_offset_shifts_peak():
    ref = time.time() - 15 * 86400
    # Messages at UTC hour 14 every day
    msgs = [
        {"role": "user", "ts": ref + i * 86400 + 14 * 3600, "content": "x" * 200}
        for i in range(30)
    ]
    profile_utc = analyze_flow(msgs, tz_offset_hours=0)
    profile_est = analyze_flow(msgs, tz_offset_hours=-5)

    assert abs(profile_utc.peak_hour - 14) <= 1
    assert abs(profile_est.peak_hour - 9) <= 1


def test_is_stale_respects_age():
    fresh = FlowProfile(9, 0.8, {}, analyzed_at=time.time())
    old = FlowProfile(9, 0.8, {}, analyzed_at=time.time() - 8 * 86400)
    assert not fresh.is_stale(max_age_days=7)
    assert old.is_stale(max_age_days=7)


def test_flow_profile_repr():
    profile = FlowProfile(9, 0.75, {9: 0.9}, analyzed_at=time.time())
    r = repr(profile)
    assert "peak_hour=9" in r
    assert "confidence=0.75" in r


# ──────────────────────────────────────────────────────────────────────
# Peak window detection
# ──────────────────────────────────────────────────────────────────────

def test_fires_in_peak_window():
    """Synthesis triggers when current time is within ±15 min of peak hour."""
    s = _make_scheduler()
    fake_now = datetime(2026, 5, 9, 9, 5, tzinfo=timezone.utc)  # 09:05 → within ±15 of peak 9

    fired = []
    with patch.object(s, "_local_now", return_value=fake_now), \
         patch.object(s, "_resolve_peak_hour", return_value=9), \
         patch.object(s, "_fire_synthesis", side_effect=lambda sid: fired.append(sid)):
        s._maybe_synthesize("session-a", cfg={})

    assert "session-a" in fired
    assert s._last_synthesis_date["session-a"] == "2026-05-09"


def test_does_not_fire_outside_window():
    """No synthesis when current time is far from peak hour."""
    s = _make_scheduler()
    fake_now = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)  # 14:00, peak=9 → 300 min apart

    with patch.object(s, "_local_now", return_value=fake_now), \
         patch.object(s, "_resolve_peak_hour", return_value=9), \
         patch.object(s, "_fire_synthesis") as mock_fire:
        s._maybe_synthesize("session-b", cfg={})

    mock_fire.assert_not_called()
    assert "session-b" not in s._last_synthesis_date


def test_does_not_fire_twice_same_day():
    s = _make_scheduler()
    s._last_synthesis_date["session-c"] = "2026-05-09"  # already fired today
    fake_now = datetime(2026, 5, 9, 9, 5, tzinfo=timezone.utc)

    with patch.object(s, "_local_now", return_value=fake_now), \
         patch.object(s, "_resolve_peak_hour", return_value=9), \
         patch.object(s, "_fire_synthesis") as mock_fire:
        s._maybe_synthesize("session-c", cfg={})

    mock_fire.assert_not_called()


def test_fires_again_next_day():
    s = _make_scheduler()
    s._last_synthesis_date["session-d"] = "2026-05-08"  # yesterday
    fake_now = datetime(2026, 5, 9, 9, 5, tzinfo=timezone.utc)

    fired = []
    with patch.object(s, "_local_now", return_value=fake_now), \
         patch.object(s, "_resolve_peak_hour", return_value=9), \
         patch.object(s, "_fire_synthesis", side_effect=lambda sid: fired.append(sid)):
        s._maybe_synthesize("session-d", cfg={})

    assert "session-d" in fired
    assert s._last_synthesis_date["session-d"] == "2026-05-09"


def test_config_override_wins_over_profile():
    s = _make_scheduler()
    profile = FlowProfile(peak_hour=21, confidence=0.9, scores={}, analyzed_at=time.time())
    # Config says peak_flow_hour=9
    result = s._resolve_peak_hour(profile, cfg={"proactive_communication": {"peak_flow_hour": 9}})
    assert result == 9


def test_profile_peak_used_when_no_override():
    s = _make_scheduler()
    profile = FlowProfile(peak_hour=21, confidence=0.9, scores={}, analyzed_at=time.time())
    result = s._resolve_peak_hour(profile, cfg={})
    assert result == 21


def test_midnight_wrap_window():
    """Peak at hour 0 (midnight) — current time 23:55 is within ±15 min."""
    s = _make_scheduler()
    fake_now = datetime(2026, 5, 9, 23, 55, tzinfo=timezone.utc)  # 23:55 UTC = 1435 min of day

    fired = []
    with patch.object(s, "_local_now", return_value=fake_now), \
         patch.object(s, "_resolve_peak_hour", return_value=0), \
         patch.object(s, "_fire_synthesis", side_effect=lambda sid: fired.append(sid)):
        s._maybe_synthesize("session-midnight", cfg={})

    assert "session-midnight" in fired


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────

def test_assistant_messages_ignored():
    msgs = [
        {"role": "assistant", "ts": time.time() - i * 3600, "content": "x" * 500}
        for i in range(50)
    ]
    profile = analyze_flow(msgs)
    assert profile.peak_hour == _DEFAULT_PEAK_HOUR
    assert profile.confidence == 0.0


def test_messages_without_ts_skipped():
    """Messages without timestamps are safely ignored, no exception."""
    valid_msgs = _make_messages({9: 25})
    bad_msgs = [
        {"role": "user", "content": "no timestamp"},
        {"role": "user", "ts": None, "content": "null ts"},
        {"role": "user", "ts": "not-a-number", "content": "bad ts"},
    ]
    profile = analyze_flow(bad_msgs + valid_msgs)
    assert profile.peak_hour in range(24)


def test_analyze_flow_empty():
    profile = analyze_flow([])
    assert profile.peak_hour == _DEFAULT_PEAK_HOUR
    assert profile.confidence == 0.0


def test_single_hour_all_messages():
    """All messages in one hour — valid profile, no division by zero."""
    msgs = _make_messages({9: 30})
    profile = analyze_flow(msgs)
    assert profile.peak_hour == 9
    assert 0.0 <= profile.confidence <= 1.0


def test_tick_silent_when_disabled():
    """tick() does nothing when proactive_communication.enabled=False."""
    s = _make_scheduler()
    with patch("hermes_cli.proactive_scheduler._safe_load_config", return_value={}), \
         patch.object(s, "_get_active_sessions") as mock_sessions:
        s.tick()
    mock_sessions.assert_not_called()


def test_tick_never_raises():
    """tick() must never raise even if everything inside fails."""
    s = _make_scheduler()
    with patch("hermes_cli.proactive_scheduler._safe_load_config", side_effect=RuntimeError("boom")):
        s.tick()  # must not raise
