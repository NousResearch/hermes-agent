"""Regression tests for active-time (wall-clock union) computation.

Kanban t_86210f57: /insights reported "Active time: ~33.0d" over a 30-day
window. Caused by summing per-session durations across parallel sessions
(cron + fleet + kanban + interactive) instead of computing the wall-clock
union of intervals.
"""

from __future__ import annotations

from agent.insights import InsightsEngine


def _make_session(start, end, **extra):
    # NB: offset by a fixed epoch base so start=0 doesn't get filtered by the
    # truthiness check inside _compute_overview ("if start and end and...").
    base_epoch = 1_700_000_000.0
    base = {
        "id": f"s{start}_{end}",
        "source": "cli",
        "model": "claude-opus-4-7",
        "started_at": base_epoch + float(start),
        "ended_at": base_epoch + float(end) if end is not None else None,
        "message_count": 1,
        "tool_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "estimated_cost_usd": 0.0,
        "actual_cost_usd": 0.0,
        "billing_provider": "anthropic",
        "billing_base_url": "",
        "billing_mode": "",
        "cost_status": "included",
        "cost_source": "",
    }
    base.update(extra)
    return base


class TestActiveTimeUnion:
    """_compute_overview computes total_hours from the union of intervals."""

    def _overview(self, sessions, message_stats=None):
        engine = InsightsEngine.__new__(InsightsEngine)
        return InsightsEngine._compute_overview(
            engine, sessions, message_stats or {}
        )

    def test_disjoint_intervals_sum_normally(self):
        # 0–3600, 7200–10800 → 2h union, 2h cumulative
        sessions = [_make_session(0, 3600), _make_session(7200, 10800)]
        o = self._overview(sessions)
        assert abs(o["total_hours"] - 2.0) < 1e-9
        assert abs(o["cumulative_hours"] - 2.0) < 1e-9

    def test_fully_overlapping_intervals_dont_double_count(self):
        # Two identical 1h sessions in parallel → 1h union, 2h cumulative
        sessions = [_make_session(0, 3600), _make_session(0, 3600)]
        o = self._overview(sessions)
        assert abs(o["total_hours"] - 1.0) < 1e-9
        assert abs(o["cumulative_hours"] - 2.0) < 1e-9

    def test_partially_overlapping_intervals_merge(self):
        # 0–3600, 1800–5400 → union 0–5400 = 1.5h, cumulative 2h
        sessions = [_make_session(0, 3600), _make_session(1800, 5400)]
        o = self._overview(sessions)
        assert abs(o["total_hours"] - 1.5) < 1e-9
        assert abs(o["cumulative_hours"] - 2.0) < 1e-9

    def test_nested_interval_subsumed(self):
        # 0–7200 contains 1800–3600 → union 2h, cumulative 2.5h
        sessions = [_make_session(0, 7200), _make_session(1800, 3600)]
        o = self._overview(sessions)
        assert abs(o["total_hours"] - 2.0) < 1e-9
        assert abs(o["cumulative_hours"] - 2.5) < 1e-9

    def test_union_never_exceeds_window(self):
        # Six 8-hour sessions all running in parallel during one 8-hour window.
        # Old code: 6 * 8 = 48h. New code: 8h.
        sessions = [_make_session(0, 8 * 3600) for _ in range(6)]
        o = self._overview(sessions)
        assert abs(o["total_hours"] - 8.0) < 1e-9, (
            f"union should be 8h but got {o['total_hours']}h "
            "(parallel sessions are double-counted)"
        )
        assert abs(o["cumulative_hours"] - 48.0) < 1e-9

    def test_no_sessions_returns_zero(self):
        o = self._overview([])
        assert o["total_hours"] == 0
        assert o["cumulative_hours"] == 0

    def test_negative_durations_ignored(self):
        # ended_at < started_at: clock drift, skip
        sessions = [_make_session(100, 50), _make_session(0, 3600)]
        o = self._overview(sessions)
        assert abs(o["total_hours"] - 1.0) < 1e-9
