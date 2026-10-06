"""Tests for gap.py — direction-aware progress, pace, milestones, rollup."""

from __future__ import annotations

from datetime import datetime, timezone

from _common import GoalDoc
from gap import compute_gap, count_milestones, rollup

# A 10-day window with `now` at day 5 -> elapsed fraction 0.5.
WINDOW = dict(start_date="2026-01-01", target_date="2026-01-11")
NOW = datetime(2026, 1, 6, 12, 0, 0, tzinfo=timezone.utc)


def _goal(**kw) -> GoalDoc:
    base = dict(domain="d", goal="g")
    base.update(kw)
    return GoalDoc(**base)


# =============================================================================
# Progress direction
# =============================================================================

class TestProgress:
    def test_increase_from_zero_baseline(self):
        g = compute_gap(_goal(direction="increase", current_value=7, target_value=10), now=NOW)
        assert g.progress_pct == 70

    def test_increase_with_baseline(self):
        # 9 -> 15, currently 12 -> (12-9)/(15-9) = 0.5
        g = compute_gap(_goal(direction="increase", baseline_value=9, current_value=12, target_value=15), now=NOW)
        assert g.progress_pct == 50

    def test_decrease_with_baseline(self):
        # 72 -> 60, currently 66 -> (72-66)/(72-60) = 0.5
        g = compute_gap(_goal(direction="decrease", baseline_value=72, current_value=66, target_value=60), now=NOW)
        assert g.progress_pct == 50

    def test_decrease_without_baseline_fallback(self):
        # target/current = 60/66 ~= 0.909
        g = compute_gap(_goal(direction="decrease", current_value=66, target_value=60), now=NOW)
        assert g.progress_pct == 91

    def test_direction_inferred_when_omitted(self):
        # target below current -> inferred decrease
        g = compute_gap(_goal(current_value=66, target_value=60, baseline_value=72), now=NOW)
        assert g.direction == "decrease" and g.progress_pct == 50

    def test_maintain_in_band(self):
        g = compute_gap(_goal(direction="maintain", current_value=100, target_value=100), now=NOW)
        assert g.progress_pct == 100

    def test_remaining_increase(self):
        g = compute_gap(_goal(current_value=12, target_value=15), now=NOW)
        assert g.remaining == 3  # 3 more to gain

    def test_large_numbers_get_thousands_separators(self):
        s = compute_gap(_goal(direction="increase", current_value=3200, target_value=10000, unit="USD"), now=NOW).summary
        assert "3,200 USD" in s and "10,000 USD" in s

    def test_remaining_decrease_is_direction_aware(self):
        # codex P2: decrease goal at 190 toward 180 has 10 to go, not -10
        g = compute_gap(_goal(direction="decrease", current_value=190, target_value=180), now=NOW)
        assert g.remaining == 10

    def test_remaining_negative_on_overshoot(self):
        g = compute_gap(_goal(direction="decrease", current_value=175, target_value=180), now=NOW)
        assert g.remaining == -5  # 5 past target


# =============================================================================
# Pace classification (elapsed = 0.5 in this window)
# =============================================================================

class TestPace:
    def _pace(self, current: float) -> str:
        g = _goal(direction="increase", current_value=current, target_value=10, **WINDOW)
        return compute_gap(g, now=NOW).pace

    def test_ahead(self):
        assert self._pace(7) == "ahead"      # progress 0.7 vs elapsed 0.5

    def test_on_track(self):
        assert self._pace(5) == "on_track"   # 0.5 vs 0.5

    def test_behind(self):
        assert self._pace(3) == "behind"     # 0.3 vs 0.5

    def test_met(self):
        assert self._pace(10) == "met"       # progress 1.0

    def test_unknown_without_dates(self):
        g = _goal(direction="increase", current_value=5, target_value=10)
        assert compute_gap(g, now=NOW).pace == "unknown"

    def test_days_left_and_overdue(self):
        ahead = compute_gap(_goal(current_value=5, target_value=10, **WINDOW), now=NOW)
        assert ahead.days_left == 5
        past = datetime(2026, 1, 20, tzinfo=timezone.utc)
        overdue = compute_gap(_goal(current_value=5, target_value=10, **WINDOW), now=past)
        assert overdue.days_left == -9


# =============================================================================
# Milestones
# =============================================================================

class TestMilestones:
    def test_count(self):
        body = "## M\n- [x] a\n- [ ] b\n* [X] c\n- [ ] d\n"
        assert count_milestones(body) == (2, 4)

    def test_progress_from_milestones_when_no_target(self):
        g = _goal(body="- [x] a\n- [x] b\n- [ ] c\n")
        res = compute_gap(g, now=NOW)
        assert not res.quantifiable
        assert res.milestones_done == 2 and res.milestones_total == 3
        assert res.progress_pct == 67

    def test_no_milestones_no_target(self):
        res = compute_gap(_goal(), now=NOW)
        assert res.progress is None
        assert "no measurable target" in res.summary

    def test_quantifiable_summary_includes_milestones(self):
        # dogfood finding: a numeric goal's persisted milestones (the proposed
        # "path") must stay visible in gap/report, not only non-numeric goals.
        g = _goal(direction="increase", current_value=11, target_value=15,
                  body="- [x] a\n- [ ] b\n- [ ] c\n", **WINDOW)
        res = compute_gap(g, now=NOW)
        assert res.quantifiable
        assert "1/3 milestones" in res.summary and "%" in res.summary


# =============================================================================
# Terminal statuses
# =============================================================================

class TestTerminalStatus:
    def test_achieved_reports_itself(self):
        res = compute_gap(_goal(status="achieved", current_value=5, target_value=10), now=NOW)
        assert res.status == "achieved" and res.summary == "achieved"

    def test_paused_keeps_milestones(self):
        res = compute_gap(_goal(status="paused", body="- [x] a\n- [ ] b\n"), now=NOW)
        assert res.status == "paused" and "1/2" in res.summary


# =============================================================================
# Roll-up
# =============================================================================

class TestRollup:
    def test_groups_and_flags(self):
        behind = (_goal(domain="finance", goal="Save", current_value=3, target_value=10, **WINDOW),)
        met = (_goal(domain="finance", goal="Debt", current_value=10, target_value=10, **WINDOW),)
        health = (_goal(domain="health", goal="HR", status="achieved"),)
        pairs = [(g[0], compute_gap(g[0], now=NOW)) for g in (behind, met, health)]
        rolls = {r.domain: r for r in rollup(pairs)}
        assert rolls["finance"].total == 2
        assert rolls["finance"].behind == ["Save"]
        assert rolls["finance"].ready_to_close == ["Debt"]
        assert rolls["health"].by_status == {"achieved": 1}

    def test_sorted_by_domain(self):
        docs = [_goal(domain="z", goal="1"), _goal(domain="a", goal="2")]
        pairs = [(d, compute_gap(d, now=NOW)) for d in docs]
        assert [r.domain for r in rollup(pairs)] == ["a", "z"]
