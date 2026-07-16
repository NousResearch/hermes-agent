"""Unit tests for the notification / attention budget governor.

The autouse hermetic fixture (tests/conftest.py) points HERMES_HOME at a
per-test tempdir and pins TZ=UTC, so should_deliver()/record_feedback() run
against an isolated state.db and a deterministic day bucket.
"""

from unittest.mock import patch

import pytest

from agent import notification_budget as nb


def _cfg(**overrides):
    """A full notifications config dict with optional per-key overrides."""
    base = {
        "enabled": True,
        "daily_cap": 3,
        "daily_ceiling": 5,
        "base_threshold": 0.5,
        "escalation_threshold": 0.8,
        "dismiss_step": 0.1,
        "act_step": 0.05,
        "threshold_min": 0.1,
        "threshold_max": 0.95,
        "p_act_ewma_alpha": 0.3,
        "default_value_hint": 0.5,
        "categories": {},
    }
    base.update(overrides)
    return base


@pytest.fixture
def cfg_patch():
    """Patch _config() to return a controllable notifications config."""

    def _apply(**overrides):
        return patch.object(nb, "_config", return_value=_cfg(**overrides))

    return _apply


def test_disabled_passes_through(cfg_patch):
    with cfg_patch(enabled=False):
        d = nb.should_deliver("test", value_hint=1.0)
    assert d.allow is True
    assert d.reason == "governor-disabled"


def test_high_value_first_send_allowed(cfg_patch):
    # p_act seed = base_threshold 0.5, value 1.0, cost 0 -> score 0.5 >= thr 0.5
    with cfg_patch():
        d = nb.should_deliver("test", value_hint=1.0)
    assert d.allow is True
    assert d.reason == "under-budget"


def test_low_value_below_threshold_deferred(cfg_patch):
    # Cold-start p_act=1.0 so score == value_hint: 0.4 < threshold 0.5.
    with cfg_patch():
        d = nb.should_deliver("test", value_hint=0.4)
    assert d.allow is False
    assert d.reason == "below-threshold"


def test_hard_ceiling_blocks_even_high_score(cfg_patch):
    # Cold-start p_act=1.0 → score == value_hint, so value 1.0 clears escalation.
    with cfg_patch():
        for i in range(5):  # fill to ceiling with escalation-clearing sends
            r = nb.should_deliver("cat", value_hint=1.0, candidate_id=f"c{i}")
            assert r.allow is True
        d = nb.should_deliver("cat", value_hint=1.0, candidate_id="over")
    assert d.allow is False
    assert d.reason == "hard-ceiling-reached"


def test_soft_cap_low_score_deferred_high_score_allowed(cfg_patch):
    with cfg_patch():
        # Below the soft cap (3): a mid score (>= threshold) is allowed.
        for i in range(3):
            r = nb.should_deliver("cat", value_hint=0.6, candidate_id=f"cap{i}")
            assert r.allow is True
        # At the soft cap: a mid score (0.6 >= thr 0.5 but < escalation 0.8)
        # is deferred.
        mid = nb.should_deliver("cat", value_hint=0.6, candidate_id="mid")
        assert mid.allow is False
        assert mid.reason == "over-soft-cap-low-score"
        # ...but a high score (>= escalation) still spends from the budget.
        hi = nb.should_deliver("cat", value_hint=0.9, candidate_id="hi")
        assert hi.allow is True


def test_idempotent_candidate_not_double_counted(cfg_patch):
    with cfg_patch():
        first = nb.should_deliver("cat", value_hint=1.0, candidate_id="dup")
        second = nb.should_deliver("cat", value_hint=1.0, candidate_id="dup")
    assert first.allow is True
    assert second.reason == "idempotent-replay"
    assert second.allow == first.allow


def test_dismiss_raises_threshold(cfg_patch):
    with cfg_patch():
        nb.record_feedback("cat", "dismiss")
        db = nb._open_db()
        stats = db.get_category_stats("cat")
    # base 0.5 + dismiss_step 0.1 = 0.6
    assert stats["threshold"] == pytest.approx(0.6)
    assert stats["dismiss_count"] == 1


def test_act_lowers_threshold(cfg_patch):
    with cfg_patch():
        nb.record_feedback("cat", "act")
        db = nb._open_db()
        stats = db.get_category_stats("cat")
    # base 0.5 - act_step 0.05 = 0.45
    assert stats["threshold"] == pytest.approx(0.45)
    assert stats["act_count"] == 1


def test_threshold_clamped_to_max(cfg_patch):
    with cfg_patch(dismiss_step=0.5):
        for _ in range(5):
            nb.record_feedback("cat", "dismiss")
        db = nb._open_db()
        stats = db.get_category_stats("cat")
    assert stats["threshold"] <= 0.95


def test_per_category_override(cfg_patch):
    # omi_commitment gets a tiny ceiling; generic keeps the default.
    overrides = {"categories": {"omi_commitment": {"daily_ceiling": 1}}}
    with cfg_patch(**overrides):
        a = nb.should_deliver("omi_commitment", value_hint=1.0, candidate_id="o1")
        b = nb.should_deliver("omi_commitment", value_hint=1.0, candidate_id="o2")
    assert a.allow is True
    assert b.allow is False
    assert b.reason == "hard-ceiling-reached"


def test_zero_ceiling_means_unbounded_not_gagged(cfg_patch):
    # daily_ceiling=0 (and cap=0) must mean "no limit", not "suppress all".
    with cfg_patch(daily_cap=0, daily_ceiling=0):
        results = [
            nb.should_deliver("cat", value_hint=1.0, candidate_id=f"z{i}")
            for i in range(10)
        ]
    assert all(r.allow for r in results)
    assert {r.reason for r in results} == {"under-budget"}


def test_fail_open_on_internal_error(cfg_patch):
    # Force an error deep in the impl; should_deliver must still ALLOW.
    with patch.object(nb, "_config", side_effect=RuntimeError("boom")):
        d = nb.should_deliver("cat", value_hint=0.1)
    assert d.allow is True
    assert d.reason == "governor-error-fail-open"


def test_unknown_feedback_signal_ignored(cfg_patch):
    with cfg_patch():
        nb.record_feedback("cat", "bogus")  # must not raise or write
        db = nb._open_db()
        assert db.get_category_stats("cat") is None


def test_budget_status_reports_usage(cfg_patch):
    with cfg_patch():
        nb.should_deliver("cat", value_hint=1.0, candidate_id="s1")  # allowed
        nb.should_deliver("cat", value_hint=0.1, candidate_id="s2")  # deferred
        status = nb.budget_status()
    assert status["allowed"] == 1
    assert status["cap"] == 3
    assert status["ceiling"] == 5
    assert len(status["deferred"]) == 1
