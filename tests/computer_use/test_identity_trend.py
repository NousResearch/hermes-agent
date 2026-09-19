"""Identity-retention trend + aggregate reporting contracts (#112734).

Deterministic fixtures only. Each test asserts a behavior contract (accumulate,
direction, aggregation, wiring), never a snapshot.
"""

import pytest

from tools.computer_use import tool as cu_tool
from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.identity_trend import (
    IdentityTrend,
    aggregate_identity_trends,
    get_identity_trend,
    record_identity_step,
    reset_identity_trends_for_tests,
)

W, H = 800, 600


def _el(index, role, label, bounds, **flags):
    return UIElement(index=index, role=role, label=label, bounds=bounds, app="TestApp",
                     pid=4242, window_id=7, attributes=dict(flags))


def _cap(elements):
    return CaptureResult(mode="ax", width=W, height=H, elements=elements,
                         app="TestApp", window_title="Settings")


@pytest.fixture(autouse=True)
def _reset():
    reset_identity_trends_for_tests()
    cu_tool.reset_shadow_state_for_tests()
    yield
    reset_identity_trends_for_tests()
    cu_tool.reset_shadow_state_for_tests()


def _step(sid, retention, **kw):
    record_identity_step(sid, retention=retention, matched=kw.get("matched", 4),
                         removed=kw.get("removed", 0), ambiguous=kw.get("ambiguous", 0),
                         mean_confidence=kw.get("mean_confidence", 1.0),
                         revision=kw.get("revision", 1))


def test_trend_accumulates_steps_and_stats():
    _step("s", 1.0, revision=2)
    _step("s", 0.5, matched=2, removed=2, revision=3)
    s = get_identity_trend("s")
    assert s["steps"] == 2
    assert s["mean_retention"] == pytest.approx(0.75)
    assert s["min_retention"] == pytest.approx(0.5)
    assert s["total_matched"] == 6 and s["total_removed"] == 2


def test_degrading_trend_direction():
    for i, r in enumerate((1.0, 1.0, 0.2, 0.2)):
        _step("s", r, revision=i + 2)
    s = get_identity_trend("s")
    assert s["trend"] == pytest.approx(-0.8)
    assert s["trend_direction"] == "degrading"


def test_improving_trend_direction():
    for i, r in enumerate((0.2, 0.2, 1.0, 1.0)):
        _step("s", r, revision=i + 2)
    s = get_identity_trend("s")
    assert s["trend_direction"] == "improving"


def test_single_step_trend_is_stable():
    _step("s", 0.5, revision=2)
    s = get_identity_trend("s")
    assert s["trend"] == pytest.approx(0.0)
    assert s["trend_direction"] == "stable"


def test_unknown_session_trend_is_empty():
    assert get_identity_trend("nope") == {}


def test_aggregate_across_runs():
    _step("run-a", 1.0, revision=2)
    _step("run-a", 1.0, revision=3)
    _step("run-b", 0.5, revision=2)
    _step("run-b", 0.1, revision=3)
    agg = aggregate_identity_trends()
    assert agg["sessions"] == 2
    assert agg["total_steps"] == 4
    assert agg["mean_retention"] == pytest.approx((1.0 + 0.3) / 2)
    assert agg["min_retention"] == pytest.approx(0.3)
    assert agg["worst_session"] == "run-b"
    assert agg["trend_directions"] == {"improving": 0, "stable": 1, "degrading": 1}
    assert set(agg["per_session"]) == {"run-a", "run-b"}


def test_aggregate_empty_is_well_formed():
    agg = aggregate_identity_trends()
    assert agg["sessions"] == 0 and agg["total_steps"] == 0
    assert agg["mean_retention"] == pytest.approx(1.0)
    assert agg["worst_session"] is None and agg["per_session"] == {}


def test_sessions_are_bounded():
    import tools.computer_use.identity_trend as it
    for i in range(it._MAX_SESSIONS + 5):
        _step(f"sess-{i}", 1.0, revision=2)
    assert len(it._trends) == it._MAX_SESSIONS
    assert get_identity_trend("sess-0") == {}  # oldest evicted first


def test_trend_wire_through_shadow_observe():
    els = [_el(1, "AXButton", "OK", (10, 10, 50, 20))]
    cu_tool._shadow_state_observe(_cap(els), "sess-w")  # first capture: no delta, no step
    assert cu_tool.get_identity_trend("sess-w") == {}
    cu_tool._shadow_state_observe(_cap(els), "sess-w")  # second capture reconciles
    s = cu_tool.get_identity_trend("sess-w")
    assert s["steps"] == 1
    assert s["mean_retention"] == pytest.approx(1.0)
    assert s["trend_direction"] == "stable"
    m = cu_tool.get_shadow_state_metrics("sess-w")
    assert m["identity_retention"] == pytest.approx(s["mean_retention"])


def test_trend_reflects_identity_loss():
    # A merely relabeled control keeps its identity (relabel pass binds same
    # role+parent+geometry at 0.5); true loss is a control disappearing while a
    # different one appears elsewhere — unbound on every pass.
    good = [_el(1, "AXButton", "OK", (10, 10, 50, 20))]
    replaced = [_el(1, "AXButton", "Cancel", (300, 300, 50, 20))]
    cu_tool._shadow_state_observe(_cap(good), "sess-loss")
    cu_tool._shadow_state_observe(_cap(replaced), "sess-loss")
    cu_tool._shadow_state_observe(_cap(replaced), "sess-loss")
    s = cu_tool.get_identity_trend("sess-loss")
    assert s["steps"] == 2
    assert s["mean_retention"] < 1.0
    assert s["trend_direction"] == "improving"  # stable after the churn
