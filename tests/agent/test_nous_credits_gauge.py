"""Tests for the Nous-credits subscription % gauge in build_nous_credits_snapshot.

Covers the monthly_credits denominator path added when the portal /api/oauth/account
subscription block began carrying `monthly_credits`. Magnitudes-only fallback, clamp,
and the non-finite / rollover guards (surfaced by adversarial review) are all asserted.
"""
from hermes_cli.nous_account import (
    NousPortalAccountInfo,
    NousPaidServiceAccessInfo,
    NousPortalSubscriptionInfo,
    _subscription_from_payload,
)
from agent.account_usage import build_nous_credits_snapshot, render_account_usage_lines


def _acct(**kwargs):
    kwargs.setdefault("logged_in", True)
    kwargs.setdefault("source", "account_api")
    kwargs.setdefault("fresh", True)
    kwargs.setdefault("portal_base_url", "https://portal.nousresearch.com")
    return NousPortalAccountInfo(**kwargs)


def _window(snap):
    return snap.windows[0] if (snap and snap.windows) else None


def test_parser_captures_monthly_credits():
    sub = _subscription_from_payload({
        "plan": "Ultra", "tier": 14, "monthly_charge": 200, "monthly_credits": 220,
        "current_period_end": "2026-06-28T05:21:54.000Z",
        "credits_remaining": 219.27341839, "rollover_credits": 0,
    })
    assert sub.monthly_credits == 220
    assert abs(sub.credits_remaining - 219.27341839) < 1e-6


def test_parser_monthly_credits_absent_is_none():
    sub = _subscription_from_payload({"plan": "Ultra", "credits_remaining": 10.0})
    assert sub.monthly_credits is None


def test_gauge_present_with_monthly_credits():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(
            plan="Ultra", monthly_credits=220, credits_remaining=219.27341839,
            current_period_end="2026-06-28"),
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=219.27, total_usable_credits=219.27),
    ))
    w = _window(snap)
    assert w is not None and w.label == "Subscription"
    assert abs(w.used_percent - (220 - 219.27341839) / 220 * 100) < 1e-9
    blob = "\n".join(render_account_usage_lines(snap))
    assert "% used" in blob or "% remaining" in blob
    assert "of $220.00 left" in blob


def test_gauge_90pct():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(monthly_credits=220, credits_remaining=22.0),
    ))
    assert abs(_window(snap).used_percent - 90.0) < 1e-9


def test_gauge_debt_clamps_to_100():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=False,
        subscription=NousPortalSubscriptionInfo(monthly_credits=220, credits_remaining=-5.0),
        paid_service_access_info=NousPaidServiceAccessInfo(subscription_credits_remaining=-5.0),
    ))
    assert _window(snap).used_percent == 100.0


def test_gauge_at_cap_is_zero_used():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(monthly_credits=220, credits_remaining=220.0),
    ))
    assert _window(snap).used_percent == 0.0


def test_no_monthly_credits_falls_back_to_magnitudes():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(plan="Ultra", credits_remaining=-0.79),
        paid_service_access_info=NousPaidServiceAccessInfo(purchased_credits_remaining=991.96),
    ))
    assert _window(snap) is None
    blob = "\n".join(render_account_usage_lines(snap))
    assert "%" not in blob
    assert "Top-up credits: $991.96" in blob


def test_nan_remaining_no_window_no_nan_string():
    """json.loads parses bare NaN by default; isinstance(nan, float) is True.
    The gauge must reject it rather than render '$nan' + a false 100% used."""
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(monthly_credits=220, credits_remaining=float("nan")),
        paid_service_access_info=NousPaidServiceAccessInfo(purchased_credits_remaining=5.0),
    ))
    assert _window(snap) is None
    assert "$nan" not in "\n".join(render_account_usage_lines(snap)).lower()


def test_inf_cap_no_window():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(monthly_credits=float("inf"), credits_remaining=10.0),
        paid_service_access_info=NousPaidServiceAccessInfo(purchased_credits_remaining=5.0),
    ))
    assert _window(snap) is None


def test_rollover_balance_exceeds_cap_no_window():
    """remaining > cap (rollover spanning the period) makes monthly_credits a
    nonsensical denominator → suppress the gauge, keep magnitudes."""
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(monthly_credits=220, credits_remaining=300, rollover_credits=80),
        paid_service_access_info=NousPaidServiceAccessInfo(subscription_credits_remaining=300.0),
    ))
    assert _window(snap) is None
    assert "of $220.00 left" not in "\n".join(render_account_usage_lines(snap))


def test_bool_monthly_credits_no_window():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(monthly_credits=True, credits_remaining=1.0),
        paid_service_access_info=NousPaidServiceAccessInfo(purchased_credits_remaining=5.0),
    ))
    assert _window(snap) is None


def test_zero_monthly_credits_no_divzero():
    snap = build_nous_credits_snapshot(_acct(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(monthly_credits=0, credits_remaining=0.0),
        paid_service_access_info=NousPaidServiceAccessInfo(purchased_credits_remaining=5.0),
    ))
    assert _window(snap) is None


def test_failopen_none_and_logged_out():
    assert build_nous_credits_snapshot(None) is None
    assert build_nous_credits_snapshot(_acct(logged_in=False)) is None
