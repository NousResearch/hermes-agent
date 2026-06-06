"""Tests for _snapshot_from_credits_state — the dev-fixture /usage renderer.

``build_nous_credits_snapshot`` maps a live portal account; ``_snapshot_from_credits_state``
maps a header-shaped CreditsState (e.g. a HERMES_DEV_CREDITS_FIXTURE) into the SAME
/usage snapshot shape, so the gauge + magnitudes are exercisable offline. These lock
the gauge math, the verbatim *_usd magnitudes (never parseFloat'd), the depletion line,
and the dev-fixture marker.
"""
from __future__ import annotations

from agent.account_usage import _snapshot_from_credits_state
from agent.credits_tracker import CreditsState


def _state(**kw) -> CreditsState:
    kw.setdefault("from_header", True)
    return CreditsState(**kw)


def test_renders_gauge_magnitudes_and_fixture_marker():
    # used_fraction = (20 - 10) / 20 = 0.5  → a 50%-used gauge window
    snap = _snapshot_from_credits_state(_state(
        remaining_micros=30_340_000, remaining_usd="30.34",
        subscription_micros=10_000_000, subscription_usd="10.00",
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        purchased_micros=12_340_000, purchased_usd="12.34",
        denominator_kind="subscription_cap", paid_access=True,
    ))
    assert snap is not None and snap.provider == "nous"

    win = next(w for w in snap.windows if w.label == "Subscription")
    assert win.used_percent is not None and abs(win.used_percent - 50.0) < 1e-9
    assert win.detail == "$10.00 of $20.00 left"  # verbatim *_usd strings, not math

    details = list(snap.details)
    assert "Subscription credits: $10.00" in details
    assert "Top-up credits: $12.34" in details
    assert "Total usable: $30.34" in details
    assert any("dev fixture" in d for d in details)  # the offline marker
    assert all("access depleted" not in d for d in details)


def test_depleted_adds_status_line():
    snap = _snapshot_from_credits_state(_state(
        remaining_micros=0, remaining_usd="0.00",
        subscription_micros=0, subscription_usd="0.00",
        purchased_micros=0, purchased_usd="0.00",
        denominator_kind="none", paid_access=False,
    ))
    assert snap is not None
    assert any("access depleted" in d for d in snap.details)


def test_no_cap_yields_no_gauge_window():
    # No subscription cap → used_fraction is None → no gauge window, magnitudes only.
    snap = _snapshot_from_credits_state(_state(
        remaining_micros=5_000_000, remaining_usd="5.00",
        subscription_micros=5_000_000, subscription_usd="5.00",
        subscription_limit_micros=None, denominator_kind="none", paid_access=True,
    ))
    assert snap is not None
    assert all(w.label != "Subscription" for w in snap.windows)
    assert "Total usable: $5.00" in snap.details


def test_none_state_is_safe():
    assert _snapshot_from_credits_state(None) is None
