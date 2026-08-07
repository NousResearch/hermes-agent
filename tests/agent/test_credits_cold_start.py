"""Tests for cold-start credits hydration at session open.

The L3 cold-start seed primes agent._credits_state from /api/oauth/account (or a
HERMES_DEV_CREDITS_FIXTURE) so depletion AND the 90% grant warning fire immediately
at session open, not only after the first inference header. These tests assert the
notice policy fires correctly for a seed-shaped CreditsState with the warn90 latch
primed the way conversation_loop does it.
"""
import time

from agent.credits_tracker import CreditsState, evaluate_credits_notices


def _cold_start_notices(state: CreditsState):
    """Mirror the conversation_loop seed: prime seen_below_90 when used_fraction is
    computable (the snapshot IS the first observation), then evaluate once."""
    latch = {"active": set(), "seen_below_90": False}
    if state.used_fraction is not None:
        latch["seen_below_90"] = True
    show, clear = evaluate_credits_notices(state, latch)
    return [n.key for n in show]


def _state(**kw) -> CreditsState:
    kw.setdefault("from_header", False)
    kw.setdefault("captured_at", time.time())
    return CreditsState(**kw)


def test_cold_start_healthy_no_notice():
    s = _state(
        remaining_micros=30_340_000, subscription_micros=18_000_000,
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=True,
    )
    assert abs(s.used_fraction - 0.1) < 1e-9
    assert _cold_start_notices(s) == []












def test_dev_fixtures_drive_cold_start():
    """Every HERMES_DEV_CREDITS_FIXTURE state produces a valid seed CreditsState."""
    import os

    from agent.credits_tracker import dev_fixture_credits_state

    expected = {
        "healthy": [],
        "sub_90pct": ["credits.usage"],
        "depleted": ["credits.depleted"],
    }
    for name, want in expected.items():
        os.environ["HERMES_DEV_CREDITS"] = "1"  # fixtures gate on the dev flag
        os.environ["HERMES_DEV_CREDITS_FIXTURE"] = name
        try:
            fx = dev_fixture_credits_state()
            assert fx is not None, name
            assert _cold_start_notices(fx) == want, (name, _cold_start_notices(fx))
        finally:
            os.environ.pop("HERMES_DEV_CREDITS_FIXTURE", None)
            os.environ.pop("HERMES_DEV_CREDITS", None)


# ── seed_credits_at_session_start: the shared session-open hydrator ───────────


class _FakeAgent:
    """Minimal agent surface for the seed helper: state slots + an emit that runs
    the real policy against the latch (mirroring run_agent._emit_credits_notices,
    including the free-model suppression flag)."""

    def __init__(self, provider="nous", model=""):
        from agent.credits_tracker import evaluate_credits_notices, is_free_tier_model

        self.provider = provider
        self.model = model
        self._credits_state = None
        self._credits_session_start_micros = None
        self._credits_latch = {"active": set(), "seen_below_90": False, "usage_band": None}
        self.emitted: list = []
        self._eval = evaluate_credits_notices
        self._is_free = is_free_tier_model

    def _emit_credits_notices(self):
        if self._credits_state is None:
            return
        show, clear = self._eval(
            self._credits_state,
            self._credits_latch,
            model_is_free=self._is_free(self.model),
        )
        self.emitted.append(([n.key for n in show], clear))


def _seed(agent, fixture):
    import os

    from agent.credits_tracker import seed_credits_at_session_start

    os.environ["HERMES_DEV_CREDITS"] = "1"  # fixtures gate on the dev flag
    os.environ["HERMES_DEV_CREDITS_FIXTURE"] = fixture
    try:
        return seed_credits_at_session_start(agent)
    finally:
        os.environ.pop("HERMES_DEV_CREDITS_FIXTURE", None)
        os.environ.pop("HERMES_DEV_CREDITS", None)












def test_live_crossing_after_seed_still_fires_grant_spent():
    """The gate opens when the session observes the grant NOT yet spent — a healthy
    seed followed by a grant-exhausted header is a real in-session crossing and must
    still announce grant_spent once."""
    a = _FakeAgent()
    assert _seed(a, "healthy") is True
    a.emitted = []
    a._credits_state = _state(  # the grant_exhausted shape, as a live header would carry it
        remaining_micros=12_340_000, subscription_micros=0,
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        purchased_micros=12_340_000, purchased_usd="12.34",
        denominator_kind="subscription_cap", paid_access=True,
    )
    a._emit_credits_notices()
    assert a.emitted == [(["credits.grant_spent"], [])]


def test_seed_is_idempotent():
    a = _FakeAgent()
    _seed(a, "sub_90pct")
    a.emitted = []
    # second call must no-op (state already populated)
    assert _seed(a, "sub_90pct") is False
    assert a.emitted == []


def test_seed_skips_non_nous():
    from agent.credits_tracker import seed_credits_at_session_start

    a = _FakeAgent(provider="openrouter")
    assert seed_credits_at_session_start(a) is False
    assert a._credits_state is None


# ── _credits_state_from_account: account → seed-state field mapping ────────────


def _account(**kwargs):
    from hermes_cli.nous_account import NousPortalAccountInfo

    kwargs.setdefault("logged_in", True)
    kwargs.setdefault("source", "account_api")
    kwargs.setdefault("fresh", True)
    return NousPortalAccountInfo(**kwargs)


def test_account_seed_used_fraction_pairs_subscription_object_fields():
    """The cold-start seed's subscription used_fraction must be computed from the
    SAME object as its denominator — ``subscription.credits_remaining`` over
    ``subscription.monthly_credits`` — so the cold-start usage-band notices match
    the /usage gauge (``build_nous_credits_snapshot``).

    ``paid_service_access_info.subscription_credits_remaining`` is a separate
    figure that can differ; pairing it with ``monthly_credits`` would compute a
    used_fraction that disagrees with what the user sees in /usage.
    """
    from agent.credits_tracker import _credits_state_from_account
    from hermes_cli.nous_account import (
        NousPaidServiceAccessInfo,
        NousPortalSubscriptionInfo,
    )

    info = _account(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(
            plan="Ultra",
            monthly_credits=20.0,
            credits_remaining=5.0,  # 75% used — the gauge numerator
        ),
        paid_service_access_info=NousPaidServiceAccessInfo(
            # Deliberately different from subscription.credits_remaining; this is
            # NOT the gauge numerator and must not drive used_fraction.
            subscription_credits_remaining=18.0,
            total_usable_credits=17.0,
        ),
    )

    state = _credits_state_from_account(info)
    assert state is not None
    # 5.0 of a 20.0 cap → 75% used (matches the /usage gauge), NOT the 10% the
    # access-object field (18.0 of 20.0) would yield.
    assert abs(state.used_fraction - 0.75) < 1e-9


def test_account_seed_fires_usage_band_matching_gauge():
    """A session opening at 75% subscription usage must fire the usage-band
    notice. Reading the wrong remaining field would compute 10% and stay silent.
    """
    from agent.credits_tracker import _credits_state_from_account
    from hermes_cli.nous_account import (
        NousPaidServiceAccessInfo,
        NousPortalSubscriptionInfo,
    )

    info = _account(
        paid_service_access=True,
        subscription=NousPortalSubscriptionInfo(
            monthly_credits=20.0, credits_remaining=5.0
        ),
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=18.0, total_usable_credits=17.0
        ),
    )

    state = _credits_state_from_account(info)
    assert state is not None
    assert "credits.usage" in _cold_start_notices(state)
