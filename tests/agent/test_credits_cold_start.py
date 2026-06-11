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


def test_cold_start_opens_already_at_90pct_warns():
    """A session that OPENS already ≥90% must warn immediately — the seed primes
    seen_below_90 so warn90 fires without a prior live crossing."""
    s = _state(
        remaining_micros=2_000_000, subscription_micros=2_000_000,
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=True,
    )
    assert s.used_fraction == 0.9
    assert "credits.usage" in _cold_start_notices(s)


def test_cold_start_grant_exhausted_warns_and_grant_spent():
    s = _state(
        remaining_micros=12_340_000, subscription_micros=0,
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        purchased_micros=12_340_000, denominator_kind="subscription_cap", paid_access=True,
    )
    assert s.used_fraction == 1.0
    keys = _cold_start_notices(s)
    assert "credits.usage" in keys
    assert "credits.grant_spent" in keys


def test_cold_start_depleted_warns():
    s = _state(
        remaining_micros=0, subscription_micros=0, purchased_micros=0,
        paid_access=False, disabled_reason="out_of_credits",
    )
    assert s.used_fraction is None  # no cap → no %, depletion keys off paid_access
    assert _cold_start_notices(s) == ["credits.depleted"]


def test_cold_start_debt_warns_and_depleted():
    """Negative subscription balance (the only signed field) → 100% used + depleted."""
    s = _state(
        remaining_micros=0, subscription_micros=-5_000_000,
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=False,
        disabled_reason="out_of_credits",
    )
    assert s.used_fraction == 1.0
    keys = _cold_start_notices(s)
    assert "credits.usage" in keys
    assert "credits.depleted" in keys


def test_cold_start_no_cap_degrades_to_depletion_only():
    """Without monthly_credits (older portals) the seed sets no limit → used_fraction
    None → only depletion can fire, never warn90."""
    healthy_no_cap = _state(
        remaining_micros=30_000_000, subscription_micros=18_000_000,
        subscription_limit_micros=None, denominator_kind="none", paid_access=True,
    )
    assert healthy_no_cap.used_fraction is None
    assert _cold_start_notices(healthy_no_cap) == []


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


def test_seed_fires_usage_band_at_session_open():
    a = _FakeAgent()
    assert _seed(a, "sub_90pct") is True
    assert a._credits_state is not None
    assert a.emitted == [(["credits.usage"], [])]


def test_seed_fires_depleted_at_session_open():
    a = _FakeAgent()
    assert _seed(a, "depleted") is True
    assert a.emitted == [(["credits.depleted"], [])]


def test_seed_depleted_suppressed_on_free_model():
    """A session that opens depleted but on a Nous ``:free`` model must NOT show
    the depleted banner — inference works fine on the free tier."""
    a = _FakeAgent(model="nvidia/nemotron-3-ultra:free")
    assert _seed(a, "depleted") is True
    assert a.emitted == [([], [])]


def test_seed_healthy_no_notice():
    a = _FakeAgent()
    assert _seed(a, "healthy") is True
    assert a.emitted == [([], [])]


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
