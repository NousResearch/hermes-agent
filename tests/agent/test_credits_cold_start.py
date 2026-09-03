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
    including the free-model suppression flag and the cross-rebuild band memory)."""

    def __init__(self, provider="nous", model="", session_id="session-1"):
        from agent.credits_tracker import evaluate_credits_notices, is_free_tier_model

        self.provider = provider
        self.model = model
        self.session_id = session_id
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
        from agent.credits_tracker import _remember_shown_band

        _remember_shown_band(self.session_id, self._credits_latch.get("usage_band"))


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


# ── desktop reap/resume rebuilds must not re-announce the same band (#101578) ─
#
# Desktop rebuilds the AIAgent (and its _credits_latch) on idle-reap/resume,
# but agent.session_id is stable across rebuilds. A fresh latch that blindly
# re-primes seen_below_90 re-announces the CURRENT band on every rebuild even
# though the user already saw it moments ago on the previous incarnation.


def _band_50_state() -> CreditsState:
    return _state(
        remaining_micros=10_000_000, subscription_micros=10_000_000,
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=True,
    )


def _band_75_state() -> CreditsState:
    return _state(
        remaining_micros=5_000_000, subscription_micros=5_000_000,
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=True,
    )


def test_rebuild_with_same_band_does_not_refire(monkeypatch):
    import agent.credits_tracker as ct

    monkeypatch.setattr(ct, "_seen_usage_bands", {})
    # First incarnation: session opens already at 50% — fires once.
    a1 = _FakeAgent(session_id="desktop-session-1")
    a1._credits_state = _band_50_state()
    a1._credits_latch["seen_below_90"] = True
    a1._emit_credits_notices()
    assert a1.emitted == [(["credits.usage"], [])]

    # Desktop idle-reap/resume: a BRAND NEW agent object, fresh empty latch,
    # SAME session_id, usage unchanged. Must not re-announce band 50.
    a2 = _FakeAgent(session_id="desktop-session-1")
    from agent.credits_tracker import _hydrate_seed_state

    _hydrate_seed_state(a2, _band_50_state())
    assert a2.emitted == [([], [])]


def test_rebuild_still_fires_on_genuine_band_change(monkeypatch):
    import agent.credits_tracker as ct

    monkeypatch.setattr(ct, "_seen_usage_bands", {})
    a1 = _FakeAgent(session_id="desktop-session-2")
    a1._credits_state = _band_50_state()
    a1._credits_latch["seen_below_90"] = True
    a1._emit_credits_notices()
    assert a1.emitted == [(["credits.usage"], [])]

    # Usage genuinely climbed to 75% before the next rebuild — must still warn.
    a2 = _FakeAgent(session_id="desktop-session-2")
    from agent.credits_tracker import _hydrate_seed_state

    _hydrate_seed_state(a2, _band_75_state())
    assert a2.emitted == [(["credits.usage"], ["credits.usage"])]


def test_unrelated_session_id_not_suppressed(monkeypatch):
    import agent.credits_tracker as ct

    monkeypatch.setattr(ct, "_seen_usage_bands", {})
    a1 = _FakeAgent(session_id="desktop-session-a")
    a1._credits_state = _band_50_state()
    a1._credits_latch["seen_below_90"] = True
    a1._emit_credits_notices()
    assert a1.emitted == [(["credits.usage"], [])]

    # A different session at the same band has never been shown anything.
    a2 = _FakeAgent(session_id="desktop-session-b")
    from agent.credits_tracker import _hydrate_seed_state

    _hydrate_seed_state(a2, _band_50_state())
    assert a2.emitted == [(["credits.usage"], [])]


def test_agent_without_session_id_falls_back_to_prior_behavior(monkeypatch):
    """CLI/plain agents may have no session_id at all -- must degrade to the
    pre-fix behavior (always re-primes), never raise."""
    import agent.credits_tracker as ct

    monkeypatch.setattr(ct, "_seen_usage_bands", {})
    a = _FakeAgent(session_id=None)
    from agent.credits_tracker import _hydrate_seed_state

    _hydrate_seed_state(a, _band_50_state())
    assert a.emitted == [(["credits.usage"], [])]
