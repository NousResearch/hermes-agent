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

    def __init__(self, provider="nous", model="", base_url=""):
        from agent.credits_tracker import evaluate_credits_notices, is_free_tier_model

        self.provider = provider
        self.model = model
        self.base_url = base_url
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
            model_is_free=self._is_free(self.model, self.base_url),
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


# ── background seed: the pricing warm the free-model gate depends on ─────────

_NOUS_BASE = "https://inference-api.nousresearch.com/v1"
# One subscription-billed row, keyed on the pre-/v1 root the picker caches under.
_SUBSCRIPTION_CATALOG = {
    "https://inference-api.nousresearch.com": {
        "openai/gpt-5.6-luna": {
            "prompt": "0.0000002000", "completion": "0.0000012000", "billing_mode": "subscription",
        },
    }
}


class _DepletedAccount:
    """The /api/oauth/account shape _credits_state_from_account reads: access off, no money."""

    class _Info:
        total_usable_credits = 0.0
        subscription_credits_remaining = 0.0
        purchased_credits_remaining = 0.0

    paid_service_access = False
    paid_service_access_info = _Info()
    subscription = None


def _cold_pricing_cache(monkeypatch):
    """Empty the process-wide pricing cache (and its expiry map) so the peek starts cold."""
    from hermes_cli import models_pricing

    monkeypatch.setattr(models_pricing, "_pricing_cache", {})
    monkeypatch.setattr(models_pricing, "_pricing_cache_retry_after", {})
    return models_pricing


def _run_bg_seed(monkeypatch, agent, *, warm):
    """Drive the seed down its BACKGROUND branch (no dev fixture) and join the thread. *warm* stands
    in for the real pricing fetch, so a test controls what the catalog holds and when."""
    import threading

    import hermes_cli.nous_account as nous_account
    from agent import credits_tracker

    monkeypatch.delenv("HERMES_DEV_CREDITS", raising=False)  # fixtures would take the sync path
    monkeypatch.setattr(credits_tracker, "_warm_nous_pricing_cache", warm)
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", lambda *a, **kw: _DepletedAccount())
    existing = set(threading.enumerate())
    result = credits_tracker.seed_credits_at_session_start(agent)
    for thread in set(threading.enumerate()) - existing:
        if thread.name == "credits-seed":
            thread.join(timeout=10)
            assert not thread.is_alive(), "seed thread hung"
    return result


def test_bg_seed_warms_pricing_so_a_subscription_model_escapes_the_banner(monkeypatch):
    """The regression: a cold process peeks an EMPTY catalog, so a subscription-billed model — which
    spends no credits — drew the depleted banner at session open. The seed fills the catalog first."""
    models_pricing = _cold_pricing_cache(monkeypatch)
    agent = _FakeAgent(model="openai/gpt-5.6-luna", base_url=_NOUS_BASE)

    assert _run_bg_seed(
        monkeypatch, agent, warm=lambda: models_pricing._pricing_cache.update(_SUBSCRIPTION_CATALOG)
    ) is True
    assert agent._credits_state.depleted is True  # the account really is out of credits...
    assert agent.emitted == [([], [])]  # ...and the session says nothing about it


def test_bg_seed_without_the_warm_still_warns(monkeypatch):
    """Guard rail for the test above: the same account and model DO warn while the catalog stays
    cold, so the suppression is the warm's doing and not the model id's."""
    _cold_pricing_cache(monkeypatch)
    agent = _FakeAgent(model="openai/gpt-5.6-luna", base_url=_NOUS_BASE)

    _run_bg_seed(monkeypatch, agent, warm=lambda: None)
    assert agent.emitted == [(["credits.depleted"], [])]


def test_bg_seed_reruns_the_policy_when_a_header_beat_it(monkeypatch):
    """A live inference header can land while the warm is still in flight; it evaluates against the
    cold catalog and shows the banner. The seed must keep that state but re-run the policy."""
    models_pricing = _cold_pricing_cache(monkeypatch)
    agent = _FakeAgent(model="openai/gpt-5.6-luna", base_url=_NOUS_BASE)
    header_state = _state(paid_access=False)

    def _warm():
        agent._credits_state = header_state  # the header lands mid-fetch...
        agent._emit_credits_notices()  # ...and evaluates against the cold catalog
        models_pricing._pricing_cache.update(_SUBSCRIPTION_CATALOG)

    _run_bg_seed(monkeypatch, agent, warm=_warm)
    assert agent._credits_state is header_state  # the seed never clobbers a live header
    assert agent.emitted[0] == (["credits.depleted"], [])  # cold: shown
    assert agent.emitted[-1] == ([], ["credits.depleted"])  # warm re-run: cleared, no "restored"


# ── inference-header path: the catalog goes cold again after the Nous TTL ────


def _mixin_agent(model="openai/gpt-5.6-luna"):
    """The real notice path (RateLimitCreditsMixin._emit_credits_notices) with a recording driver."""
    from agent.rate_limit_credits import RateLimitCreditsMixin
    from agent.status_output import StatusOutputMixin

    class _Agent(RateLimitCreditsMixin, StatusOutputMixin):
        provider = "nous"
        base_url = _NOUS_BASE

        def __init__(self):
            self.model = model
            self.shown: list = []
            self.cleared: list = []
            self.notice_callback = lambda n: self.shown.append(n.key)
            self.notice_clear_callback = self.cleared.append
            self._credits_state = _state(paid_access=False)

    return _Agent()


def _join_pricing_warm(agent):
    thread = getattr(agent, "_credits_pricing_warm", None)
    if thread is not None:
        thread.join(timeout=10)
        assert not thread.is_alive(), "pricing warm hung"


def test_header_after_ttl_expiry_rewarms_instead_of_flashing_the_banner(monkeypatch):
    """The session-start warm is one-shot; a Nous catalog expires after _NOUS_CATALOG_TTL_SECONDS. A
    header landing after that saw a cold peek and brought the depleted banner back for a
    subscription-billed model. Now the cold peek starts a re-warm and the banner never shows."""
    from agent import credits_tracker

    models_pricing = _cold_pricing_cache(monkeypatch)
    models_pricing._cache_catalog(
        _NOUS_BASE[:-3] + models_pricing._PRICING_AUTH_KEY_PREFIX + "abc",
        _SUBSCRIPTION_CATALOG[_NOUS_BASE[:-3]], models_pricing._NOUS_CATALOG_TTL_SECONDS)
    agent = _mixin_agent()
    agent._emit_credits_notices()
    assert agent.shown == []  # warm catalog: suppressed
    models_pricing._pricing_cache_retry_after = {  # ...then the TTL passes
        k: v - models_pricing._NOUS_CATALOG_TTL_SECONDS - 1 for k, v in models_pricing._pricing_cache_retry_after.items()}
    monkeypatch.setattr(credits_tracker, "_warm_nous_pricing_cache",
                        lambda: models_pricing._pricing_cache.update(_SUBSCRIPTION_CATALOG))

    agent._emit_credits_notices()  # the next inference header
    _join_pricing_warm(agent)
    assert agent.shown == []
    assert agent.cleared == []


def test_header_on_a_cold_catalog_still_warns_when_the_warm_fails(monkeypatch):
    """Fail-open guard rail: a warm that leaves the catalog cold decides against the cold peek — the
    banner shows — and the warm's own re-run does not spawn another warm."""
    import threading

    from agent import credits_tracker

    _cold_pricing_cache(monkeypatch)
    monkeypatch.setattr(credits_tracker, "_warm_nous_pricing_cache", lambda: None)
    agent = _mixin_agent()
    before = {t for t in threading.enumerate() if t.name == "credits-pricing-warm"}

    agent._emit_credits_notices()
    _join_pricing_warm(agent)
    assert agent.shown == ["credits.depleted"]
    assert len({t for t in threading.enumerate() if t.name == "credits-pricing-warm"} - before) == 0
    assert agent._credits_pricing_warm is not None and not agent._credits_pricing_warm.is_alive()
