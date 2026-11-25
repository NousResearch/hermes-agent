"""Pacing for a primary route whose 429s are token-budgeted (the Qwen4 free fair share).

Measured 2026-11-20: 26k tokens → 52s of refusal, 84k → 122s, 23 tokens → accepted right after a
refill. The bucket is what makes a second large turn wait instead of colliding with the first.
"""
import json
import time
from types import SimpleNamespace

import pytest

from agent import primary_budget_pacer as pacer

KEY = "nous/qwen/qwen4-max:free"
CFG = {"enabled": True, "providers": ["nous"], "tokens_per_minute": 33000.0,
       "burst_tokens": 48000.0, "max_wait_seconds": 240.0}
RATE = 33000.0 / 60.0  # 550 tokens/s


@pytest.fixture()
def state(tmp_path, monkeypatch):
    path = tmp_path / "primary_budget.json"
    monkeypatch.setattr(pacer, "_state_path", lambda: str(path))
    # Hermetic: the real ledger on this machine usually holds an active 429 window.
    from agent import model_health_ledger as ledger
    monkeypatch.setattr(ledger, "model_window_remaining", lambda *a, **k: 0.0)
    return path


def _balance(path):
    return json.loads(path.read_text())[KEY]["tokens"]


def test_first_request_takes_the_burst_and_charges_it(state):
    granted, wait = pacer.reserve("nous", "qwen/qwen4-max:free", 30000, cfg=CFG)
    assert (granted, wait) == (True, 0.0)
    assert _balance(state) == pytest.approx(18000.0)


def test_second_request_waits_for_what_the_first_consumed(state):
    assert pacer.reserve("nous", "qwen/qwen4-max:free", 40000, cfg=CFG)[0]
    granted, clearance = pacer.reserve("nous", "qwen/qwen4-max:free", 40000, cfg=CFG)
    assert not granted, "a second 40k turn must not collide with the first"
    # The first charge left 8k of the 48k burst, so only 32k has to refill.
    assert clearance == pytest.approx((40000 - 8000) / RATE, rel=0.01)


def test_refill_is_time_based_and_capped_at_the_burst(state):
    assert pacer.reserve("nous", "qwen/qwen4-max:free", 48000, cfg=CFG)[0]
    assert not pacer.reserve("nous", "qwen/qwen4-max:free", 30000, cfg=CFG)[0]
    # Well past a full refill: the bucket is capped at the burst, never more.
    later = time.time() + 600
    granted, wait = pacer.reserve("nous", "qwen/qwen4-max:free", 48000, now=later, cfg=CFG)
    assert granted and wait == 0.0
    assert _balance(state) == pytest.approx(0.0)


def test_request_larger_than_the_burst_reserves_the_whole_bucket(state):
    assert pacer.reserve("nous", "qwen/qwen4-max:free", 200000, cfg=CFG)[0]
    assert _balance(state) == pytest.approx(0.0)


def test_estimate_counts_messages_and_tools():
    api = {"messages": [{"role": "user", "content": "x" * 4000}],
           "tools": [{"name": "t", "description": "y" * 400}]}
    assert pacer.estimate_prompt_tokens(api) == 1101  # (4000 + 4 + 400 + 1) // 4
    assert pacer.estimate_prompt_tokens({}) == 0


def _agent(**overrides):
    agent = SimpleNamespace(
        _provider_fallback_active=False,
        _fallback_activated=False,
        provider="nous",
        model="qwen/qwen4-max:free",
        base_url="https://inference-api.nousresearch.com/v1",
        _primary_runtime={"provider": "nous", "model": "qwen/qwen4-max:free",
                          "base_url": "https://inference-api.nousresearch.com/v1",
                          "api_mode": "chat_completions"},
    )
    agent.__dict__.update(overrides)
    return agent


def test_pace_waits_for_clearance_and_reports_it(state, monkeypatch):
    monkeypatch.setattr(pacer, "_settings", lambda: dict(CFG))
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(pacer, "_now", lambda: clock["t"])

    def fake_sleep(seconds):
        clock["t"] += seconds

    api = {"messages": [{"role": "user", "content": "x" * (40000 * 4)}]}
    assert pacer.estimate_prompt_tokens(api) >= 40000
    assert pacer.reserve("nous", "qwen/qwen4-max:free", 40000, cfg=CFG)[0]

    waited = pacer.pace_primary_request(_agent(), api, sleep=fake_sleep)
    # Same 8k left by the first charge; the pacer must wait out exactly the shortfall.
    assert waited == pytest.approx((pacer.estimate_prompt_tokens(api) - 8000) / RATE, rel=0.05)


def test_pace_skips_a_fallback_route(state, monkeypatch):
    monkeypatch.setattr(pacer, "_settings", lambda: dict(CFG))
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    assert pacer.pace_primary_request(_agent(_fallback_activated=True), api) == 0.0
    assert not state.exists(), "fallback has its own budget; the primary bucket must stay untouched"


def test_pace_skips_when_disabled(state, monkeypatch):
    monkeypatch.setattr(pacer, "_settings", lambda: {**CFG, "enabled": False})
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    assert pacer.pace_primary_request(_agent(), api) == 0.0
    assert not state.exists()


def test_pace_skips_a_provider_not_named_in_config(state, monkeypatch):
    monkeypatch.setattr(pacer, "_settings", lambda: dict(CFG))
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    assert pacer.pace_primary_request(_agent(provider="openrouter"), api) == 0.0
    assert not state.exists()


def test_pace_waits_out_a_window_the_provider_already_declared(state, monkeypatch):
    """Observed 13:00:27: the local bucket was already healthy after 60s, but another lane had
    recorded a 112s provider window, so the send 429'd. The declared window outranks the bucket."""
    monkeypatch.setattr(pacer, "_settings", lambda: dict(CFG))
    from agent import model_health_ledger as ledger
    calls = {"n": 0}

    def remaining(provider, model):
        calls["n"] += 1
        return 112.0 if calls["n"] == 1 else 0.0

    monkeypatch.setattr(ledger, "model_window_remaining", remaining)
    slept = []
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    waited = pacer.pace_primary_request(_agent(), api, sleep=slept.append)
    assert slept == [pytest.approx(113.0)], "must wait the declared window before sending"
    assert waited == pytest.approx(113.0)


def test_pace_bounds_the_window_wait_by_max_wait(state, monkeypatch):
    monkeypatch.setattr(pacer, "_settings", lambda: {**CFG, "max_wait_seconds": 60.0})
    from agent import model_health_ledger as ledger
    monkeypatch.setattr(ledger, "model_window_remaining", lambda *a, **k: 900.0)
    slept = []
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    assert pacer.pace_primary_request(_agent(), api, sleep=slept.append) == pytest.approx(60.0)
    assert slept == [pytest.approx(60.0)]


def test_pace_skips_a_window_longer_than_its_window_budget(state, monkeypatch):
    """A 442s free-tier window cannot open inside a 120s budget.

    Observed 15:48:10 → 15:55:33: the pacer waited the declared 442s to the second and the very
    next request was refused again (retry_after=46). Sleeping part of such a window only delays
    the 429 that moves the turn to its fallback.
    """
    monkeypatch.setattr(pacer, "_settings", lambda: {**CFG, "max_window_wait_seconds": 120.0})
    from agent import model_health_ledger as ledger
    monkeypatch.setattr(ledger, "model_window_remaining", lambda *a, **k: 442.0)
    slept = []
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    assert pacer.pace_primary_request(_agent(), api, sleep=slept.append) == 0.0
    assert slept == [], "a window past the budget must not be slept through"


def test_pace_rechecks_a_window_a_sibling_lane_records_during_the_wait(state, monkeypatch):
    """The value sampled before the sleep is stale by the time we wake: re-read the shared ledger."""
    monkeypatch.setattr(pacer, "_settings", lambda: dict(CFG))
    from agent import model_health_ledger as ledger
    remaining = iter([20.0, 10.0, 0.0])
    monkeypatch.setattr(ledger, "model_window_remaining", lambda *a, **k: next(remaining))
    slept = []
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    waited = pacer.pace_primary_request(_agent(), api, sleep=slept.append)
    assert slept == [pytest.approx(21.0), pytest.approx(11.0)]
    assert waited == pytest.approx(32.0)


def test_pace_readopts_nous_credentials_after_a_wait(state, monkeypatch):
    """The JWT adopted in turn prep can expire during the pace; re-adopt before the send.

    Observed 15:46:14: a request that waited 158s came back 401 "API key is invalid, blocked or
    out of funds", recovered only by a forced credential refresh.
    """
    monkeypatch.setattr(pacer, "_settings", lambda: dict(CFG))
    from agent import model_health_ledger as ledger
    reads = {"n": 0}

    def remaining(*a, **k):
        reads["n"] += 1
        return 30.0 if reads["n"] == 1 else 0.0

    monkeypatch.setattr(ledger, "model_window_remaining", remaining)
    adopted = []
    agent = _agent(_adopt_nous_key_before_expiry=lambda: adopted.append(1))
    api = {"messages": [{"role": "user", "content": "x" * 4000}]}
    pacer.pace_primary_request(agent, api, sleep=lambda seconds: None)
    assert adopted == [1], "the pacer must re-adopt the key it may have outlived, exactly once"


def test_pace_fails_open_on_a_broken_payload(state, monkeypatch):
    monkeypatch.setattr(pacer, "_settings", lambda: dict(CFG))

    class Boom(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    assert pacer.pace_primary_request(_agent(), Boom()) == 0.0
