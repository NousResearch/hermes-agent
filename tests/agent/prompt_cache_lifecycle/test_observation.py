"""The plugin seam observes real lifecycle transitions, never the diagnostic status property."""

import builtins
import json
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.prompt_cache_lifecycle import warmer as lifecycle
from agent.prompt_cache_lifecycle.observation import WarmObservations
from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.agent.prompt_cache_lifecycle.test_warmer import _agent, _arm, _fire, _request, _Usage, priced


@pytest.fixture
def chat(request):
    request.getfixturevalue("priced")
    agent = _agent(_user_turn_count=1)
    ctx = PluginContext(PluginManifest(name="cache-observer"), PluginManager())
    try:
        yield agent, ctx
    finally:
        lifecycle.cancel_prompt_cache_warming(agent)


def status(ctx, session="s"):
    result = ctx.prompt_cache_warm_status(session_id=session)
    assert set(result) == {"managed", "next_warm_at", "reason"}
    assert result["managed"] == (result["reason"] in {"scheduled", "refreshing", "expiring"})
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    return result


def test_scheduled_reads_are_memory_only_and_do_not_rearm(chat, monkeypatch):
    agent, ctx = chat
    assert _arm(agent) is None
    warmer = agent._prompt_cache_warmer
    timer = warmer._run.timer
    expected = status(ctx)
    assert expected["reason"] == "scheduled" and expected["next_warm_at"] > time.time()

    def forbidden(*args, **kwargs):
        raise AssertionError("status attempted I/O, pricing, history or controller work")

    with monkeypatch.context() as m:
        for target in ("agent.usage_pricing.get_pricing_entry", "agent.usage_pricing.estimate_usage_cost",
                       "agent.prompt_cache_clock.cache_window", "agent.prompt_cache_clock.context_witness",
                       "agent.prompt_cache_lifecycle.warmer.get_prompt_cache_warmer"):
            m.setattr(target, forbidden)
        m.setattr(warmer, "_evaluate", forbidden)
        m.setattr(Path, "resolve", forbidden)
        m.setattr(Path, "stat", forbidden)
        m.setattr(builtins, "open", forbidden)
        m.setattr(socket, "socket", forbidden)
        for _ in range(10):
            assert status(ctx) == expected
    assert warmer._run.timer is timer and timer.is_alive()
    # An elapsed scheduled timestamp derives expiry without changing the stored observation.
    with monkeypatch.context() as m:
        m.setattr("agent.prompt_cache_lifecycle.observation.time.time", lambda: expected["next_warm_at"] + 1)
        assert status(ctx) == {"managed": True, "next_warm_at": None, "reason": "expiring"}
    assert status(ctx) == expected


@pytest.mark.parametrize("transition", ["reset", "route", "user_turn"])
def test_refreshing_snapshot_and_replacement_fence_a_real_inflight_completion(chat, monkeypatch, transition):
    agent, ctx = chat
    entered, finish = threading.Event(), threading.Event()
    client = agent._anthropic_client

    def blocked(**kwargs):
        entered.set()
        assert finish.wait(5)
        return SimpleNamespace(usage=_Usage(0, 1, 0, 120000), content=[], stop_reason="max_tokens")

    monkeypatch.setattr(client.messages, "create", blocked)
    assert _arm(agent) is None
    warmer = agent._prompt_cache_warmer
    warmer._miss_streak = (lifecycle.REAL_MISSES_TO_STOP - 1, lifecycle.REAL_MISSES_TO_STOP - 1)
    run = warmer._run
    run.timer.cancel()
    thread = threading.Thread(target=warmer._refresh, args=(run,))
    thread.start()
    try:
        assert entered.wait(5)
        assert status(ctx) == {"managed": True, "next_warm_at": None, "reason": "refreshing"}
        if transition == "reset":
            lifecycle.reset_prompt_cache_session(agent)
            agent._session_reset_epoch = 1
            assert status(ctx)["reason"] == "unavailable"
        elif transition == "route":
            agent.model = "claude-opus-4-6"
        else:
            agent._user_turn_count += 1
        payload = _request()
        payload["model"] = agent.model
        assert _arm(agent, payload) is None
        replacement = status(ctx)
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive() and status(ctx) == replacement
    assert not warmer.warming_disabled
    expected_streak = (0, 0) if transition == "reset" else (1, 1)
    assert warmer._miss_streak == expected_streak
    # A route switch keeps real spend in this user-turn budget. Its next normal request can
    # legitimately end that budget on money; a reset/new user turn must instead start afresh.
    if transition != "route":
        assert _arm(agent, payload) is None


@pytest.mark.parametrize("transition", ["disabled", "skip", "api_mode", "caching", "client", "replay", "route"])
@pytest.mark.parametrize("reopen", [False, True])
def test_superseding_policy_fences_late_miss_but_keeps_its_bill(chat, monkeypatch, transition, reopen):
    agent, ctx = chat
    entered, finish = threading.Event(), threading.Event()
    client = agent._anthropic_client
    miss = SimpleNamespace(usage=_Usage(0, 1, 0, 120000), content=[], stop_reason="max_tokens")
    monkeypatch.setattr(client.messages, "create", lambda **kwargs: miss)
    assert _arm(agent) is None
    warmer = agent._prompt_cache_warmer
    _fire(warmer)  # Establish one real miss, without manufacturing controller state.

    def blocked(**kwargs):
        entered.set()
        assert finish.wait(5)
        return miss

    monkeypatch.setattr(client.messages, "create", blocked)
    run = warmer._run
    run.timer.cancel()
    thread = threading.Thread(target=warmer._refresh, args=(run,))
    thread.start()
    try:
        assert entered.wait(5)
        payload = _request()
        attribute, value = {
            "disabled": ("_cache_disabled", True), "skip": ("_skip_prompt_cache_warming", True),
            "api_mode": ("api_mode", "codex_responses"), "caching": ("_use_prompt_caching", False),
            "client": ("_anthropic_client", None), "route": ("model", "claude-opus-4-6"),
            "replay": ("_skip_prompt_cache_warming", False),
        }[transition]
        previous = getattr(agent, attribute, False)
        if transition == "replay":
            payload["thinking"] = {"type": "enabled", "budget_tokens": 2048}
        else:
            setattr(agent, attribute, value)
        if transition in {"client", "replay"}:
            assert _arm(agent, payload) is not None
        else:
            lifecycle.cancel_prompt_cache_warming(agent, "new request in flight")
        expected_reason = "unknown" if transition == "route" else (
            "disabled" if transition in {"disabled", "caching"} else "unsupported")
        assert status(ctx)["reason"] == expected_reason
        if reopen:
            if transition != "replay":
                setattr(agent, attribute, previous)
            assert _arm(agent) is None
        expected = status(ctx)
        before = agent.session_estimated_cost_usd
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert status(ctx) == expected
    assert agent.session_cache_warm_calls == 2
    assert agent.session_estimated_cost_usd - before == pytest.approx(120000 * 6.25e-6 + 25e-6)
    assert run.budget.spent_usd > 0  # Reconcile the admitted request even though publication is fenced.
    assert not run.budget.ended and not warmer.warming_disabled
    assert warmer._miss_streak == (1, 1)


def _expire_count(agent, warmer, monkeypatch):
    for _ in range(lifecycle.WARM_BUDGET):
        _fire(warmer)


def _expire_money(agent, warmer, monkeypatch):
    warmer._budget.spent_usd = 100
    _fire(warmer)


def _expire_miss(agent, warmer, monkeypatch):
    monkeypatch.setattr(agent._anthropic_client.messages, "create", lambda **kw: SimpleNamespace(
        usage=_Usage(0, 1, 0, 120000), content=[], stop_reason="max_tokens"))
    for _ in range(lifecycle.REAL_MISSES_TO_STOP):
        _fire(warmer)


def _expire_learning(agent, warmer, monkeypatch):
    monkeypatch.setattr("agent.prompt_cache_routes.rule_for", lambda *a, **kw: "write-on-hit")
    warmer.start(_request(), SimpleNamespace(usage=None))


def _expire_economics(agent, warmer, monkeypatch):
    monkeypatch.setattr(warmer, "_pricing", lambda: None)
    _arm(agent)


def _expire_late(agent, warmer, monkeypatch):
    warmer._run.refresh_deadline_at = 0
    _fire(warmer)


def _expire_failure(agent, warmer, monkeypatch):
    def failed(*a):
        raise RuntimeError("offline replay failure")
    monkeypatch.setattr(warmer, "_send", failed)
    _fire(warmer)


@pytest.mark.parametrize("expire, renew_budget", [
    (_expire_count, True), (_expire_money, True), (_expire_miss, False), (_expire_learning, False),
    (_expire_economics, False), (_expire_late, False), (_expire_failure, False),
])
def test_deliberate_expiry_survives_eviction_and_rebuild(chat, monkeypatch, expire, renew_budget):
    agent, ctx = chat
    assert _arm(agent) is None
    warmer = agent._prompt_cache_warmer
    expire(agent, warmer, monkeypatch)
    assert status(ctx) == {"managed": True, "next_warm_at": None, "reason": "expiring"}
    warmer.cancel("clients released")
    del agent._prompt_cache_warmer
    rebuilt = lifecycle.get_prompt_cache_warmer(agent)
    assert status(ctx)["reason"] == "expiring" and rebuilt._run is None
    if renew_budget:
        assert _arm(agent) == "warm budget spent this turn"
        agent._user_turn_count += 1
        assert _arm(agent) is None
        assert status(ctx)["reason"] == "scheduled"


@pytest.mark.parametrize("change, expected", [
    ({"_cache_disabled": True}, "disabled"),
    ({"api_mode": "codex_responses"}, "unsupported"),
    ({"_anthropic_client": None}, "unsupported"),
    ({"_skip_prompt_cache_warming": True}, "unsupported"),
])
def test_current_opt_out_supersedes_exhausted_ownership(chat, change, expected):
    agent, ctx = chat
    assert _arm(agent) is None
    agent._prompt_cache_warmer._budget.ended = True
    for key, value in change.items():
        setattr(agent, key, value)
    _arm(agent)
    assert status(ctx) == {"managed": False, "next_warm_at": None, "reason": expected}


def test_unknown_window_and_route_change_do_not_inherit_expiry(chat, monkeypatch):
    agent, ctx = chat
    assert _arm(agent) is None
    agent._prompt_cache_warmer.cancel()
    monkeypatch.setattr(lifecycle, "guaranteed_window", lambda *a: None)
    # Invalidating a deliberately expired window does not relinquish ownership.
    _arm(agent)
    assert status(ctx)["reason"] == "expiring"
    agent.model = "another-model"
    _arm(agent)
    assert status(ctx) == {"managed": False, "next_warm_at": None, "reason": "unknown"}


def test_explicit_disable_then_reenable_without_a_window_is_unknown(chat, monkeypatch):
    agent, ctx = chat
    agent._cache_disabled = True
    _arm(agent)
    assert status(ctx)["reason"] == "disabled"
    agent._cache_disabled = False
    monkeypatch.setattr(lifecycle, "guaranteed_window", lambda *a: None)
    _arm(agent)
    assert status(ctx)["reason"] == "unknown"


def test_rebuild_keeps_expiry_not_a_new_route_or_current_opt_out(chat):
    agent, ctx = chat
    assert _arm(agent) is None
    lifecycle.cancel_prompt_cache_warming(agent, "release_clients")
    del agent._prompt_cache_warmer
    agent._user_turn_count = 0  # construction is not a new-user-response budget transition
    lifecycle.get_prompt_cache_warmer(agent)
    assert status(ctx)["reason"] == "expiring"
    del agent._prompt_cache_warmer
    agent.model = "another-model"
    lifecycle.get_prompt_cache_warmer(agent)
    assert status(ctx)["reason"] == "unknown"
    agent._cache_disabled = True
    lifecycle.cancel_prompt_cache_warming(agent, "release_clients")
    assert status(ctx)["reason"] == "disabled"


def test_a_route_switch_replaces_a_retained_miss_with_a_new_policy(chat):
    agent, ctx = chat
    assert _arm(agent) is None
    warmer = agent._prompt_cache_warmer
    with pytest.MonkeyPatch.context() as patch:
        _expire_miss(agent, warmer, patch)
    assert status(ctx)["reason"] == "expiring"
    agent.model = "claude-opus-4-6"
    agent._user_turn_count += 1
    payload = _request()
    payload["model"] = agent.model
    assert _arm(agent, payload) is None
    assert status(ctx)["reason"] == "scheduled"


def test_compaction_carries_only_validated_intent_and_reset_clears_it(chat, monkeypatch):
    agent, ctx = chat
    assert _arm(agent) is None
    agent._prompt_cache_warmer.cancel("deliberate compaction")
    lifecycle.carry_missed_route("s", "compaction-child")
    assert status(ctx, "compaction-child")["reason"] == "expiring"
    lifecycle.carry_missed_route("unrelated", "unknown-child")
    assert status(ctx, "unknown-child")["reason"] == "unavailable"
    agent.session_id = "compaction-child"
    child = lifecycle.get_prompt_cache_warmer(agent)
    monkeypatch.setattr(lifecycle, "guaranteed_window", lambda *a: None)
    _arm(agent)
    assert status(ctx, "compaction-child")["reason"] == "expiring"
    lifecycle.reset_prompt_cache_session(agent)
    lifecycle.reset_prompt_cache_session(agent)
    assert status(ctx, "compaction-child")["reason"] == "unavailable"
    assert child._run is None


def test_same_session_is_scoped_to_manager_not_ambient_poller_home(chat, tmp_path):
    agent, ctx = chat
    assert _arm(agent) is None
    other_home = tmp_path / "another-home"
    other_home.mkdir()
    token = set_hermes_home_override(other_home)
    other = _agent(_cache_disabled=True, _user_turn_count=1)
    try:
        other_ctx = PluginContext(PluginManifest(name="cache-observer"), PluginManager())
        _arm(other)
        assert status(other_ctx)["reason"] == "disabled"
        assert status(ctx)["reason"] == "scheduled"
        lifecycle._remember_missed_route("s", ("route",))
    finally:
        lifecycle.cancel_prompt_cache_warming(other)
        reset_hermes_home_override(token)
    assert lifecycle.get_prompt_cache_warmer(_agent())._missed_route is None
    assert status(other_ctx)["reason"] == "disabled"


def test_bounded_store_has_no_read_mutation_and_stale_owners_cannot_publish():
    store = WarmObservations(capacity=2)
    key = ("home", "session")
    old = store.claim(key, object())
    store.update(key, replace(old, reason="scheduled", next_warm_at=float("inf")))
    assert store.snapshot(*key)["reason"] == "expiring"
    new = store.claim(key, object())
    store.update(key, replace(old, reason="disabled"))
    assert store.snapshot(*key)["reason"] == new.reason
    store.carry(key, key)
    store.update(key, replace(new, reason="scheduled", next_warm_at=time.time() + 60))
    assert store.snapshot(*key)["reason"] == "scheduled", "in-place compaction keeps the controller's identity"
    child = ("home", "child")
    child_item = store.claim(child, object())
    store.update(child, replace(child_item, reason="disabled"))
    store.carry(key, child)
    assert store.snapshot(*child)["reason"] == "disabled", "current child policy supersedes its ancestor"
    store.clear(key)
    store.update(key, replace(new, reason="scheduled", next_warm_at=time.time() + 60))
    assert store.snapshot(*key)["reason"] == "unavailable"
    store.claim(key, object())
    store.claim(("home", "second"), object())
    store.snapshot(*key)
    store.claim(("home", "third"), object())
    assert store.snapshot(*key)["reason"] == "unavailable", "reads must not renew retention"
