"""Defer-on-429: an unattended run parks on the primary's reset instead of
walking the fallback chain down to a paid floor.

Covers the policy gates (``agent/unattended_defer.py``), the walk-site refusal
(``chat_completion_helpers.try_activate_fallback``), the kanban ``not_before``
re-queue, and the cron silent-skip tick.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from agent import unattended_defer as ud


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every defer-relevant env var starts unset; the ContextVar starts cleared."""
    for var in (ud.ENV_POLICY, ud.ENV_LATENCY_CRITICAL, ud.ENV_UNATTENDED,
                "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    token = ud.set_latency_critical(None)
    yield
    ud.reset_latency_critical(token)


class _FakePool:
    def __init__(self, next_at):
        self._next_at = next_at

    def next_available_at(self):
        return self._next_at


def _agent(*, pool_reset=None, fallback_activated=False, provider="anthropic",
           model="claude-opus-5", primary_provider="anthropic"):
    return SimpleNamespace(
        provider=provider, model=model,
        _credential_pool=_FakePool(pool_reset) if pool_reset else None,
        _fallback_activated=fallback_activated,
        _primary_runtime={"provider": primary_provider},
    )


RESET_IN_1H = staticmethod(lambda: time.time() + 3600)


# --------------------------------------------------------------------------
# policy resolution
# --------------------------------------------------------------------------

def test_policy_defaults_to_walk_so_existing_configs_are_untouched():
    assert ud.resolve_policy({}) == ud.POLICY_WALK
    assert ud.resolve_policy(None) in (ud.POLICY_WALK, ud.POLICY_DEFER)  # load_config path


def test_policy_reads_config_key():
    assert ud.resolve_policy({"fallback": {"unattended_on_rate_limit": "defer"}}) == ud.POLICY_DEFER
    assert ud.resolve_policy({"fallback": {"unattended_on_rate_limit": "walk"}}) == ud.POLICY_WALK


def test_policy_ignores_typo_rather_than_parking_the_fleet():
    """An unrecognized value must not silently become 'defer'."""
    assert ud.resolve_policy({"fallback": {"unattended_on_rate_limit": "deffer"}}) == ud.POLICY_WALK


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv(ud.ENV_POLICY, "defer")
    assert ud.resolve_policy({"fallback": {"unattended_on_rate_limit": "walk"}}) == ud.POLICY_DEFER


# --------------------------------------------------------------------------
# surface detection
# --------------------------------------------------------------------------

def test_kanban_worker_is_unattended(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    assert ud.is_unattended_run() is True


def test_explicit_flag_is_unattended(monkeypatch):
    monkeypatch.setenv(ud.ENV_UNATTENDED, "1")
    assert ud.is_unattended_run() is True


def test_cron_session_is_unattended():
    with patch("gateway.session_context.get_session_env", return_value="1"):
        assert ud.is_unattended_run() is True


def test_plain_shell_is_not_unattended():
    with patch("gateway.session_context.get_session_env", return_value=""):
        assert ud.is_unattended_run() is False


def test_latency_critical_contextvar_beats_env(monkeypatch):
    monkeypatch.setenv(ud.ENV_LATENCY_CRITICAL, "1")
    token = ud.set_latency_critical(False)
    try:
        assert ud.is_latency_critical() is False
    finally:
        ud.reset_latency_critical(token)
    assert ud.is_latency_critical() is True


# --------------------------------------------------------------------------
# resolve_deferral — the gate stack
# --------------------------------------------------------------------------

DEFER_CFG = {"fallback": {"unattended_on_rate_limit": "defer"}}


def test_defers_when_every_gate_passes(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    reset = time.time() + 3600
    got = ud.resolve_deferral(_agent(pool_reset=reset), FailoverReason.rate_limit, config=DEFER_CFG)
    assert got == pytest.approx(reset)


def test_walk_policy_never_defers(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    got = ud.resolve_deferral(
        _agent(pool_reset=time.time() + 3600), FailoverReason.rate_limit,
        config={"fallback": {"unattended_on_rate_limit": "walk"}},
    )
    assert got is None


def test_interactive_session_never_defers():
    """A human is waiting: degraded answer now beats no answer."""
    with patch("gateway.session_context.get_session_env", return_value=""):
        got = ud.resolve_deferral(
            _agent(pool_reset=time.time() + 3600), FailoverReason.rate_limit, config=DEFER_CFG,
        )
    assert got is None


def test_latency_critical_unattended_run_still_walks(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    monkeypatch.setenv(ud.ENV_LATENCY_CRITICAL, "1")
    got = ud.resolve_deferral(
        _agent(pool_reset=time.time() + 3600), FailoverReason.rate_limit, config=DEFER_CFG,
    )
    assert got is None


@pytest.mark.parametrize("reason", [
    FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit,
])
def test_all_rate_limit_class_reasons_defer(monkeypatch, reason):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    assert ud.resolve_deferral(
        _agent(pool_reset=time.time() + 3600), reason, config=DEFER_CFG,
    ) is not None


def test_non_rate_limit_reason_walks(monkeypatch):
    """A 500 or a context overflow is not a quota problem — the chain still helps."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    assert ud.resolve_deferral(
        _agent(pool_reset=time.time() + 3600), FailoverReason.server_error, config=DEFER_CFG,
    ) is None


def test_no_reset_timestamp_means_no_deferral(monkeypatch):
    """THE load-bearing gate: parking with no known resume time is worse than a cheap rung."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    assert ud.resolve_deferral(_agent(pool_reset=None), FailoverReason.rate_limit, config=DEFER_CFG) is None


def test_reset_from_error_context_when_pool_is_silent(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    reset = time.time() + 1800
    got = ud.resolve_deferral(
        _agent(pool_reset=None), FailoverReason.rate_limit,
        error_context={"reset_at": reset}, config=DEFER_CFG,
    )
    assert got == pytest.approx(reset)


def test_already_on_a_fallback_rung_keeps_walking(monkeypatch):
    """A 429 while a fallback is active came from THAT rung, not the primary."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    agent = _agent(pool_reset=time.time() + 3600, fallback_activated=True,
                   provider="venice", primary_provider="anthropic")
    assert ud.resolve_deferral(agent, FailoverReason.rate_limit, config=DEFER_CFG) is None


def test_imminent_reset_is_not_worth_a_round_trip(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    got = ud.resolve_deferral(
        _agent(pool_reset=time.time() + 5), FailoverReason.rate_limit, config=DEFER_CFG,
    )
    assert got is None


def test_absurd_reset_is_clamped_to_24h(monkeypatch):
    """A malformed header must not park a card for a fortnight."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    got = ud.resolve_deferral(
        _agent(pool_reset=time.time() + 30 * 86400), FailoverReason.rate_limit, config=DEFER_CFG,
    )
    assert got is not None
    assert got - time.time() == pytest.approx(ud.MAX_DEFER_SECONDS, abs=5)


def test_resolve_deferral_fails_open_on_internal_error(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    with patch.object(ud, "resolve_policy", side_effect=RuntimeError("boom")):
        assert ud.resolve_deferral(_agent(pool_reset=time.time() + 3600),
                                   FailoverReason.rate_limit, config=DEFER_CFG) is None


# --------------------------------------------------------------------------
# walk-site integration
# --------------------------------------------------------------------------

def test_try_activate_fallback_refuses_to_walk_when_deferring(monkeypatch):
    """The whole point: a capped primary must NOT reach the next rung."""
    from agent import chat_completion_helpers as cch

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    monkeypatch.setenv(ud.ENV_POLICY, "defer")
    agent = _agent(pool_reset=time.time() + 3600)
    agent._fallback_chain = [{"provider": "venice", "model": "z-ai-glm-5-3-flash"}]
    agent._fallback_index = 0
    agent._buffer_status = lambda *_a, **_k: None
    agent._pending_fallback_notice = None
    agent._rate_limit_backoff_count = 0

    assert cch.try_activate_fallback(agent, FailoverReason.rate_limit) is False
    # Chain untouched: nothing was tried.
    assert agent._fallback_index == 0
    assert ud.get_deferral(agent) is not None


def test_try_activate_fallback_still_walks_for_interactive(monkeypatch):
    """Regression guard: the default path must be byte-for-byte unchanged."""
    from agent import chat_completion_helpers as cch

    monkeypatch.setenv(ud.ENV_POLICY, "walk")
    agent = _agent(pool_reset=time.time() + 3600)
    agent._fallback_chain = []
    agent._fallback_index = 0
    agent._rate_limit_backoff_count = 0
    with patch("gateway.session_context.get_session_env", return_value=""):
        # Empty chain -> False, but via the EXHAUSTED path, not the defer path.
        assert cch.try_activate_fallback(agent, FailoverReason.rate_limit) is False
    assert ud.get_deferral(agent) is None


def test_deferral_is_sticky_across_later_walk_attempts(monkeypatch):
    """The max-retries site calls with reason=None; it must not undo the deferral."""
    from agent import chat_completion_helpers as cch

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    monkeypatch.setenv(ud.ENV_POLICY, "defer")
    agent = _agent(pool_reset=time.time() + 3600)
    agent._fallback_chain = [{"provider": "venice", "model": "z-ai-glm-5-3-flash"}]
    agent._fallback_index = 0
    agent._buffer_status = lambda *_a, **_k: None
    agent._pending_fallback_notice = None
    agent._rate_limit_backoff_count = 0

    assert cch.try_activate_fallback(agent, FailoverReason.rate_limit) is False
    # Second call, no reason at all — the sticky guard must still refuse.
    assert cch.try_activate_fallback(agent, None) is False
    assert agent._fallback_index == 0


# --------------------------------------------------------------------------
# kanban: not_before re-queue
# --------------------------------------------------------------------------

@pytest.fixture()
def board(monkeypatch, tmp_path):
    """Fresh HERMES_HOME + kanban DB, re-importing so the new home is picked up."""
    import sys
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    kb.create_board(slug="default", name="Test")
    yield kb, kbc, kbd


def _new_running_task(kb, kbc):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="t", assignee="linuxops")
        kb.claim_task(conn, task_id, claimer="host:1")
    return task_id


def test_defer_task_requeues_with_not_before(board):
    kb, kbc, _kbd = board
    task_id = _new_running_task(kb, kbc)
    reset = time.time() + 3600

    with kbc.connect_closing() as conn:
        assert kb.defer_task(conn, task_id, not_before=reset, reason="rate limited") is True

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, not_before, claim_lock, consecutive_failures FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    assert row["status"] == "ready"
    assert row["not_before"] == int(reset)
    assert row["claim_lock"] is None
    # A provider cap is not the worker's failure — the breaker must not advance.
    assert row["consecutive_failures"] == 0


def test_deferred_task_is_invisible_to_dispatch_until_the_reset(board):
    kb, kbc, kbd = board
    task_id = _new_running_task(kb, kbc)
    with kbc.connect_closing() as conn:
        kb.defer_task(conn, task_id, not_before=time.time() + 3600, reason="rate limited")

    with kbc.connect_closing() as conn:
        assert [r["id"] for r in kbd._lane_rows(conn, "ready")] == []
        # ...and health telemetry must not call it a stall.
        assert kbd.has_spawnable_ready(conn) is False


def test_deferred_task_returns_to_dispatch_once_the_reset_passes(board):
    kb, kbc, kbd = board
    task_id = _new_running_task(kb, kbc)
    with kbc.connect_closing() as conn:
        kb.defer_task(conn, task_id, not_before=time.time() - 1, reason="rate limited")

    with kbc.connect_closing() as conn:
        assert [r["id"] for r in kbd._lane_rows(conn, "ready")] == [task_id]


def test_undeferred_tasks_are_unaffected(board):
    kb, kbc, kbd = board
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="t", assignee="linuxops")
    with kbc.connect_closing() as conn:
        assert [r["id"] for r in kbd._lane_rows(conn, "ready")] == [task_id]


def test_clear_task_deferral_makes_it_dispatchable_now(board):
    kb, kbc, kbd = board
    task_id = _new_running_task(kb, kbc)
    with kbc.connect_closing() as conn:
        kb.defer_task(conn, task_id, not_before=time.time() + 3600)
    with kbc.connect_closing() as conn:
        assert kb.clear_task_deferral(conn, task_id) is True
    with kbc.connect_closing() as conn:
        assert [r["id"] for r in kbd._lane_rows(conn, "ready")] == [task_id]


def test_defer_task_records_an_auditable_event(board):
    kb, kbc, _kbd = board
    task_id = _new_running_task(kb, kbc)
    with kbc.connect_closing() as conn:
        kb.defer_task(conn, task_id, not_before=time.time() + 3600, reason="capped until X")
    with kbc.connect_closing() as conn:
        kinds = [e["kind"] for e in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ?", (task_id,)).fetchall()]
    assert "deferred" in kinds


def test_defer_task_respects_expected_run_id(board):
    """A stale worker must not park a card its successor already owns."""
    kb, kbc, _kbd = board
    task_id = _new_running_task(kb, kbc)
    with kbc.connect_closing() as conn:
        assert kb.defer_task(
            conn, task_id, not_before=time.time() + 60, expected_run_id=999_999,
        ) is False


# --------------------------------------------------------------------------
# cron: silent skip
# --------------------------------------------------------------------------

def test_cron_reads_deferred_until_off_the_turn_result():
    from cron import scheduler

    assert scheduler._deferred_run_result({"deferred_until": 123.0}) == 123.0
    assert scheduler._deferred_run_result({}) is None
    assert scheduler._deferred_run_result(None) is None
    assert scheduler._deferred_run_result({"deferred_until": "nope"}) is None


def test_cron_deferred_tick_is_silent_and_successful():
    """A rolled-over quota bucket is not a job failure and must not alert."""
    from cron import scheduler

    ok, doc, response, error = scheduler._deferred_tick_result(
        {"id": "j1"}, "j1", "job", "prompt", time.time() + 3600,
    )
    assert ok is True
    assert error is None
    assert response == scheduler.SILENT_MARKER
    assert "deferred until" in doc.lower()


# --------------------------------------------------------------------------
# The constant-relocation trap
# --------------------------------------------------------------------------

def test_rate_limit_reasons_survive_the_constant_moving_modules():
    """``_RATE_LIMIT_FAILOVER_REASONS`` must be found wherever upstream keeps it.

    This is not hypothetical. The constant lived in ``chat_completion_helpers``,
    then moved to ``fallback_cooldown``; the original code imported it from the
    single hardcoded module INSIDE ``resolve_deferral``'s blanket ``except``, so
    the relocation turned the whole feature INERT — every unattended run walked
    the chain again, with no error and no log line. Only an end-to-end assertion
    catches that class, because each unit gate still passed in isolation.
    """
    from agent import unattended_defer as ud
    from agent.error_classifier import FailoverReason

    ud._REASONS_CACHE = None  # force a real resolution, not a memoized one
    try:
        reasons = ud._rate_limit_reasons()
        assert reasons, "defer-on-429 is INERT: no rate-limit reasons resolved"
        for name in ("rate_limit", "billing", "upstream_rate_limit"):
            assert getattr(FailoverReason, name) in reasons, f"{name} missing"
    finally:
        ud._REASONS_CACHE = None


def test_feature_is_wired_into_the_real_walk_site(monkeypatch):
    """End-to-end guard: the REAL ``try_activate_fallback`` must refuse to walk.

    Deliberately calls the production function rather than the policy helper — a
    passing policy unit test proved nothing when the walk site's import was dead.
    """
    from agent import chat_completion_helpers as cch
    from agent import unattended_defer as ud
    from agent.error_classifier import FailoverReason

    ud._REASONS_CACHE = None
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_wired")
    monkeypatch.setenv("HERMES_UNATTENDED_ON_RATE_LIMIT", "defer")
    monkeypatch.delenv("HERMES_UNATTENDED_LATENCY_CRITICAL", raising=False)

    reset_at = time.time() + 3600
    agent = _agent(pool_reset=reset_at)
    # The walk site needs a real chain to consume; _agent() only carries the gates.
    agent._fallback_index = 0
    agent._fallback_chain = [
        {"provider": "anthropic", "model": "claude-sonnet-5"},
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"},  # paid floor
    ]
    agent._rate_limit_backoff_count = 0
    agent._pending_fallback_notice = None
    agent._buffer_status = lambda *a, **k: None
    walked = cch.try_activate_fallback(agent, FailoverReason.rate_limit)

    assert walked is False, "unattended run walked the chain"
    assert agent._fallback_index == 0, "chain was consumed despite the deferral"
    assert ud.get_deferral(agent) is not None, "no deferral recorded"
    ud._REASONS_CACHE = None
