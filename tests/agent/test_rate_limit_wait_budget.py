"""The primary's rate-limit wait budget: ride out a short window, hand off a long one.

Qwen4's free route answered with ``retry_after`` values from 7s to 442s on 2026-11-20. The live
config clamped ``agent.api_max_retries`` to 1, so every single 429 fell through to the local 27B
with no retry at all ("attempt 1/1" in agent.log). Restoring the retry budget alone is not enough:
honouring a 442s window three times would idle a lane for 22 minutes before the fallback ran, so
``agent.max_rate_limit_wait_seconds`` turns a window it cannot afford into an immediate hand-off.

The other half matters just as much: a 55s refusal fits the budget, and switching models on it
costs the primary the turn for nothing. The budget therefore also suppresses the eager 429
fallback for that first, affordable window (bounded to one wait per turn), and overrides the
attended free-tier heuristic so the window is actually ridden out.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest

from agent import turn_api_error
from agent.error_classifier import FailoverReason
from agent.turn_recovery import (
    rate_limit_wait_budget_seconds, rate_limit_wait_exceeds_budget, rate_limit_wait_fits_budget,
    rate_limit_window_seconds, route_classified_error,
)
from agent.turn_retry_state import TurnRetryState


class _Refusal(Exception):
    def __init__(self, retry_after):
        super().__init__("429 fair-share rate limit")
        self.status_code = 429
        self.body = {"status": 429, "reason": "rate_limited", "retry_after": retry_after}


@pytest.mark.parametrize("retry_after, budget, expected", [
    (46, 120, False),    # fits: wait it out and retry on the primary
    (120, 120, False),   # at the budget: still worth one attempt
    (442, 120, True),    # cannot fit: hand the turn to the fallback chain
    (442, 0, False),     # off: preserve the old Retry-After behaviour
    (None, 120, False),  # no declared window: the adaptive backoff owns it
])
def test_budget_only_rejects_a_declared_window_it_cannot_afford(retry_after, budget, expected):
    agent = SimpleNamespace(max_rate_limit_wait_seconds=budget)
    assert rate_limit_wait_exceeds_budget(agent, _Refusal(retry_after), {}) is expected


def test_fits_budget_requires_an_explicit_budget_and_a_declared_window():
    agent = SimpleNamespace(max_rate_limit_wait_seconds=120)
    assert rate_limit_wait_fits_budget(agent, _Refusal(46), {}) is True
    assert rate_limit_wait_fits_budget(agent, _Refusal(120), {}) is True
    assert rate_limit_wait_fits_budget(agent, _Refusal(442), {}) is False
    assert rate_limit_wait_fits_budget(SimpleNamespace(max_rate_limit_wait_seconds=0), _Refusal(46), {}) is False
    assert rate_limit_wait_fits_budget(agent, _Refusal(None), {}) is False
    assert rate_limit_wait_fits_budget(SimpleNamespace(), _Refusal(46), {}) is False
    assert rate_limit_wait_budget_seconds(SimpleNamespace(max_rate_limit_wait_seconds="oops")) == 0.0
    assert rate_limit_window_seconds(_Refusal(46), {}) == 46


def test_a_window_only_the_extracted_context_carries_still_counts():
    """A message-only window carries no header or body field. The turn's ``error_context``
    (``extract_api_error_context``) is where the message grammar lands, so the budget reads the
    same ``reset_at`` as the cooldown ledger and the backoff — one context, no classifier copy."""
    agent = SimpleNamespace(max_rate_limit_wait_seconds=120)
    assert rate_limit_window_seconds(_Refusal(None), {"reset_at": time.time() + 30}) == pytest.approx(30, abs=2)
    assert rate_limit_wait_fits_budget(agent, _Refusal(None), {"reset_at": time.time() + 30}) is True
    assert rate_limit_wait_exceeds_budget(
        agent, _Refusal(None), {"reset_at": time.time() + 300}
    ) is True


def test_an_explicit_retry_after_wins_over_a_context_reset():
    """The body may carry a short throttle beside a longer quota reset. The explicit wait is the
    provider's ask (the pool's precedence), so the budget must not read the longer reset."""
    assert rate_limit_window_seconds(_Refusal(13), {"reset_at": time.time() + 442}) == 13


def _classified():
    return SimpleNamespace(
        reason=FailoverReason.rate_limit, retryable=True, should_compress=False,
        should_rotate_credential=False, should_fallback=True, is_auth=False, status_code=429,
        error_context={},
    )


def _agent(budget=120.0, fallback=True):
    calls = {}

    def activate_fallback(*, reason=None, reset_at=None):
        calls["fallback"] = {"reason": reason, "reset_at": reset_at}
        return fallback

    agent = SimpleNamespace(
        log_prefix="", max_rate_limit_wait_seconds=budget, provider="nous",
        _credential_pool=None, _fallback_chain=[{"provider": "custom", "model": "local"}],
        _fallback_index=0, _rate_limit_state=None,
        _try_recover_primary_transport=lambda *a, **k: False,
        _has_pending_fallback=lambda: True,
        _try_activate_fallback=activate_fallback,
        _buffer_diagnostic_status=lambda *a, **k: None,
        calls=calls,
    )
    return agent


def _route(agent, retry_after, *, retry_count=1, max_retries=3):
    # 1 is the production convention: ``handle_api_error`` increments before routing, so the
    # first 429 of a turn reaches ``route_classified_error`` as 1, the second as 2.
    return route_classified_error(
        agent, api_error=_Refusal(retry_after), classified=_classified(),
        _retry=TurnRetryState(), error_msg="429 fair-share rate limit", error_context={},
        recovered_with_pool=False, base_url="https://inference-api.nousresearch.com/v1",
        model="qwen/qwen4-max:free", messages=[],
        api_messages=[{"role": "user", "content": "hi"}], system_message="sys",
        active_system_prompt="sys", conversation_history=[], retry_count=retry_count,
        max_retries=max_retries, compression_attempts=0, max_compression_attempts=3,
        api_call_count=1, effective_task_id=None,
    )


def _settle(agent, retry_after, *, retry_count=1, max_retries=3):
    return turn_api_error.settle_unrecovered_error(
        agent, api_error=_Refusal(retry_after), classified=_classified(),
        _retry=TurnRetryState(), status_code=429, error_msg="429 fair-share rate limit",
        is_context_length_error=False, is_rate_limited=True, _is_zai_coding_overload=False,
        _provider="nous", _base="https://inference-api.nousresearch.com/v1",
        _model="qwen/qwen4-max:free", messages=[], api_messages=[], api_kwargs={},
        active_system_prompt="sys", conversation_history=[], approx_tokens=1000,
        retry_count=retry_count, max_retries=max_retries, compression_attempts=0,
        api_call_count=1, error_context={}, current_turn_user_idx=None,
    )


def test_a_long_window_hands_the_turn_to_the_fallback_without_waiting(monkeypatch, caplog):
    monkeypatch.setattr(turn_api_error, "free_tier_cooldown_ends_turn", lambda *a, **k: False)
    monkeypatch.setattr("agent.conversation_loop._arm_fallback_restart",
                        lambda agent, messages, system_prompt, retry: system_prompt)
    slept = []
    monkeypatch.setattr(turn_api_error, "interruptible_backoff_sleep", lambda *a, **k: slept.append(a) or None)
    agent = _agent(budget=120.0)
    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        verdict = _settle(agent, 442)
    assert verdict.action == "break"
    assert agent.calls["fallback"]["reason"] == FailoverReason.rate_limit
    assert slept == [], "a window past the budget must not be slept through"
    assert any("exceeds the 120s primary wait budget" in r.getMessage() for r in caplog.records)


def test_an_affordable_window_is_not_consumed_by_the_eager_429_fallback(monkeypatch, caplog):
    monkeypatch.setattr("agent.conversation_loop._arm_fallback_restart",
                        lambda agent, messages, system_prompt, retry: system_prompt)
    agent = _agent(budget=120.0)
    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        verdict = _route(agent, 55)
    assert verdict.action == "fallthrough", "a 55s window fits the budget; stay on the primary"
    assert "fallback" not in agent.calls, "the eager 429 fallback must not fire for an affordable window"
    assert any("55.0s fits the 120s primary wait budget" in r.getMessage() for r in caplog.records)


def test_a_second_refusal_after_the_wait_hands_off(monkeypatch):
    """One wait per turn: a widening window must not idle the lane once per retry."""
    monkeypatch.setattr("agent.conversation_loop._arm_fallback_restart",
                        lambda agent, messages, system_prompt, retry: system_prompt)
    agent = _agent(budget=120.0)
    verdict = _route(agent, 55, retry_count=2)
    assert verdict.action == "break"
    assert agent.calls["fallback"]["reason"] == FailoverReason.rate_limit


def test_an_unaffordable_window_still_falls_back_eagerly(monkeypatch):
    monkeypatch.setattr("agent.conversation_loop._arm_fallback_restart",
                        lambda agent, messages, system_prompt, retry: system_prompt)
    agent = _agent(budget=120.0)
    verdict = _route(agent, 442)
    assert verdict.action == "break"
    assert agent.calls["fallback"]["reason"] == FailoverReason.rate_limit


def test_explicit_budget_overrides_the_attended_free_tier_heuristic(monkeypatch):
    monkeypatch.setattr(turn_api_error, "free_tier_cooldown_ends_turn", lambda *a, **k: True)
    monkeypatch.setattr(turn_api_error, "compute_error_backoff", lambda *a, **k: 90.0)
    waits = []
    monkeypatch.setattr(turn_api_error, "interruptible_backoff_sleep", lambda *a, **k: waits.append(a) or None)
    agent = _agent(budget=120.0)
    verdict = _settle(agent, 90)
    assert verdict.action == "fallthrough"
    assert "fallback" not in agent.calls
    assert waits and waits[0][1] == 90.0


def test_a_short_window_is_still_waited_out_and_retried_on_the_primary(monkeypatch):
    monkeypatch.setattr(turn_api_error, "free_tier_cooldown_ends_turn", lambda *a, **k: False)
    monkeypatch.setattr(turn_api_error, "compute_error_backoff", lambda *a, **k: 46.0)
    waits = []
    monkeypatch.setattr(turn_api_error, "interruptible_backoff_sleep", lambda *a, **k: waits.append(a) or None)
    agent = _agent(budget=120.0)
    verdict = _settle(agent, 46)
    assert verdict.action == "fallthrough"
    assert "fallback" not in agent.calls, "a 46s window fits the budget; stay on the primary"
    assert waits, "the retry must wait out the declared window"
    assert waits[0][1] == 46.0


# ── end-to-end through handle_api_error ──────────────────────────────────────
#
# The routing tests above call ``route_classified_error`` directly, so they cannot see the
# ``retry_count += 1`` that ``handle_api_error`` performs first. That increment is exactly what
# silently killed the fits-budget branch once (the guard read 0, production delivered 1), so the
# convention is pinned by driving the handler the retry loop itself calls.


def _handler_agent(budget=120.0):
    """Every attribute ``handle_api_error`` reads on the way to a verdict, with the real context
    extractor where the window resolution depends on it."""
    from agent.agent_runtime_helpers import extract_api_error_context

    calls = {"fallback": []}

    def activate_fallback(*, reason=None, reset_at=None):
        calls["fallback"].append({"reason": reason, "reset_at": reset_at})
        return True

    return SimpleNamespace(
        provider="nous", model="qwen/qwen4-max:free",
        base_url="https://inference-api.nousresearch.com/v1",
        api_key="test-key", api_mode="chat_completions",
        max_rate_limit_wait_seconds=budget,
        _extract_api_error_context=extract_api_error_context,
        thinking_callback=lambda _message: None,
        log_prefix="", verbose_logging=False, compression_enabled=True,
        _interrupt_requested=False, context_compressor=None,
        _unicode_sanitization_passes=0, _image_rejecting_models=set(),
        _fast_mode_unavailable_models=set(), _rate_limit_state=None,
        _credential_pool=None, _fallback_chain=[{"provider": "custom", "model": "local"}],
        _fallback_index=0, _fallback_activated=False,
        _vprint=lambda *a, **k: None, _buffer_vprint=lambda *a, **k: None,
        _buffer_diagnostic_status=lambda *a, **k: None,
        _emit_diagnostic_status=lambda *a, **k: None,
        _emit_diagnostic_wait=lambda *a, **k: None,
        _invoke_api_request_error_hook=lambda **k: None,
        _touch_activity=lambda *a, **k: None,
        _summarize_api_error=lambda error: str(error),
        _client_log_context=lambda: "", _is_openrouter_url=lambda: False,
        _recover_with_credential_pool=lambda **k: (False, False),
        _has_pending_fallback=lambda: True,
        _try_activate_fallback=activate_fallback,
        _try_recover_primary_transport=lambda *a, **k: False,
        _adopt_nous_key_before_expiry=lambda: None,
        calls=calls,
    )


def test_handle_api_error_rides_out_the_first_window_and_hands_off_the_second(monkeypatch, caplog):
    """The retry loop's real convention: the handler is entered with the counter the previous
    attempt left (0 on the first error), increments it BEFORE routing (the router sees 1, then 2),
    and rebinds the loop to the verdict's value. If that increment moves again, either the first
    window stops earning its wait or the second stops falling back."""
    from agent.turn_api_error import handle_api_error

    agent = _handler_agent()
    waited: list[float] = []
    monkeypatch.setattr(turn_api_error, "interruptible_backoff_sleep",
                        lambda _agent, seconds, *a, **k: waited.append(seconds))
    monkeypatch.setattr("agent.conversation_loop._arm_fallback_restart",
                        lambda _agent, messages, prompt, retry: prompt)
    routed: list[int] = []
    real_route = turn_api_error.route_classified_error
    monkeypatch.setattr(
        turn_api_error, "route_classified_error",
        lambda _agent, *a, **k: routed.append(k["retry_count"]) or real_route(_agent, *a, **k),
    )
    state = {"retry_count": 0}

    def call():
        verdict = handle_api_error(
            agent, api_error=_Refusal(13), _retry=TurnRetryState(), thinking_spinner=None,
            messages=[], api_messages=[{"role": "user", "content": "hi"}], api_kwargs={},
            system_message="sys", active_system_prompt="sys", conversation_history=[],
            approx_tokens=1000, retry_count=state["retry_count"], max_retries=3,
            compression_attempts=0, max_compression_attempts=3, api_call_count=1,
            api_request_id="req-1", api_start_time=time.time(), effective_task_id="task-1",
            turn_id="turn-1",
        )
        state["retry_count"] = verdict.retry_count  # what the loop rebinds after the handler
        return verdict

    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        first = call()
    assert first.action == "fallthrough", "a 13s window fits the budget: retry on Qwen4"
    assert agent.calls["fallback"] == []
    assert waited == [pytest.approx(13.0)], "the handler must wait the declared window"
    assert any("13.0s fits the 120s primary wait budget" in r.getMessage() for r in caplog.records)

    second = call()
    assert routed == [1, 2], (
        "handle_api_error must increment before routing: the first failure reaches the router as 1 "
        "(the fits-budget guard's bound), the second as 2 (the hand-off)"
    )
    assert second.action == "break"
    assert [c["reason"] for c in agent.calls["fallback"]] == [FailoverReason.rate_limit]
