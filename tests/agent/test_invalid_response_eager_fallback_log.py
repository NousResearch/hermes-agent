"""``retry_invalid_response`` eager-fallback trace: the "switching to fallback" WARNING
must fire only when a fallback actually activates.

The guard ``_fallback_index < len(_fallback_chain)`` admits chains that
``_try_activate_fallback`` then declines (deferred-by-reset, or every candidate
unconfigured). Logging the switch from the guard produced a false WARNING — repeated
every retry — on a turn that never switched (#124874 follow-up)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import agent.conversation_loop as _cl
import agent.turn_recovery as _tr
from agent.turn_response_check import retry_invalid_response
from agent.turn_retry_state import TurnRetryState


def _agent(activate: bool) -> MagicMock:
    ag = MagicMock()
    ag.log_prefix = ""
    ag.provider = "primary-provider"
    ag._fallback_index = 0
    ag._fallback_chain = ["fallback-entry"]
    ag._has_pending_fallback.return_value = True
    ag._try_activate_fallback.return_value = activate
    ag._clean_error_message.side_effect = lambda m: m
    return ag


def _neutralize(monkeypatch):
    # No Codex soft failure, so the pool-rotation branch is skipped.
    monkeypatch.setattr(_tr, "classify_codex_soft_failure", lambda agent, response: (None, {}))
    monkeypatch.setattr(_tr, "describe_invalid_response", lambda agent, response, d: ("msg", "primary-provider", "hint"))
    monkeypatch.setattr(_tr, "interruptible_backoff_sleep", lambda *a, **k: None)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(_cl, "_arm_fallback_restart", lambda agent, api_messages, asp, retry: asp)


def _run(ag, **over):
    kw = dict(
        response=SimpleNamespace(error=None), error_details=["boom"],
        _retry=SimpleNamespace(restart_with_redirected_messages=False, has_retried_429=False),
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs={}, active_system_prompt=None,
        conversation_history=[], retry_count=0, max_retries=5, compression_attempts=0, api_call_count=1,
        api_request_id="r", api_start_time=0.0, api_duration=0.0, effective_task_id="t", turn_id="u",
    )
    kw.update(over)
    return retry_invalid_response(ag, **kw)


def test_eager_fallback_log_fires_on_actual_switch(monkeypatch, caplog):
    _neutralize(monkeypatch)
    ag = _agent(activate=True)
    with caplog.at_level(logging.WARNING):
        verdict = _run(ag)
    assert verdict.action == "break"
    switch_logs = [r.getMessage() for r in caplog.records if "switching to fallback" in r.getMessage()]
    assert len(switch_logs) == 1
    # The trace names the primary provider whose response was invalid, not the swapped-in one.
    assert "primary-provider" in switch_logs[0]


def test_no_eager_fallback_log_when_activation_declined(monkeypatch, caplog):
    _neutralize(monkeypatch)
    ag = _agent(activate=False)
    with caplog.at_level(logging.WARNING):
        verdict = _run(ag)
    # Guard admitted the chain, but the switch never happened — no false "switching" trace.
    assert verdict.action != "break"
    assert not any("switching to fallback" in r.getMessage() for r in caplog.records)
