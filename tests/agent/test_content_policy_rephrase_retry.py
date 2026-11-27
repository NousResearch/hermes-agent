"""Content-policy refusals never switch providers (#160728).

A refusal arms ONE same-provider rephrase retry (the directive rides the ephemeral system
prompt, so ``messages`` stays untouched); a second refusal in the streak surfaces as terminal.
The fallback chain stays reserved for platform outages — ``try_activate_fallback`` gates
``content_policy_blocked`` centrally, so no call site can switch providers on a refusal.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.error_classifier import FailoverReason
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import (
    _RISK_REPHRASE_NUDGE, arm_refusal_rephrase_retry, clear_refusal_rephrase_nudge,
    handle_content_policy_refusal,
)


def _agent() -> MagicMock:
    from agent.transports import get_transport

    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.log_prefix = "anthropic_messages", "anthropic", "claude", ""
    agent._is_anthropic_oauth = False
    agent._get_transport.return_value = get_transport("anthropic_messages")
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._extract_reasoning.return_value = ""
    agent._session_db = None
    agent.ephemeral_system_prompt = ""
    agent._refusal_rephrase_saved_esp = None
    agent._refusal_rephrase_tried = False
    return agent


def _refusal_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "high risk"})


def _run(agent, retry_state):
    return handle_content_policy_refusal(
        agent, _refusal_response(), retry_state, thinking_spinner=None, messages=[], api_messages=[],
        api_kwargs={}, active_system_prompt=None, conversation_history=[], api_call_count=1,
        effective_task_id="t", turn_id="u", api_request_id="r", api_start_time=0.0,
        retry_count=0, max_retries=0, current_turn_user_idx=0)


def test_first_refusal_arms_rephrase_retry_on_the_same_provider(caplog):
    import logging

    agent = _agent()
    retry_state = TurnRetryState()
    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        verdict = _run(agent, retry_state)
    assert verdict.action == "break"
    assert retry_state.restart_with_rebuilt_messages is True
    agent._try_activate_fallback.assert_not_called()
    assert agent._refusal_rephrase_tried is True
    assert _RISK_REPHRASE_NUDGE in agent.ephemeral_system_prompt


def test_second_refusal_in_the_streak_is_terminal():
    agent = _agent()
    verdict = _run(agent, TurnRetryState())          # first refusal → rephrase retry armed
    assert verdict.action == "break"
    verdict = _run(agent, TurnRetryState())          # retried request refused again → terminal
    assert verdict.action == "return"
    assert verdict.result is not None and "high risk" in verdict.result["error"]
    agent._try_activate_fallback.assert_not_called()
    # Streak over: budget refreshed for the next turn, no nudge leaks forward.
    assert agent._refusal_rephrase_tried is False
    assert agent.ephemeral_system_prompt == ""


def test_rephrase_nudge_is_one_shot_and_restores_the_prior_prompt():
    agent = _agent()
    agent.ephemeral_system_prompt = "skills prompt"
    retry_state = TurnRetryState()
    assert arm_refusal_rephrase_retry(agent, retry_state) is True
    assert agent.ephemeral_system_prompt == "skills prompt\n\n" + _RISK_REPHRASE_NUDGE
    assert arm_refusal_rephrase_retry(agent, retry_state) is False  # one per streak
    clear_refusal_rephrase_nudge(agent)
    assert agent.ephemeral_system_prompt == "skills prompt"
    assert agent._refusal_rephrase_saved_esp is None


def test_fallback_gate_suppresses_content_policy_but_not_outages(caplog):
    """The central gate: content_policy_blocked can never reach the fallback chain, while
    platform-outage reasons (rate limit etc.) pass through to the switch machinery."""
    import logging

    from agent.chat_completion_helpers import try_activate_fallback

    agent = _agent()
    agent._fallback_chain, agent._fallback_index = [{"provider": "custom", "model": "fb"}], 0
    with caplog.at_level(logging.INFO, logger="agent.chat_completion_helpers"):
        assert try_activate_fallback(agent, reason=FailoverReason.content_policy_blocked) is False
    assert agent._fallback_index == 0  # chain untouched
    assert any("Fallback suppressed for content_policy_blocked" in r.getMessage() for r in caplog.records)

    # An outage reason is NOT gated: the suppression line must not appear for it.
    agent2 = _agent()
    agent2._fallback_chain, agent2._fallback_index = [], 0
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="agent.chat_completion_helpers"):
        try_activate_fallback(agent2, reason=FailoverReason.rate_limit)
    assert not any("Fallback suppressed" in r.getMessage() for r in caplog.records)
