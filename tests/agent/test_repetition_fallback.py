"""A runaway-repetition stream cut must escalate to the fallback chain before ending the turn.

The cut is Hermes' own guard firing on a degenerate model (flash tiers loop in the reasoning
channel), so retrying the primary re-enters the loop — the same escalation the content-filter
path already uses. Without a fallback (or when activation fails, or outside a continuable
api_mode) the turn ends with the repetition error card, exactly as before.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import recover_from_truncation

_USER_TURN = {"role": "user", "content": "hi"}


def _agent(fallback_chain: list[dict[str, str]] | None, activated: bool = True,
           api_mode: str = "chat_completions"):
    agent = SimpleNamespace()
    agent.model = "z-ai/glm-5.3-flash"
    agent.log_prefix = ""
    agent.api_mode = api_mode
    agent._fallback_chain = fallback_chain or []
    agent._fallback_index = 0
    agent._session_messages = []
    agent._vprint = MagicMock()
    agent._emit_diagnostic_status = MagicMock()
    agent._persist_session = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._get_messages_up_to_last_assistant = MagicMock(
        side_effect=lambda messages: messages[:-1]
    )

    def _activate(reason=None, reset_at=None):
        if not agent._fallback_chain:
            return False
        if activated:
            agent._fallback_index = len(agent._fallback_chain)
            return True
        return False

    agent._try_activate_fallback = MagicMock(side_effect=_activate)
    return agent


def _recover(agent, *, partial: bool = False):
    response = SimpleNamespace(_runaway_repetition=True)
    _retry = TurnRetryState()
    _retry.primary_recovery_attempted = True
    messages: list = [_USER_TURN]
    if partial:
        messages.append({
            "role": "assistant", "content": "partial", "_length_continuation_nudge": True,
        })
    tv = recover_from_truncation(
        agent, response, "length", _retry, messages=messages,
        conversation_history=None, api_kwargs={}, api_call_count=1,
        effective_task_id=None, current_turn_user_idx=0,
        length_continue_retries=0, truncated_response_parts=[("partial", True)] if partial else [],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )
    return tv, _retry


def test_runaway_cut_escalates_to_fallback_and_restarts():
    agent = _agent([{"provider": "openrouter", "model": "z-ai/glm-5.3"}])
    tv, _retry = _recover(agent, partial=True)

    assert tv.action == "break"
    assert agent._try_activate_fallback.called
    assert agent._fallback_index == 1
    assert agent._emit_diagnostic_status.called
    # Restart signals: the loop re-issues the call against the fallback provider.
    assert _retry.restart_with_rebuilt_messages is True
    assert _retry.primary_recovery_attempted is False
    # The partial fragment was rolled back to the last clean turn: the real
    # _get_messages_up_to_last_assistant decided the surviving messages.
    assert agent._get_messages_up_to_last_assistant.called
    assert tv.messages == [_USER_TURN]
    assert agent._session_messages == tv.messages
    assert all(
        "_length_continuation_nudge" not in frag and "_length_continuation_fragment" not in frag
        for frag in tv.messages if isinstance(frag, dict)
    )
    assert tv.result is None


def test_runaway_cut_resets_truncation_counters_on_restart():
    agent = _agent([{"provider": "openrouter", "model": "z-ai/glm-5.3"}])
    tv, _ = _recover(agent)

    assert tv.action == "break"
    assert tv.length_continue_retries == 0
    assert tv.truncated_response_parts == []
    assert tv.retry_count == 0
    assert tv.compression_attempts == 0


def test_runaway_cut_with_exhausted_chain_ends_turn_with_repetition_error():
    agent = _agent([{"provider": "openrouter", "model": "z-ai/glm-5.3"}])
    agent._fallback_index = len(agent._fallback_chain)
    tv, _retry = _recover(agent)

    assert tv.action == "return"
    assert not agent._try_activate_fallback.called
    assert "repetition loop" in tv.result["error"]
    assert tv.result["completed"] is False
    assert tv.result["failure_reason"] == "truncated"
    assert tv.result["failure_retryable"] is True


def test_runaway_cut_with_no_chain_configured_ends_turn_with_repetition_error():
    agent = _agent(None)
    tv, _ = _recover(agent)

    assert tv.action == "return"
    assert not agent._try_activate_fallback.called
    assert "repetition loop" in tv.result["error"]


def test_runaway_cut_when_activation_fails_ends_turn_with_repetition_error():
    agent = _agent([{"provider": "openrouter", "model": "z-ai/glm-5.3"}], activated=False)
    tv, _ = _recover(agent)

    assert tv.action == "return"
    assert agent._try_activate_fallback.called
    assert "repetition loop" in tv.result["error"]


def test_runaway_cut_outside_continuable_api_mode_never_escalates():
    agent = _agent([{"provider": "openrouter", "model": "z-ai/glm-5.3"}],
                   api_mode="custom_noncontinuable")
    tv, _ = _recover(agent)

    assert tv.action == "return"
    assert not agent._try_activate_fallback.called
    assert "repetition loop" in tv.result["error"]
