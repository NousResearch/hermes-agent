"""handle_content_policy_refusal: an Anthropic refusal with an empty body still tells the user why."""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.error_classifier import FailoverReason
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import handle_content_policy_refusal


def _agent():
    import agent.transports.anthropic  # noqa: F401
    from agent.transports import get_transport

    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.log_prefix = "anthropic_messages", "anthropic", "claude", ""
    agent._is_anthropic_oauth = False
    agent._get_transport.return_value = get_transport("anthropic_messages")
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._extract_reasoning.return_value = ""
    return agent


def test_empty_refusal_reports_stop_details_explanation():
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "general_harms", "explanation": "classifier halt"},
    )
    verdict = handle_content_policy_refusal(
        _agent(), response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
        api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
    )
    assert verdict.action == "return"
    assert "classifier halt" in verdict.result["error"]


def test_refusal_that_falls_back_still_logs_native_stop_reason(caplog):
    """When a fallback is configured the function breaks early — the native cause must
    still be recorded, and the failover labelled content_policy_blocked, not the generic
    "provider failure" (#124874)."""
    agent = _agent()
    agent._has_pending_fallback.return_value = True
    agent._try_activate_fallback.return_value = True
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "general_harms", "explanation": "classifier halt"},
    )
    with caplog.at_level(logging.WARNING):
        verdict = handle_content_policy_refusal(
            agent, response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
            active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
            api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
        )
    assert verdict.action == "break"
    # Cause recorded before the fallback even though the terminal warning is skipped.
    messages = [r.getMessage() for r in caplog.records]
    assert any("native_stop_reason=refusal" in m and "general_harms" in m for m in messages)
    # Failover activated with the accurate, specific reason.
    _, kwargs = agent._try_activate_fallback.call_args
    assert kwargs.get("reason") == FailoverReason.content_policy_blocked
