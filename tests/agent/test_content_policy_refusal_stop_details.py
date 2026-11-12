"""handle_content_policy_refusal: an Anthropic refusal with an empty body still tells the user why."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_agent.agent.turn_retry_state import TurnRetryState
from hermes_agent.agent.turn_truncation import handle_content_policy_refusal


def _agent():
    import hermes_agent.agent.transports.anthropic
    from hermes_agent.agent.transports import get_transport

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


def test_refusal_that_falls_back_names_the_cause(caplog):
    """A refusal answered by the fallback still says why: the switch notice gets the
    content-policy reason (not "provider failure") and agent.log keeps the native stop reason."""
    from hermes_agent.agent.error_classifier import FailoverReason

    agent = _agent()
    agent._has_pending_fallback.return_value = True
    agent._try_activate_fallback.return_value = True
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    with caplog.at_level("WARNING", logger="hermes_agent.agent.conversation_loop"):
        verdict = handle_content_policy_refusal(
            agent, response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
            active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
            api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
        )
    assert verdict.action == "break"
    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.content_policy_blocked)
    assert "native_stop_reason=refusal" in caplog.text and "cyber" in caplog.text


def test_refusal_log_caps_a_long_stop_details_explanation(caplog):
    agent = _agent()
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "x" * 2000 + "SENTINEL"},
    )
    with caplog.at_level("WARNING", logger="hermes_agent.agent.conversation_loop"):
        handle_content_policy_refusal(
            agent, response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
            active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
            api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
        )
    line = next(r.getMessage() for r in caplog.records if "native_stop_reason" in r.getMessage())
    assert "cyber" in line and "SENTINEL" not in line
