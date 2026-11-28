"""handle_content_policy_refusal: an Anthropic refusal with an empty body still tells the user why."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import handle_content_policy_refusal


def _agent():
    import agent.transports.anthropic
    from agent.transports import get_transport

    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.log_prefix = "anthropic_messages", "anthropic", "claude", ""
    agent._is_anthropic_oauth = False
    agent._get_transport.return_value = get_transport("anthropic_messages")
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._extract_reasoning.return_value = ""
    agent._refusal_rephrase_tried = True  # budget spent: exercise the terminal path
    agent._refusal_rephrase_saved_esp = None
    agent.ephemeral_system_prompt = ""
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


def test_refusal_retries_with_rephrase_no_provider_switch(caplog):
    """A first refusal never switches providers (#160728): it arms one
    same-provider rephrase retry, and the log keeps the native stop reason."""
    agent = _agent()
    agent._refusal_rephrase_tried = False
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    retry_state = TurnRetryState()
    with caplog.at_level("WARNING", logger="agent.conversation_loop"):
        verdict = handle_content_policy_refusal(
            agent, response, retry_state, thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
            active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
            api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
        )
    assert verdict.action == "break"
    assert retry_state.restart_with_rebuilt_messages is True
    agent._try_activate_fallback.assert_not_called()
    assert "Risk-control retry directive" in agent.ephemeral_system_prompt
    assert "native_stop_reason=refusal" in caplog.text and "cyber" in caplog.text


def test_refusal_log_caps_a_long_stop_details_explanation(caplog):
    agent = _agent()
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "x" * 2000 + "SENTINEL"},
    )
    with caplog.at_level("WARNING", logger="agent.conversation_loop"):
        handle_content_policy_refusal(
            agent, response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
            active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
            api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
        )
    line = next(r.getMessage() for r in caplog.records if "native_stop_reason" in r.getMessage())
    assert "cyber" in line and "SENTINEL" not in line
