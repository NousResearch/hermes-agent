"""handle_content_policy_refusal: an Anthropic refusal with an empty body still tells the user why."""
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_empty_refusal_reports_stop_details_explanation(caplog):
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "general_harms", "explanation": "classifier halt"},
    )
    with caplog.at_level("WARNING", logger="agent.turn_truncation"):
        verdict = handle_content_policy_refusal(
            _agent(), response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
            active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
            api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
        )
    assert verdict.action == "return"
    assert verdict.result["error"] == "content_policy_blocked: classifier halt"
    assert "native_stop_reason=refusal" in caplog.text and "general_harms" in caplog.text
