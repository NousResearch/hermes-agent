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


def test_has_explicit_refusal_distinguishes_stream_drop_from_real_refusal():
    from agent.turn_truncation import has_explicit_refusal

    drop_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(refusal=None))], stop_reason=None)
    drop_norm = SimpleNamespace(provider_data={})
    assert has_explicit_refusal(drop_resp, drop_norm) is False

    openai_refusal = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(refusal="I cannot help"))], stop_reason=None)
    assert has_explicit_refusal(openai_refusal, drop_norm) is True

    anthropic_refusal = SimpleNamespace(choices=None, stop_reason="refusal")
    assert has_explicit_refusal(anthropic_refusal, drop_norm) is True

    provider_data_refusal = SimpleNamespace(provider_data={"stop_details": {"type": "refusal"}})
    assert has_explicit_refusal(drop_resp, provider_data_refusal) is True


def test_empty_content_filter_without_explicit_refusal_does_not_leak_reasoning():
    agent = _agent()
    agent._extract_reasoning.return_value = "internal chain of thought before stream dropped"
    drop_resp = SimpleNamespace(
        content=[], stop_reason="end_turn", usage=None, model="claude", stop_details=None,
    )
    verdict = handle_content_policy_refusal(
        agent, drop_resp, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
        api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
    )
    assert verdict.action == "return"
    assert "internal chain of thought" not in verdict.result["error"]
