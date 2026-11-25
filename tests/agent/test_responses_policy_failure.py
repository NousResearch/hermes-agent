"""Explicit Responses refusals stop the real turn without retry or provider fallback."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from run_agent import AIAgent


@pytest.mark.parametrize("code", ["cyber_policy", "content_policy_violation", "content_filter"])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("output", [[], [SimpleNamespace(type="message", role="assistant", content=[])]])
def test_failed_policy_response_is_terminal(code, typed, output):
    agent = AIAgent(
        model="primary-model", provider="custom", base_url="http://127.0.0.1:1/v1", api_key="test-key",
        api_mode="codex_responses", enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
        skip_memory=True, fallback_model={"provider": "custom", "model": "backup-model",
            "base_url": "http://127.0.0.1:1/v1", "api_key": "test-key", "api_mode": "codex_responses"},
    )
    agent._disable_streaming = True
    agent.compression_enabled = False
    error = {"code": code, "message": "Provider policy rejection"}
    response = SimpleNamespace(status="failed", error=SimpleNamespace(**error) if typed else error,
                               output=output, output_text="")
    agent._interruptible_api_call = Mock(return_value=response)
    result = agent.run_conversation("Compare two fictional game consoles", conversation_history=[])
    assert agent._interruptible_api_call.call_count == 1
    assert agent.model == "primary-model"
    assert result["failed"] is True
    assert result["failure_retryable"] is False
    assert "safety filter" in result["final_response"]
    assert "Provider policy rejection" in result["final_response"]
    assert "Invalid API response" not in result["final_response"]


@pytest.mark.parametrize("status,output,valid", [
    ("completed", [], False),
    ("failed", [], False),
    ("completed", [SimpleNamespace(type="message", role="assistant", content=[])], True),
])
def test_nonpolicy_response_validation_is_preserved(status, output, valid):
    from agent.transports.codex import ResponsesApiTransport

    response = SimpleNamespace(status=status, output=output,
                               error={"code": "server_error", "message": "Temporarily unavailable"})
    assert ResponsesApiTransport().validate_response(response) is valid


def test_auxiliary_normalization_preserves_structured_refusal():
    from agent.codex_responses_adapter import _normalize_codex_response, ResponsesPolicyError
    from agent.error_classifier import classify_api_error, FailoverReason

    response = SimpleNamespace(status="failed", output=[], error={"code": "cyber_policy", "message": "Refused"})
    with pytest.raises(ResponsesPolicyError) as raised:
        _normalize_codex_response(response)
    verdict = classify_api_error(raised.value)
    assert verdict.reason == FailoverReason.content_policy_blocked
    assert not verdict.retryable and not verdict.should_fallback
