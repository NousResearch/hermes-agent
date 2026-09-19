"""404 'insufficient_credits_for_paid_model' triggers the fallback chain (#115702)."""

import logging
from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import log_credit_exhaustion_fallback, route_classified_error
from agent.turn_retry_state import TurnRetryState


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _classified_credit_404():
    e = MockAPIError(
        "Not Found",
        status_code=404,
        body={"error": {"code": "insufficient_credits_for_paid_model", "message": "Not Found"}},
    )
    return classify_api_error(e, provider="nous", model="openai/gpt-5.5-pro")


def test_credit_404_classifies_like_429_exhaustion():
    result = _classified_credit_404()
    assert result.reason == FailoverReason.billing
    assert result.retryable is False
    assert result.should_fallback is True


def test_credit_404_fallback_logs_actionable_error_naming_credits_and_target(caplog):
    agent = SimpleNamespace(model="openai/gpt-5.5-free", provider="nous", log_prefix="")
    with caplog.at_level(logging.ERROR, logger="agent.conversation_loop"):
        log_credit_exhaustion_fallback(agent, _classified_credit_404())
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    text = errors[0].getMessage().lower()
    assert "credit" in text
    assert "insufficient_credits_for_paid_model" in text
    assert "openai/gpt-5.5-free" in text


def test_credit_404_fallback_log_silent_without_marker(caplog):
    agent = SimpleNamespace(model="x", provider="nous", log_prefix="")
    classified = SimpleNamespace(reason=FailoverReason.billing, error_context={})
    with caplog.at_level(logging.ERROR, logger="agent.conversation_loop"):
        log_credit_exhaustion_fallback(agent, classified)
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_credit_404_routes_to_fallback_chain(caplog):
    """End to end through the classifier + eager-fallback routing: the 404
    credit error attempts the fallback_model chain and logs the ERROR."""
    classified = _classified_credit_404()
    calls = []

    agent = SimpleNamespace(
        model="openai/gpt-5.5-pro",
        provider="nous",
        log_prefix="",
        _fallback_index=0,
        _fallback_chain=[{"provider": "nous", "model": "openai/gpt-5.5-free"}],
        _credential_pool=None,
    )

    def _activate(reason=None):
        calls.append(reason)
        agent.model = "openai/gpt-5.5-free"
        return True

    agent._try_activate_fallback = _activate
    agent._buffer_diagnostic_status = lambda msg: calls.append(msg)

    with caplog.at_level(logging.ERROR, logger="agent.conversation_loop"):
        verdict = route_classified_error(
            agent, MockAPIError("Not Found", 404), classified, TurnRetryState(),
            error_msg="Not Found", error_context={}, recovered_with_pool=False,
            base_url="", model="openai/gpt-5.5-pro", messages=[], api_messages=[],
            system_message=None, active_system_prompt="sys", conversation_history=[],
            retry_count=0, max_retries=3, compression_attempts=0,
            max_compression_attempts=2, api_call_count=1, effective_task_id=None,
        )
    assert verdict.action == "break"
    assert FailoverReason.billing in calls
    assert agent.model == "openai/gpt-5.5-free"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    text = errors[0].getMessage().lower()
    assert "credit" in text and "openai/gpt-5.5-free" in text
