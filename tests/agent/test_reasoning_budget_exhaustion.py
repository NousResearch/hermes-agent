"""Reasoning output exhaustion must not compress the input context (#158133)."""
from unittest.mock import Mock, patch

import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_overflow import recover_from_overflow
from agent.turn_retry_state import TurnRetryState


_MESSAGE = (
    "The output token budget was exhausted during reasoning before an answer was produced. "
    "Increase max_tokens or reduce reasoning effort before submitting a new request."
)


@pytest.mark.parametrize("status", [None, 400, 500, 503])
@pytest.mark.parametrize("message", [_MESSAGE, _MESSAGE.split(". ")[0].upper()])
def test_reasoning_budget_exhaustion_is_incomplete_not_overflow(status, message):
    error = openai.APIError(message, request=None, body=None)
    error.status_code = status
    verdict = classify_api_error(error, provider="custom", model="claude-opus-5-5",
                                 approx_tokens=183_949, context_length=200_000, num_messages=216)
    assert verdict.reason is FailoverReason.incomplete_response
    assert (verdict.should_compress, verdict.retryable, verdict.should_fallback) == (False, False, True)
    assert verdict.should_rotate_credential is False


@pytest.mark.parametrize("status", [None, 400, 500, 503])
@pytest.mark.parametrize("message", [
    "input tokens + max_tokens exceeds context length",
    "max_tokens is too large: input tokens + max_tokens exceed the context window",
])
def test_real_context_overflow_with_max_tokens_still_compresses(status, message):
    error = openai.APIError(message, request=None, body=None)
    error.status_code = status
    verdict = classify_api_error(error, provider="custom")
    assert verdict.reason is FailoverReason.context_overflow
    assert verdict.should_compress is True


def test_empty_error_is_not_reasoning_exhaustion():
    verdict = classify_api_error(openai.APIError("", request=None, body=None), provider="custom")
    assert verdict.reason is FailoverReason.unknown


class _Agent:
    log_prefix = ""
    verbose = False
    provider = "custom"
    model = "claude-opus-5-5"
    base_url = ""
    _credential_pool = None

    def __init__(self, fallback):
        self._compress_context = Mock(side_effect=AssertionError("must not compress"))
        self._has_pending_fallback = Mock(return_value=fallback)
        self._try_activate_fallback = Mock(return_value=fallback)

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.mark.parametrize("fallback", [False, True])
def test_reasoning_exhaustion_skips_compression_and_uses_existing_fallback(fallback):
    agent = _Agent(fallback)
    error = openai.APIError(_MESSAGE, request=None, body=None)
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    retry = TurnRetryState()
    messages = [{"role": "user", "content": "hello"}]
    overflow = recover_from_overflow(
        agent, error, classified, retry, status_code=None, error_msg=str(error),
        wrapped_output_cap_budget=None, messages=messages, api_messages=messages,
        system_message="", active_system_prompt="", conversation_history=None,
        approx_tokens=10, compression_attempts=0, max_compression_attempts=2,
        api_call_count=1, effective_task_id=None,
    )
    assert overflow.action == "fallthrough"
    assert overflow.is_context_length_error is False
    with patch("agent.conversation_loop._credential_refresh_profile", return_value=None), patch(
        "agent.conversation_loop._arm_fallback_restart", return_value=""
    ):
        result = settle_unrecovered_error(
            agent, api_error=error, classified=classified, _retry=retry, status_code=None,
            error_msg=str(error), is_context_length_error=overflow.is_context_length_error,
            is_rate_limited=False, _is_zai_coding_overload=False, _provider=agent.provider,
            _base=agent.base_url, _model=agent.model, messages=messages, api_messages=messages,
            api_kwargs={}, active_system_prompt="", conversation_history=None, approx_tokens=10,
            retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
        )
    agent._try_activate_fallback.assert_called_once_with(
        reason=FailoverReason.incomplete_response, reset_at=None,
    )
    agent._compress_context.assert_not_called()
    assert result.action == ("break" if fallback else "return")
    if not fallback:
        assert result.result["failure_reason"] == "incomplete_response"
