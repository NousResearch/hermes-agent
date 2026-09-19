"""A non-retryable API failure names itself instead of promising a second attempt.

A 401 on a static-key route (no credential to refresh, no pool entry to rotate to)
goes straight to the fallback chain, so the log used to read ``API call failed
(attempt 1/3)`` while ``attempt 2/3`` never appeared — which reads as a retry
counter that failed to advance (#73237). The classifier's verdict now rides the
same line on both surfaces (logger + buffered status trace)."""

import logging
import time
from unittest.mock import MagicMock, patch

from agent.turn_recovery import log_api_error_attempt


def _agent():
    agent = MagicMock()
    agent._summarize_api_error.return_value = "HTTP 401 invalid key"
    agent._client_log_context.return_value = "provider=custom"
    agent._is_openrouter_url.return_value = False
    agent.verbose_logging = False
    agent.provider, agent.base_url, agent.model = "custom", "https://x.test/v1", "m"
    return agent


def _call(agent, retryable):
    return log_api_error_attempt(
        agent, RuntimeError("401"), retry_count=1, max_retries=3, status_code=401,
        elapsed_time=0.1, api_messages=[], approx_tokens=10, retryable=retryable,
    )


def test_non_retryable_failure_is_named_on_log_and_status_line(caplog):
    agent = _agent()
    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        _call(agent, retryable=False)
    assert "attempt 1/3, not retryable" in caplog.text
    assert "not retryable" in agent._buffer_vprint.call_args_list[0].args[0]


def test_production_entry_forwards_the_classifier_verdict_to_the_attempt_line(caplog):
    """``handle_api_error`` must hand ``classified.retryable`` to the log line; a bare
    ``attempt 1/3`` there is the exact regression #73237 reported."""
    from types import SimpleNamespace

    from agent.error_classifier import ClassifiedError, FailoverReason
    from agent.turn_api_error import handle_api_error

    agent = _agent()
    agent._interrupt_requested = True  # leave the loop right after the attempt line
    agent.clear_interrupt.return_value = True
    verdict = ClassifiedError(reason=FailoverReason.auth, status_code=401, retryable=False)
    with patch("agent.turn_api_error.classify_api_error", return_value=verdict), patch(
        "agent.turn_api_error.recover_before_classification", return_value=(False, "sys")
    ), patch(
        "agent.turn_api_error.recover_after_classification", return_value=(False, False)
    ), caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        out = handle_api_error(
            agent, api_error=RuntimeError("401"), _retry=SimpleNamespace(), thinking_spinner=None,
            messages=[], api_messages=[], api_kwargs={}, system_message=None,
            active_system_prompt="sys", conversation_history=[], approx_tokens=10, retry_count=0,
            max_retries=3, compression_attempts=0, max_compression_attempts=1, api_call_count=1,
            api_request_id="r", api_start_time=time.time(), effective_task_id=None, turn_id="t",
        )
    assert out.action == "break"
    assert "attempt 1/3, not retryable" in caplog.text
    assert "not retryable" in agent._buffer_vprint.call_args_list[0].args[0]
