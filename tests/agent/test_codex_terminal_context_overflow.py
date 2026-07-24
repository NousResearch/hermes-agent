"""Regression tests for terminal Codex Responses context-overflow recovery."""

from types import SimpleNamespace

import pytest

from agent.conversation_loop import (
    _CodexTerminalResponseError,
    _codex_terminal_context_error,
)
from agent.error_classifier import FailoverReason, classify_api_error


@pytest.mark.parametrize("code", ["context_length_exceeded", "max_tokens_exceeded"])
def test_terminal_codex_context_failure_requests_compression(code):
    """A terminal Responses failure must enter compression recovery, not invalid-response retry."""
    response = SimpleNamespace(
        status="failed",
        error=SimpleNamespace(
            code=code,
            message="The request exceeds the model context window.",
        ),
    )

    error = _codex_terminal_context_error(response)

    assert error is not None
    classified = classify_api_error(error)
    assert classified.reason is FailoverReason.context_overflow
    assert classified.retryable is True
    assert classified.should_compress is True


def test_structured_terminal_code_survives_message_formatter_changes():
    """Recovery must use the machine code rather than depend on display wording."""
    error = _CodexTerminalResponseError(
        "synthetic terminal failure",
        code="context_length_exceeded",
    )

    classified = classify_api_error(error)

    assert classified.reason is FailoverReason.context_overflow
    assert classified.should_compress is True


@pytest.mark.parametrize(
    "status,code",
    [
        ("failed", "invalid_api_key"),
        ("failed", "rate_limit_exceeded"),
        ("completed", "context_length_exceeded"),
    ],
)
def test_non_overflow_terminal_responses_do_not_request_compaction(status, code):
    """Unrelated failures and completed responses must stay out of the destructive compaction path."""
    response = SimpleNamespace(
        status=status,
        error=SimpleNamespace(code=code, message="synthetic provider response"),
    )

    assert _codex_terminal_context_error(response) is None
