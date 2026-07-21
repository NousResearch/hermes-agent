"""#68771: retryable 5xx (502/503/500/529) triggers fallback chain.

Source-level + classification tests ensuring server_error and overloaded
are included in the _should_fallback gate alongside rate_limit and transport.
"""

from __future__ import annotations

import inspect

import pytest

from agent.conversation_loop import run_conversation
from agent.error_classifier import FailoverReason


def test_should_fallback_includes_retryable_5xx():
    """The _should_fallback expression in conversation_loop must include
    FailoverReason.server_error and FailoverReason.overloaded."""
    src = inspect.getsource(run_conversation)
    assert "is_retryable_5xx" in src, "is_retryable_5xx variable not found"
    assert "FailoverReason.server_error" in src, "server_error not in fallback gate"
    assert "FailoverReason.overloaded" in src, "overloaded not in fallback gate"
    assert "is_retryable_5xx and retry_count >= 1" in src, \
        "5xx fallback gate missing retry_count >= 1 condition"


def test_should_fallback_keeps_rate_limit_and_transport():
    """Existing triggers must still be present."""
    src = inspect.getsource(run_conversation)
    assert "is_rate_limited" in src
    assert "_is_transport_failure and retry_count >= 2" in src


def test_503_classified_as_overloaded():
    """503 must classify as overloaded (retryable 5xx for fallback)."""
    from agent.error_classifier import classify_api_error

    class _Err(Exception):
        status_code = 503
        response = None

    result = classify_api_error(_Err("Service Unavailable"), provider="nous")
    assert result.reason == FailoverReason.overloaded


def test_502_classified_as_server_error():
    """502 must classify as server_error (retryable 5xx for fallback)."""
    from agent.error_classifier import classify_api_error

    class _Err(Exception):
        status_code = 502
        response = None

    result = classify_api_error(_Err("Bad Gateway"), provider="openrouter")
    assert result.reason == FailoverReason.server_error


def test_500_classified_as_server_error():
    """500 must classify as server_error (retryable 5xx for fallback)."""
    from agent.error_classifier import classify_api_error

    class _Err(Exception):
        status_code = 500
        response = None

    result = classify_api_error(_Err("Internal Server Error"), provider="anthropic")
    assert result.reason == FailoverReason.server_error


def test_5xx_fallback_status_message_present():
    """A status message for 5xx fallback must exist in the source."""
    src = inspect.getsource(run_conversation)
    assert "Provider returned HTTP" in src, \
        "5xx fallback status message not found"