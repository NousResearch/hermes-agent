"""Tests for anthropic_adapter.extract_stream_error_payload and
format_stream_error_message.

Regression tests for the SDK streaming accumulator behaviour where
Bedrock/Anthropic error events at the start of a stream would surface
as a cryptic "Unexpected event order" RuntimeError with no payload.
"""
import pytest

from anthropic.lib.streaming._messages import accumulate_event
from agent.anthropic_adapter import (
    extract_stream_error_payload,
    format_stream_error_message,
)


def _trigger_sdk_accumulator_error(event_payload):
    """Reproduce the real SDK RuntimeError by calling accumulate_event."""
    try:
        accumulate_event(event=event_payload, current_snapshot=None)
    except RuntimeError as exc:
        return exc
    raise AssertionError("accumulate_event did not raise")


class TestExtractStreamErrorPayload:
    """Recover the original service-side error event from the SDK's
    RuntimeError traceback."""

    def test_recovers_overloaded_error_payload(self):
        exc = _trigger_sdk_accumulator_error({
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Bedrock is currently overloaded, please retry",
            },
        })
        payload = extract_stream_error_payload(exc)

        assert payload is not None
        assert payload["event_type"] == "error"
        assert payload["error"]["type"] == "overloaded_error"
        assert "overloaded" in payload["error"]["message"]

    def test_recovers_rate_limit_error_payload(self):
        exc = _trigger_sdk_accumulator_error({
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Number of request tokens has exceeded your per-minute rate limit",
            },
        })
        payload = extract_stream_error_payload(exc)

        assert payload is not None
        assert payload["error"]["type"] == "rate_limit_error"

    def test_returns_none_for_unrelated_runtime_error(self):
        """Don't accidentally match arbitrary RuntimeErrors."""
        exc = RuntimeError("something completely different")
        assert extract_stream_error_payload(exc) is None

    def test_returns_none_for_non_runtime_error(self):
        """Reject non-RuntimeError exceptions even with matching message."""
        exc = ValueError("Unexpected event order, got error before \"message_start\"")
        assert extract_stream_error_payload(exc) is None

    def test_returns_none_for_runtime_error_without_traceback(self):
        """Unraised RuntimeError (no traceback) returns None, not crash."""
        exc = RuntimeError("Unexpected event order, got error before \"message_start\"")
        assert extract_stream_error_payload(exc) is None


class TestFormatStreamErrorMessage:
    """Format the recovered payload into a human-readable error message."""

    def test_formats_typed_error_with_type_and_message(self):
        exc = RuntimeError("Unexpected event order, got error")
        payload = {
            "event_type": "error",
            "error": {"type": "overloaded_error", "message": "Please retry"},
        }
        msg = format_stream_error_message(exc, payload)

        assert "overloaded_error" in msg
        assert "Please retry" in msg
        assert "anthropic SDK" in msg  # original SDK message preserved as context

    def test_formats_payload_with_only_message(self):
        exc = RuntimeError("Unexpected event order")
        payload = {"error": {"message": "bare message"}}
        msg = format_stream_error_message(exc, payload)

        assert "bare message" in msg

    def test_formats_payload_with_only_type(self):
        exc = RuntimeError("Unexpected event order")
        payload = {"error": {"type": "invalid_request_error"}}
        msg = format_stream_error_message(exc, payload)

        assert "invalid_request_error" in msg

    def test_formats_raw_event_fallback(self):
        exc = RuntimeError("Unexpected event order")
        payload = {"raw_event": {"type": "unknown", "foo": "bar"}}
        msg = format_stream_error_message(exc, payload)

        assert "unknown" in msg or "foo" in msg


class TestEndToEndRecovery:
    """End-to-end: SDK raises, extract+format produces an actionable message."""

    def test_sdk_error_event_becomes_readable_message(self):
        exc = _trigger_sdk_accumulator_error({
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Bedrock is currently overloaded, please retry",
            },
        })
        payload = extract_stream_error_payload(exc)
        msg = format_stream_error_message(exc, payload)

        # Message now contains the actual root cause, not just the SDK's
        # cryptic fallback
        assert "overloaded_error" in msg
        assert "Bedrock is currently overloaded" in msg
        assert "anthropic SDK:" in msg
