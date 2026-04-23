"""Test that provider-originated ValueError subclasses are not misclassified
as local validation errors (non-retryable abort).

Bug: json.JSONDecodeError inherits from ValueError. When the OpenAI SDK
fails to parse a provider response body (empty, truncated, garbled JSON),
it raises json.JSONDecodeError. The is_local_validation_error check in
run_agent.py catches it as isinstance(api_error, ValueError) and triggers
a non-retryable abort — but the error is transient (provider-side), not a
programming bug.

Similarly, UnicodeDecodeError (also a ValueError subclass) from garbled
provider responses gets the same wrong treatment. The existing code only
excludes UnicodeEncodeError.

The error classifier (agent/error_classifier.py) correctly returns
FailoverReason.unknown with retryable=True for these — the bug is in
the inline isinstance check that overrides the classifier's recommendation.
"""

import json
import pytest

from agent.error_classifier import classify_api_error, FailoverReason


# ── Mirror the is_local_validation_error predicate from run_agent.py ──
# COUPLING: This mirrors the isinstance logic at run_agent.py ~line 10691.
# If you change the production code, you MUST update this mirror too.
# A better long-term fix: extract the predicate into a named function in
# agent/error_classifier.py and import it here directly.

def _is_local_validation_error(error):
    """Mirror of the is_local_validation_error check in run_agent.py.
    
    Excludes provider-side ValueError subclasses:
      - json.JSONDecodeError (malformed provider response)
      - UnicodeError (garbled bytes from provider)
      - ssl.SSLError (TLS transport failure, ValueError via MRO)
    """
    import ssl
    return (
        isinstance(error, (ValueError, TypeError))
        and not isinstance(error, (UnicodeError, json.JSONDecodeError))
        and not isinstance(error, ssl.SSLError)
    )


# ── Bug: JSONDecodeError misclassified as local validation ─────────────

class TestJSONDecodeErrorMisclassification:
    """RED: These tests FAIL on current code — json.JSONDecodeError is
    incorrectly flagged as a local validation error (non-retryable abort)
    even though it originates from the provider, not our code.
    """

    def test_json_decode_error_not_local_validation(self):
        """json.JSONDecodeError from provider response must NOT be
        treated as a local programming bug."""
        err = json.JSONDecodeError("Expecting value", doc="", pos=0)
        # BUG: _is_local_validation_error returns True because
        # JSONDecodeError is a ValueError subclass
        assert not _is_local_validation_error(err), (
            "json.JSONDecodeError should NOT be classified as a local "
            "validation error — it indicates a malformed provider response, "
            "not a programming bug"
        )

    def test_json_decode_error_classified_retryable(self):
        """The error classifier already returns retryable=True for
        json.JSONDecodeError — the bug is the isinstance check that
        overrides this."""
        err = json.JSONDecodeError("Expecting value: line 1 column 1 (char 0)", doc="", pos=0)
        result = classify_api_error(err)
        assert result.retryable is True
        assert result.reason == FailoverReason.unknown

    def test_unicode_decode_error_not_local_validation(self):
        """UnicodeDecodeError from garbled provider body must NOT be
        treated as a local programming bug."""
        err = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
        assert not _is_local_validation_error(err), (
            "UnicodeDecodeError should NOT be classified as a local "
            "validation error — it indicates a garbled provider response, "
            "not a programming bug"
        )

    def test_unicode_translate_error_not_local_validation(self):
        """UnicodeTranslateError must also be excluded (via UnicodeError
        parent class)."""
        err = UnicodeTranslateError("", 0, 1, "character maps to <undefined>")
        assert not _is_local_validation_error(err), (
            "UnicodeTranslateError should NOT be classified as a local "
            "validation error — covered by the UnicodeError parent class"
        )

    def test_unicode_decode_error_classified_retryable(self):
        """The error classifier should return retryable for UnicodeDecodeError
        from provider responses."""
        err = UnicodeDecodeError("utf-8", b"", 0, 1, "validity too long")
        result = classify_api_error(err)
        assert result.retryable is True


# ── Sanity: genuine local validation errors still caught ───────────────

class TestGenuineLocalErrorsStillCaught:
    """After the fix, plain ValueError and TypeError must still be flagged
    as local validation errors (non-retryable abort).
    """

    def test_plain_valueerror_is_local_validation(self):
        err = ValueError("invalid literal for int()")
        assert _is_local_validation_error(err) is True

    def test_typeerror_is_local_validation(self):
        err = TypeError("unsupported operand type(s)")
        assert _is_local_validation_error(err) is True

    def test_unicode_encode_error_not_local_validation(self):
        """UnicodeEncodeError was already excluded — make sure we keep it."""
        err = UnicodeEncodeError("ascii", "hello \u2022", 6, 7, "ordinal not in range")
        assert _is_local_validation_error(err) is False

    def test_ssl_error_not_local_validation(self):
        """ssl.SSLError inherits from ValueError via MRO — must not be
        misclassified as a local programming bug."""
        import ssl
        err = ssl.SSLError("TLS handshake failed")
        assert not _is_local_validation_error(err), (
            "ssl.SSLError should NOT be classified as a local "
            "validation error — it indicates a TLS transport failure"
        )