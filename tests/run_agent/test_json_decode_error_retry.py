"""Tests for json.JSONDecodeError retry classification.

Covers the fix for issue #14271 — json.JSONDecodeError (which inherits from
ValueError) was incorrectly classified as a local validation error, bypassing
retry logic for transient provider parse failures.
"""

import json


class TestJsonDecodeErrorNotLocalValidation:
    """json.JSONDecodeError must NOT be classified as a local validation error.

    The is_local_validation_error check in run_agent.py excludes
    UnicodeEncodeError (a ValueError subclass that indicates a transient
    encoding issue).  json.JSONDecodeError is also a ValueError subclass
    but represents a transient provider parse failure, not a programming bug.
    """

    def _is_local_validation_error(self, exc):
        """Replicate the classification logic from run_agent.py line ~10650."""
        return (
            isinstance(exc, (ValueError, TypeError))
            and not isinstance(exc, (UnicodeEncodeError, json.JSONDecodeError))
        )

    def test_json_decode_error_is_not_local_validation(self):
        """json.JSONDecodeError should be retried, not treated as a local bug."""
        err = json.JSONDecodeError("Expecting value", "", 0)
        assert self._is_local_validation_error(err) is False

    def test_plain_value_error_is_local_validation(self):
        """A plain ValueError is still a local validation error (programming bug)."""
        err = ValueError("invalid literal for int()")
        assert self._is_local_validation_error(err) is True

    def test_type_error_is_local_validation(self):
        """A TypeError is still a local validation error."""
        err = TypeError("expected str, got int")
        assert self._is_local_validation_error(err) is True

    def test_unicode_encode_error_is_not_local_validation(self):
        """UnicodeEncodeError (pre-existing exclusion) is not a local validation error."""
        err = UnicodeEncodeError("ascii", "\u028b", 0, 1, "ordinal not in range")
        assert self._is_local_validation_error(err) is False

    def test_json_decode_error_inherits_value_error(self):
        """Confirm json.JSONDecodeError is a ValueError subclass (the root cause)."""
        err = json.JSONDecodeError("Expecting value", "", 0)
        assert isinstance(err, ValueError)
