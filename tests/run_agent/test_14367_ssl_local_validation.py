"""Tests for #14367 — ssl.SSLCertVerificationError mis-classified as local validation error.

ssl.SSLCertVerificationError inherits from SSLError → OSError → ValueError.
The is_local_validation_error guard in run_agent.py caught it via isinstance(e, ValueError)
and triggered a non-retryable abort, even though the error classifier returns retryable=True
for all OSError subclasses (transport heuristic, step 7).

Fix: add ssl.SSLError to the exclusion tuple alongside UnicodeEncodeError.
"""

import ssl


# ── Helper: replicate the fixed guard from run_agent.py ───────────────────

def _is_local_validation_error(api_error: BaseException) -> bool:
    """Mirror of the is_local_validation_error check in run_agent.py."""
    return (
        isinstance(api_error, (ValueError, TypeError))
        and not isinstance(api_error, (UnicodeEncodeError, ssl.SSLError))
    )


# ── Tests ──────────────────────────────────────────────────────────────────

class TestSSLErrorNotLocalValidation:
    """ssl.SSLError and its subclasses must not be treated as local
    validation errors — they are transport errors and should be retried."""

    def test_ssl_cert_verification_error_is_not_local(self):
        """SSLCertVerificationError (the reported bug case) should not be
        classified as a local validation error."""
        e = ssl.SSLCertVerificationError("certificate verify failed: self-signed certificate")
        assert not _is_local_validation_error(e)

    def test_ssl_error_base_is_not_local(self):
        """ssl.SSLError itself (parent of all SSL errors) should not be
        classified as a local validation error."""
        e = ssl.SSLError("unknown SSL error")
        assert not _is_local_validation_error(e)

    def test_ssl_cert_verification_error_is_value_error_via_mro(self):
        """Confirm the MRO bug exists: SSLCertVerificationError IS a ValueError,
        which is why the pre-fix guard misfired."""
        e = ssl.SSLCertVerificationError("cert error")
        assert isinstance(e, ValueError), (
            "Prerequisite: SSLCertVerificationError must subclass ValueError via MRO"
        )


class TestExistingExclusionsPreserved:
    """Existing exclusions (UnicodeEncodeError) must remain effective."""

    def test_unicode_encode_error_is_not_local(self):
        """Pre-existing: UnicodeEncodeError should not be a local validation error."""
        e = UnicodeEncodeError("utf-8", "x", 0, 1, "invalid char")
        assert not _is_local_validation_error(e)


class TestValueErrorAndTypeErrorStillLocal:
    """Plain ValueError and TypeError remain local validation errors — the fix
    must not weaken detection of real programming bugs."""

    def test_plain_value_error_is_local(self):
        e = ValueError("unexpected keyword argument 'stream_options'")
        assert _is_local_validation_error(e)

    def test_plain_type_error_is_local(self):
        e = TypeError("got an unexpected keyword argument 'stream_options'")
        assert _is_local_validation_error(e)

    def test_json_decode_error_is_still_local(self):
        """json.JSONDecodeError is a ValueError subclass — still local until
        #14366 lands its own exclusion."""
        import json
        e = json.JSONDecodeError("Expecting value", "doc", 0)
        assert _is_local_validation_error(e)
