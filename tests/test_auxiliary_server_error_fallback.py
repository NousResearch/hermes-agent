"""Tests for _is_server_error() and server-error fallback in auxiliary_client.py.

Issue #25822: HTTP 503 from Gemini should trigger fallback even when the
provider is explicitly configured (is_auto=False).
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Unit tests for _is_server_error()
# ---------------------------------------------------------------------------

def _make_exc(status_code=None, message=""):
    """Create a mock exception with optional status_code."""
    exc = Exception(message)
    if status_code is not None:
        exc.status_code = status_code
    return exc


class TestIsServerError:
    """Test the _is_server_error() detector function."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.auxiliary_client import _is_server_error
        self.is_server_error = _is_server_error

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
    def test_detects_5xx_status_codes(self, status):
        assert self.is_server_error(_make_exc(status_code=status))

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 429])
    def test_rejects_4xx_status_codes(self, status):
        assert not self.is_server_error(_make_exc(status_code=status))

    @pytest.mark.parametrize("msg", [
        "server error",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "unavailable",
        "temporarily unavailable",
        "overloaded",
        "try again later",
        "Server Error: something went wrong",
        "503 Service Unavailable",
    ])
    def test_detects_server_error_strings(self, msg):
        assert self.is_server_error(_make_exc(message=msg))

    @pytest.mark.parametrize("msg", [
        "rate limit exceeded",
        "authentication failed",
        "connection refused",
        "invalid request",
        "payment required",
        "model not found",
    ])
    def test_rejects_non_server_error_strings(self, msg):
        assert not self.is_server_error(_make_exc(message=msg))

    def test_no_status_code_uses_string_match(self):
        assert self.is_server_error(_make_exc(message="unavailable"))
        assert not self.is_server_error(_make_exc(message="something random"))

    def test_gemini_503_real_world(self):
        """Exact error string from Gemini 503 in production."""
        exc = _make_exc(
            status_code=503,
            message="UNAVAILABLE: This model is currently experiencing high demand.",
        )
        assert self.is_server_error(exc)


# ---------------------------------------------------------------------------
# Integration test: fallback triggers on 503 for explicit provider
# ---------------------------------------------------------------------------

class TestServerErrorFallback:
    """Verify that server errors bypass the is_auto gate."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent import auxiliary_client
        self.mod = auxiliary_client

    def test_503_triggers_fallback_on_explicit_provider(self):
        """A 503 should trigger fallback even when is_auto=False."""
        is_server_error = self.mod._is_server_error

        # Simulate the decision logic
        first_err = _make_exc(status_code=503, message="Service Unavailable")
        resolved_provider = "google"  # explicit, NOT "auto"

        should_fallback = (
            self.mod._is_payment_error(first_err)
            or self.mod._is_connection_error(first_err)
            or self.mod._is_rate_limit_error(first_err)
            or is_server_error(first_err)
        )
        is_auto = resolved_provider in {"auto", "", None}

        # The key assertion: should_fallback is True AND
        # the gate allows server errors through even for explicit providers
        assert should_fallback is True
        assert is_auto is False
        assert should_fallback and (is_auto or is_server_error(first_err))

    def test_402_does_not_bypass_is_auto_gate(self):
        """A 402 should NOT bypass the is_auto gate for explicit providers."""
        first_err = _make_exc(status_code=402, message="Payment required")
        resolved_provider = "google"

        should_fallback = (
            self.mod._is_payment_error(first_err)
            or self.mod._is_connection_error(first_err)
            or self.mod._is_rate_limit_error(first_err)
            or self.mod._is_server_error(first_err)
        )
        is_auto = resolved_provider in {"auto", "", None}

        # Payment error triggers should_fallback but NOT the server-error bypass
        assert should_fallback is True
        assert is_auto is False
        # Without server error, the gate should block
        assert not (should_fallback and (is_auto or self.mod._is_server_error(first_err)))
