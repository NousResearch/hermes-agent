"""Tests for the email allowlist feature of SelfHostedOIDCProvider.

Covers:
  - Mixed-case matching
  - Empty and missing email claims
  - Malformed non-empty allowlist (fail-closed)
  - Unverified email claim rejection
  - Backward compatibility (unset = allow any)
"""

import pytest
from unittest.mock import patch, MagicMock

# These imports are guarded so the test file can be collected even if the
# plugin module is not on the path in certain CI environments.
try:
    from plugins.dashboard_auth.self_hosted import SelfHostedOIDCProvider
    from hermes_cli.dashboard_auth import ProviderError
except ImportError:
    pytest.skip("dashboard_auth not available", allow_module_level=True)

_CLAIMS_BASE = {
    "sub": "user-123",
    "email": "alice@example.com",
    "email_verified": True,
    "exp": 9999999999,
    "iss": "https://idp.example.com",
    "aud": "client-id",
}


def _make_provider(allowed_emails=""):
    return SelfHostedOIDCProvider(
        issuer="https://idp.example.com",
        client_id="client-id",
        allowed_emails=allowed_emails,
    )


class TestEmailAllowlistMatching:
    """Email matching behavior."""

    def test_exact_match_accepted(self):
        p = _make_provider("alice@example.com")
        session = p._session_from_tokens(
            id_token="tok", refresh_token="", claims=dict(_CLAIMS_BASE)
        )
        assert session.email == "alice@example.com"

    def test_mixed_case_match_accepted(self):
        p = _make_provider("Alice@Example.COM")
        session = p._session_from_tokens(
            id_token="tok", refresh_token="", claims=dict(_CLAIMS_BASE)
        )
        assert session.email == "alice@example.com"

    def test_claim_uppercase_accepted(self):
        p = _make_provider("alice@example.com")
        claims = dict(_CLAIMS_BASE)
        claims["email"] = "ALICE@EXAMPLE.COM"
        session = p._session_from_tokens(
            id_token="tok", refresh_token="", claims=claims
        )
        assert session is not None

    def test_non_match_rejected(self):
        p = _make_provider("alice@example.com")
        claims = dict(_CLAIMS_BASE)
        claims["email"] = "bob@example.com"
        with pytest.raises(ProviderError, match="not in the allowed list"):
            p._session_from_tokens(id_token="tok", refresh_token="", claims=claims)

    def test_multiple_emails_match(self):
        p = _make_provider("alice@example.com,bob@example.com")
        claims = dict(_CLAIMS_BASE)
        claims["email"] = "bob@example.com"
        session = p._session_from_tokens(
            id_token="tok", refresh_token="", claims=claims
        )
        assert session.email == "bob@example.com"


class TestEmptyAndMissingEmail:
    """Edge cases for empty/missing email claims."""

    def test_missing_email_with_allowlist_rejected(self):
        p = _make_provider("alice@example.com")
        claims = dict(_CLAIMS_BASE)
        del claims["email"]
        with pytest.raises(ProviderError):
            p._session_from_tokens(id_token="tok", refresh_token="", claims=claims)

    def test_empty_email_with_allowlist_rejected(self):
        p = _make_provider("alice@example.com")
        claims = dict(_CLAIMS_BASE)
        claims["email"] = ""
        with pytest.raises(ProviderError):
            p._session_from_tokens(id_token="tok", refresh_token="", claims=claims)

    def test_missing_email_without_allowlist_accepted(self):
        p = _make_provider("")  # no allowlist
        claims = dict(_CLAIMS_BASE)
        del claims["email"]
        session = p._session_from_tokens(
            id_token="tok", refresh_token="", claims=claims
        )
        assert session.email == ""


class TestMalformedAllowlist:
    """Malformed non-empty allowlist must fail closed."""

    def test_comma_only_raises_value_error(self):
        with pytest.raises(ValueError, match="no valid email"):
            _make_provider(",")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="no valid email"):
            _make_provider("   ")

    def test_empty_string_is_ok(self):
        p = _make_provider("")
        assert p._allowed_emails == set()


class TestEmailVerified:
    """Unverified email claims should be rejected."""

    def test_unverified_email_rejected(self):
        p = _make_provider("alice@example.com")
        claims = dict(_CLAIMS_BASE)
        claims["email_verified"] = False
        with pytest.raises(ProviderError, match="not verified"):
            p._session_from_tokens(id_token="tok", refresh_token="", claims=claims)

    def test_verified_email_accepted(self):
        p = _make_provider("alice@example.com")
        claims = dict(_CLAIMS_BASE)
        claims["email_verified"] = True
        session = p._session_from_tokens(
            id_token="tok", refresh_token="", claims=claims
        )
        assert session.email == "alice@example.com"

    def test_missing_email_verified_accepted(self):
        """If IDP doesn't send email_verified, don't enforce it."""
        p = _make_provider("alice@example.com")
        claims = dict(_CLAIMS_BASE)
        del claims["email_verified"]
        session = p._session_from_tokens(
            id_token="tok", refresh_token="", claims=claims
        )
        assert session.email == "alice@example.com"
