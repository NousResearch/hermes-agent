"""Tests for Nous Portal URL validation (Issue #44710)."""
import pytest
from hermes_cli.auth import _validate_nous_portal_url


class TestNousPortalURLValidation:
    """Test _validate_nous_portal_url() host allowlist."""

    def test_valid_portal_url(self):
        """Valid portal.nousresearch.com URL should pass through."""
        result = _validate_nous_portal_url("https://portal.nousresearch.com/api/oauth/token")
        assert result == "https://portal.nousresearch.com/api/oauth/token"

    def test_valid_portal_url_trailing_slash(self):
        """Trailing slashes should be stripped."""
        result = _validate_nous_portal_url("https://portal.nousresearch.com/")
        assert result == "https://portal.nousresearch.com"

    def test_invalid_host_api(self):
        """api.nousresearch.com should be rejected (Issue #44710)."""
        result = _validate_nous_portal_url("https://api.nousresearch.com/api/oauth/token")
        assert result is None

    def test_non_https_rejected(self):
        """Non-HTTPS URLs should be rejected."""
        result = _validate_nous_portal_url("http://portal.nousresearch.com/api/oauth/token")
        assert result is None

    def test_none_input(self):
        """None should return None."""
        result = _validate_nous_portal_url(None)
        assert result is None

    def test_empty_string(self):
        """Empty string should return None."""
        result = _validate_nous_portal_url("")
        assert result is None

    def test_whitespace_only(self):
        """Whitespace-only string should return None."""
        result = _validate_nous_portal_url("   ")
        assert result is None

    def test_malformed_url(self):
        """Malformed URL should return None."""
        result = _validate_nous_portal_url("not-a-url!!!")
        assert result is None

    def test_non_string_type(self):
        """Non-string input should return None."""
        result = _validate_nous_portal_url(12345)
        assert result is None

    def test_arbitrary_host_rejected(self):
        """Arbitrary external host should be rejected (SSRF protection)."""
        result = _validate_nous_portal_url("https://evil.com/steal-token")
        assert result is None

    def test_subdomain_variant_rejected(self):
        """Subdomain of portal.nousresearch.com should be rejected."""
        result = _validate_nous_portal_url("https://fake.portal.nousresearch.com/")
        assert result is None

    def test_allowlist_contains_correct_host(self):
        """The allowlist should contain only portal.nousresearch.com."""
        from hermes_cli.auth import _ALLOWED_NOUS_PORTAL_HOSTS
        assert "portal.nousresearch.com" in _ALLOWED_NOUS_PORTAL_HOSTS
        assert "api.nousresearch.com" not in _ALLOWED_NOUS_PORTAL_HOSTS
