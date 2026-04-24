"""Tests for _normalize_custom_provider_entry placeholder URL handling (#14457).

URLs containing {placeholder} tokens are validated before expansion,
causing them to be rejected as invalid URLs and the provider silently dropped.
"""
import pytest


class TestPlaceholderURLAccepted:
    """Placeholder URLs must be accepted without validation (#14457)."""

    def _normalize(self, entry, key="test-provider"):
        from hermes_cli.config import _normalize_custom_provider_entry
        return _normalize_custom_provider_entry(entry, provider_key=key)

    def test_placeholder_url_accepted(self):
        """URLs with {placeholder} tokens should not be rejected."""
        result = self._normalize({"base_url": "{base_url}/v1", "name": "test"})
        assert result is not None
        assert result["base_url"] == "{base_url}/v1"

    def test_env_var_placeholder_accepted(self):
        result = self._normalize({"base_url": "${API_ENDPOINT}/v1", "name": "test"})
        assert result is not None

    def test_region_placeholder_accepted(self):
        result = self._normalize({"base_url": "https://{region}.api.example.com/v1", "name": "test"})
        assert result is not None

    def test_normal_url_still_works(self):
        result = self._normalize({"base_url": "https://api.example.com/v1", "name": "test"})
        assert result is not None
        assert result["base_url"] == "https://api.example.com/v1"

    def test_invalid_url_without_placeholder_still_rejected(self):
        """Non-placeholder URLs without scheme/host should still be rejected."""
        result = self._normalize({"base_url": "not-a-url", "name": "test"})
        assert result is None
