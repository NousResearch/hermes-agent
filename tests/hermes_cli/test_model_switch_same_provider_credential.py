"""Regression tests for same-provider model switch credential fallback.

When resolve_runtime_provider() returns an empty api_key/base_url during a
same-provider switch, the current working credentials should be preserved
instead of being overwritten with empty strings.

Regression test for issue #44490: opencode-go same-provider switch produces
empty api_key because resolve_runtime_provider() does not re-resolve env-var
credentials for the same provider.
"""

import pytest
from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


class TestSameProviderCredentialFallback:
    """Same-provider switch should preserve current credentials when resolution
    returns empty values."""

    @patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
    @patch("hermes_cli.model_switch.get_model_info", return_value=None)
    @patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_empty_resolved_key_falls_back_to_current(
        self, mock_resolve, mock_validate, mock_info, mock_caps
    ):
        """When resolve_runtime_provider returns empty api_key, keep current key."""
        mock_resolve.return_value = {
            "api_key": "",
            "base_url": "https://api.opencode-go.com/v1",
            "api_mode": "chat",
        }

        result = switch_model(
            raw_input="kimi-k2.5",
            current_provider="opencode-go",
            current_model="mimo-v2.5",
            current_api_key="***",
            current_base_url="https://api.opencode-go.com/v1",
        )

        assert result.success is True
        assert result.api_key == "***"

    @patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
    @patch("hermes_cli.model_switch.get_model_info", return_value=None)
    @patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_empty_resolved_base_falls_back_to_current(
        self, mock_resolve, mock_validate, mock_info, mock_caps
    ):
        """When resolve_runtime_provider returns empty base_url, keep current."""
        mock_resolve.return_value = {
            "api_key": "sk-new-key",
            "base_url": "",
            "api_mode": "chat",
        }

        result = switch_model(
            raw_input="kimi-k2.5",
            current_provider="opencode-go",
            current_model="mimo-v2.5",
            current_api_key="***",
            current_base_url="https://api.opencode-go.com/v1",
        )

        assert result.success is True
        assert result.base_url == "https://api.opencode-go.com/v1"

    @patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
    @patch("hermes_cli.model_switch.get_model_info", return_value=None)
    @patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_non_empty_resolved_values_preferred_over_current(
        self, mock_resolve, mock_validate, mock_info, mock_caps
    ):
        """When resolve returns non-empty values, use them (credential rotation)."""
        mock_resolve.return_value = {
            "api_key": "***",
            "base_url": "https://api.opencode-go.com/v2",
            "api_mode": "chat",
        }

        result = switch_model(
            raw_input="kimi-k2.5",
            current_provider="opencode-go",
            current_model="mimo-v2.5",
            current_api_key="sk-old-key",
            current_base_url="https://api.opencode-go.com/v1",
        )

        assert result.success is True
        assert result.api_key == "***"
        assert result.base_url == "https://api.opencode-go.com/v2"

    @patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
    @patch("hermes_cli.model_switch.get_model_info", return_value=None)
    @patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_both_empty_falls_back_to_both_current(
        self, mock_resolve, mock_validate, mock_info, mock_caps
    ):
        """When resolve returns both api_key and base_url as empty, keep both current."""
        mock_resolve.return_value = {
            "api_key": "",
            "base_url": "",
            "api_mode": "",
        }

        result = switch_model(
            raw_input="kimi-k2.5",
            current_provider="opencode-go",
            current_model="mimo-v2.5",
            current_api_key="***",
            current_base_url="https://api.opencode-go.com/v1",
        )

        assert result.success is True
        assert result.api_key == "***"
        assert result.base_url == "https://api.opencode-go.com/v1"
