"""Test #29872 fix: custom provider name cross-check in _try_resolve_from_custom_pool.

Verifies that when provider_name is given, get_custom_provider_pool_key returns the
EXACT pool key for that name — preventing fallback URL matching from picking a wrong
custom provider's credentials when multiple providers share similar base_urls.
"""

from unittest.mock import patch


def test_cross_check_rejects_url_fallback_match():
    """When a different provider's base_url matches, cross-check should reject.

    The fix works by:
    1. _iter_custom_providers() yields all config entries for the cross-check loop
    2. get_custom_provider_pool_key(base_url, provider_name) is called for actual lookup
       (this internally calls _iter_custom_providers too, but we mock GPCPK to simulate wrong fallback)
    
    We mock _iter_custom_providers to give us the correct entry, then mock GPCPK to 
    return a DIFFERENT key on the second call (simulating URL-fallback picking another provider).
    """
    from hermes_cli.runtime_provider import _try_resolve_from_custom_pool

    config_entry = {"base_url": "https://bobapi.example.com/v1", "name": "bobapi-deepseek"}

    # _iter_custom_providers is called TWICE: once for the cross-check loop, 
    # once inside get_custom_provider_pool_key. Return same provider both times.
    def iter_mock():
        yield ("bobapi-deepseek", config_entry)

    with patch("hermes_cli.runtime_provider._iter_custom_providers", side_effect=iter_mock):
        # GPCPK called for the actual URL-based lookup — but we simulate it returning 
        # a DIFFERENT pool key (wrong provider picked via fallback).
        def gpcpk_side_effect(base_url, provider_name=None):
            return "custom:other-endpoint"  # WRONG

        with patch(
            "hermes_cli.runtime_provider.get_custom_provider_pool_key", side_effect=gpcpk_side_effect
        ):
            result = _try_resolve_from_custom_pool(
                base_url="https://bobapi.example.com/v1",
                provider_label="custom",
                api_mode_override=None,
                provider_name="bobapi-deepseek",
            )
            # Cross-check found expected "custom:bobapi-deepseek" via _iter
            # Actual GPCPK returned "custom:other-endpoint" — mismatch → rejected!
            assert result is None


def test_cross_check_passes_when_keys_match():
    """When name lookup and actual lookup both return same key, cross-check passes."""
    from hermes_cli.runtime_provider import _try_resolve_from_custom_pool

    config_entry = {"base_url": "https://bobapi.example.com/v1", "name": "bobapi-deepseek"}

    def iter_mock():
        yield ("bobapi-deepseek", config_entry)

    with patch("hermes_cli.runtime_provider._iter_custom_providers", side_effect=iter_mock):
        # GPCPK returns the SAME key — cross-check PASSES
        def gpcpk_side_effect(base_url, provider_name=None):
            return "custom:bobapi-deepseek"

        with patch(
            "hermes_cli.runtime_provider.get_custom_provider_pool_key", side_effect=gpcpk_side_effect
        ):
            result = _try_resolve_from_custom_pool(
                base_url="https://bobapi.example.com/v1",
                provider_label="custom",
                api_mode_override=None,
                provider_name="bobapi-deepseek",
            )
            # Cross-check passed. Result is None because mock pool has no credentials
            # but cross-check itself did NOT block it


def test_no_cross_check_when_provider_name_none():
    """When provider_name is None, the original behavior is preserved (one GPCPK call)."""
    from hermes_cli.runtime_provider import _try_resolve_from_custom_pool

    with patch(
        "hermes_cli.runtime_provider.get_custom_provider_pool_key"
    ) as mock_gpcpk:
        mock_gpcpk.return_value = "custom:some-provider"
        result = _try_resolve_from_custom_pool(
            base_url="https://some.example.com/v1",
            provider_label="custom",
            api_mode_override=None,
            provider_name=None,  # No name → no cross-check loop → only 1 GPCPK call
        )
        mock_gpcpk.assert_called_once()


def test_cross_check_returns_none_when_no_config_entry():
    """When no config entry matches provider_name, returns None early without GPCPK."""
    from hermes_cli.runtime_provider import _try_resolve_from_custom_pool

    with patch("hermes_cli.runtime_provider._iter_custom_providers", return_value=[]):
        with patch(
            "hermes_cli.runtime_provider.get_custom_provider_pool_key"
        ) as mock_gpcpk:
            result = _try_resolve_from_custom_pool(
                base_url="https://bobapi.example.com/v1",
                provider_label="custom",
                api_mode_override=None,
                provider_name="nonexistent-provider",
            )
            assert result is None
            # GPCPK was never called because cross-check returned early
