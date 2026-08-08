"""Cron credential pool key scoping for custom providers (#78427)."""

from unittest.mock import patch

from cron.scheduler import _resolve_cron_credential_pool_key


def test_named_provider_passes_through():
    assert _resolve_cron_credential_pool_key({"provider": "openai"}) == "openai"
    assert _resolve_cron_credential_pool_key({"provider": "OpenRouter"}) == "openrouter"


def test_empty_provider_returns_none():
    assert _resolve_cron_credential_pool_key({}) is None
    assert _resolve_cron_credential_pool_key({"provider": ""}) is None


def test_custom_provider_scopes_by_name_and_base_url():
    runtime = {
        "provider": "custom",
        "requested_provider": "custom:longcat",
        "base_url": "https://longcat.example/v1",
    }
    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:longcat",
    ) as mock_key:
        assert _resolve_cron_credential_pool_key(runtime) == "custom:longcat"
        mock_key.assert_called_once_with(
            "https://longcat.example/v1",
            provider_name="longcat",
        )


def test_custom_provider_uses_requested_name_without_prefix():
    runtime = {
        "provider": "custom",
        "requested_provider": "LongCat",
        "base_url": "https://longcat.example/v1",
    }
    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:longcat",
    ) as mock_key:
        assert _resolve_cron_credential_pool_key(runtime) == "custom:longcat"
        mock_key.assert_called_once_with(
            "https://longcat.example/v1",
            provider_name="LongCat",
        )


def test_custom_provider_without_match_does_not_return_bare_custom():
    """Regression: bare 'custom' must never be returned — that merges all pools."""
    runtime = {
        "provider": "custom",
        "requested_provider": "custom",
        "base_url": "https://unknown.example/v1",
    }
    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value=None,
    ):
        assert _resolve_cron_credential_pool_key(runtime) is None
