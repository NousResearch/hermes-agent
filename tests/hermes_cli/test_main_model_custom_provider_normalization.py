"""Dashboard main-model writes preserve declared provider identities."""

from unittest.mock import patch

from hermes_cli.web_server_config import _normalize_main_model_assignment


def _normalize(config, provider, model="vendor/model-a"):
    with patch("hermes_cli.config.load_config", return_value=config):
        return _normalize_main_model_assignment(provider, model)


def test_providers_block_keeps_declared_bare_slug():
    result = _normalize(
        {"providers": {"commandcode": {"base_url": "http://localhost:55990/v1"}}},
        "commandcode",
    )

    assert result == ("commandcode", "vendor/model-a")


def test_custom_provider_name_canonicalizes_to_durable_slug():
    config = {
        "custom_providers": [
            {"name": "US Azure", "base_url": "http://localhost:18025/v1"}
        ]
    }

    assert _normalize(config, "US Azure") == ("custom:us-azure", "vendor/model-a")
    assert _normalize(config, "custom:us-azure") == (
        "custom:us-azure",
        "vendor/model-a",
    )


def test_unknown_vendor_still_uses_aggregator_fallback():
    with patch("hermes_cli.models_detect.provider_has_credentials", lambda p: p == "openrouter"):
        assert _normalize({}, "unconfigured-vendor") == (
            "openrouter",
            "vendor/model-a",
        )


def test_unknown_vendor_without_openrouter_key_is_not_reassigned():
    """No key for the guessed aggregator → keep the pair as sent instead of persisting a provider
    the user never selected."""
    with patch("hermes_cli.models_detect.provider_has_credentials", lambda p: False):
        assert _normalize({}, "unconfigured-vendor") == ("unconfigured-vendor", "vendor/model-a")