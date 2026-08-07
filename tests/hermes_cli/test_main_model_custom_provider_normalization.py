"""Dashboard main-model writes preserve declared provider identities."""

from unittest.mock import patch

from hermes_cli.web_server import _normalize_main_model_assignment


def _normalize(config, provider, model="vendor/model-a"):
    with patch("hermes_cli.web_server.load_config", return_value=config):
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
    assert _normalize({}, "unconfigured-vendor") == (
        "openrouter",
        "vendor/model-a",
    )