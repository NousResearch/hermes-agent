"""Tests for brokerage configuration and domain models."""

from brokerage.config import BrokerageSettings


def test_brokerage_settings_defaults_to_paper_mode_and_local_service():
    settings = BrokerageSettings()
    assert settings.service_url == "http://127.0.0.1:8787"
    assert settings.default_account_mode == "paper"
    assert settings.confirmation_ttl_seconds == 120
