"""Regression tests for #56465 — vendor-prefix models must not pin unauthed natives."""

from unittest.mock import patch

import pytest

from hermes_cli import models
from hermes_cli.model_switch import switch_model


def test_detect_provider_for_model_prefers_authenticated_nous_over_xiaomi(monkeypatch):
    monkeypatch.setattr(
        models,
        "_aggregator_catalog_models",
        lambda provider: ["xiaomi/mimo-v2.5"] if provider == "nous" else [],
    )

    result = models.detect_provider_for_model(
        "mimo-v2.5",
        "nous",
        authenticated_providers=["nous"],
    )

    assert result == ("nous", "xiaomi/mimo-v2.5")


def test_detect_provider_for_model_skips_unauthed_xiaomi_when_nous_authed(monkeypatch):
    monkeypatch.setattr(models, "_aggregator_catalog_models", lambda provider: [])

    result = models.detect_provider_for_model(
        "mimo-v2.5",
        "nous",
        authenticated_providers=["nous"],
    )

    assert result is None


def test_detect_provider_for_model_keeps_xiaomi_when_authed():
    result = models.detect_provider_for_model(
        "mimo-v2.5",
        "nous",
        authenticated_providers=["nous", "xiaomi"],
    )

    assert result == ("xiaomi", "mimo-v2.5")


def test_switch_model_mimo_v25_stays_on_nous_without_xiaomi_key():
    runtime = {
        "api_key": "nous-oauth-token",
        "base_url": "https://api.nousresearch.com/v1",
        "api_mode": "",
    }

    with patch(
        "hermes_cli.model_switch.get_authenticated_provider_slugs",
        return_value=["nous"],
    ), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=runtime,
    ), patch(
        "hermes_cli.models.validate_requested_model",
        return_value={
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": "",
        },
    ):
        result = switch_model(
            raw_input="mimo-v2.5",
            current_provider="nous",
            current_model="deepseek/deepseek-v4-flash",
            is_global=False,
        )

    assert result.success, result.error_message
    assert result.target_provider == "nous"
    assert result.new_model == "xiaomi/mimo-v2.5"
