"""Regression tests for aggregator model switching with bare vendor models."""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


def test_bare_vendor_model_stays_on_current_aggregator():
    """Bare MiMo IDs on Nous should become aggregator slugs, not direct Xiaomi.

    Regression for #56465: a static native-catalog match for ``mimo-v2.5`` used
    to rewrite ``model.provider`` to ``xiaomi`` before any credential check, so
    users authenticated through Nous OAuth got wedged on the next gateway turn.
    """
    with (
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.models.validate_requested_model",
            return_value={"accepted": True, "persist": True, "recognized": True, "message": ""},
        ),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
    ):
        result = switch_model(
            "mimo-v2.5",
            current_provider="nous",
            current_model="openai/gpt-5.4",
            current_api_key="oauth-token",
            current_base_url="https://inference-api.nousresearch.com/v1",
        )

    assert result.success
    assert result.target_provider == "nous"
    assert result.provider_changed is False
    assert result.new_model == "xiaomi/mimo-v2.5"
