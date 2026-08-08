"""Regression tests for bare vendor model selection with aggregators.

Issue #56465: ``/model mimo-v2.5`` should not switch from an authenticated
aggregator to direct Xiaomi when Xiaomi credentials are missing.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_ACCEPTED = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _runtime_resolver_without_xiaomi(**kwargs):
    requested = kwargs.get("requested")
    if requested == "xiaomi":
        raise RuntimeError("XIAOMI_API_KEY is not configured")
    return {
        "api_key": f"{requested}-token",
        "base_url": f"https://{requested}.example/v1",
        "api_mode": "chat_completions",
    }


def _runtime_resolver_with_xiaomi(**kwargs):
    requested = kwargs.get("requested")
    return {
        "api_key": f"{requested}-token",
        "base_url": f"https://{requested}.example/v1",
        "api_mode": "chat_completions",
    }


def _run_switch(
    *,
    raw_input: str,
    current_provider: str,
    explicit_provider: str = "",
    runtime_resolver=_runtime_resolver_without_xiaomi,
    catalog: list[str] | None = None,
):
    with patch("hermes_cli.model_switch.list_provider_models", return_value=catalog or []), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=runtime_resolver), \
         patch("hermes_cli.models.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="old-model",
            current_api_key=f"{current_provider}-token",
            explicit_provider=explicit_provider,
        )


def test_bare_mimo_stays_on_nous_when_xiaomi_credentials_missing():
    result = _run_switch(raw_input="mimo-v2.5", current_provider="nous")

    assert result.success is True, result.error_message
    assert result.target_provider == "nous"
    assert result.new_model == "xiaomi/mimo-v2.5"
    assert result.api_key == "nous-token"


def test_bare_mimo_stays_on_openrouter_when_xiaomi_credentials_missing():
    result = _run_switch(raw_input="mimo-v2.5", current_provider="openrouter")

    assert result.success is True, result.error_message
    assert result.target_provider == "openrouter"
    assert result.new_model == "xiaomi/mimo-v2.5"
    assert result.api_key == "openrouter-token"


def test_explicit_xiaomi_still_reports_missing_credentials():
    result = _run_switch(
        raw_input="mimo-v2.5",
        current_provider="nous",
        explicit_provider="xiaomi",
    )

    assert result.success is False
    assert result.target_provider == "xiaomi"
    assert "Could not resolve credentials for provider 'Xiaomi MiMo'" in result.error_message
    assert "XIAOMI_API_KEY is not configured" in result.error_message


def test_bare_mimo_uses_xiaomi_when_credentials_available():
    result = _run_switch(
        raw_input="mimo-v2.5",
        current_provider="nous",
        runtime_resolver=_runtime_resolver_with_xiaomi,
    )

    assert result.success is True, result.error_message
    assert result.target_provider == "xiaomi"
    assert result.new_model == "mimo-v2.5"
    assert result.api_key == "xiaomi-token"


def test_mimo_alias_still_uses_aggregator_catalog_match():
    result = _run_switch(
        raw_input="mimo",
        current_provider="openrouter",
        catalog=["xiaomi/mimo-v2.5-pro"],
    )

    assert result.success is True, result.error_message
    assert result.target_provider == "openrouter"
    assert result.new_model == "xiaomi/mimo-v2.5-pro"
    assert result.resolved_via_alias == "mimo"
