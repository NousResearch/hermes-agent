"""Regression tests for the Anthropic-Messages-API ``/v1/models`` fallback warning.

When a user runs ``/model <name>`` against an endpoint whose ``api_mode`` is
``anthropic_messages`` and that endpoint does not implement ``GET /v1/models``,
Hermes used to print a generic warning that started with "Many Anthropic-
compatible proxies do not implement GET /v1/models".

That wording confused users who had just typed a non-Anthropic-looking model
string (e.g. ``openai-codex/gpt-5.5`` while still on a Claude-flavored proxy):
the message implied the model itself was Anthropic-related when in fact only
the *currently selected provider* was Anthropic-shaped.

This test pins the improved wording which now names the active provider and
endpoint explicitly so the warning makes sense regardless of what the user
typed.
"""

from unittest.mock import patch

from hermes_cli.models import validate_requested_model


def _stub_probe(*_args, **_kwargs):
    return {
        "models": None,
        "probed_url": "http://127.0.0.1:18801/v1/models",
        "resolved_base_url": "http://127.0.0.1:18801/v1",
        "suggested_base_url": None,
        "used_fallback": False,
    }


def test_anthropic_messages_warning_names_provider_and_endpoint():
    """The fallback warning must name the active provider and endpoint, not
    just say "Anthropic-compatible proxies" generically.
    """
    with patch("hermes_cli.models.fetch_api_models", return_value=None), \
         patch("hermes_cli.models.probe_api_models", side_effect=_stub_probe):
        result = validate_requested_model(
            "openai-codex/gpt-5.5",
            "claude-api-proxy",
            api_key="sk-test",
            base_url="http://127.0.0.1:18801/v1",
            api_mode="anthropic_messages",
        )

    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is False

    message = result["message"]
    # The improved warning must name BOTH the active provider and endpoint.
    assert "claude-api-proxy" in message, message
    assert "http://127.0.0.1:18801/v1" in message, message
    # It must still explain why verification failed (Anthropic Messages API).
    assert "Anthropic Messages API" in message, message
    # It must not use the old vague wording that confused users.
    assert "Many Anthropic-compatible proxies" not in message, message


def test_anthropic_messages_warning_handles_missing_base_url():
    """If base_url is not provided we must still produce a sensible message
    rather than embedding ``None`` or a blank.
    """
    with patch("hermes_cli.models.fetch_api_models", return_value=None), \
         patch("hermes_cli.models.probe_api_models", side_effect=_stub_probe):
        result = validate_requested_model(
            "gpt-5.5",
            "claude-api-proxy",
            api_key="sk-test",
            base_url=None,
            api_mode="anthropic_messages",
        )

    message = result["message"]
    assert "None" not in message, message
    assert "the configured endpoint" in message, message
