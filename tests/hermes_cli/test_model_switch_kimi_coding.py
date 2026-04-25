"""Regression tests for Kimi /v1 stripping during /model switch.

When switching to an Anthropic-routed Kimi model mid-session
(``/model kimi-for-coding`` on kimi-coding), the resolved base_url must
have its trailing ``/v1`` stripped before being handed to the Anthropic SDK.

Without the strip, the SDK prepends its own ``/v1/messages`` path and
requests hit ``https://api.kimi.com/coding/v1/v1/messages`` — a double
``/v1`` that returns Kimi's 404 error.

``hermes_cli.runtime_provider.resolve_runtime_provider`` already strips
``/v1`` at fresh agent init, but the ``/model`` mid-session switch path in
``hermes_cli.model_switch.switch_model`` was missing the same logic.
"""

from unittest.mock import patch

import pytest

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _run_kimi_switch(
    raw_input: str,
    current_provider: str,
    current_model: str,
    current_base_url: str,
    explicit_provider: str = "",
    runtime_base_url: str = "",
):
    """Run switch_model with Kimi mocks and return the result."""
    effective_runtime_base = runtime_base_url or current_base_url
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "sk-kimi-fake",
                "base_url": effective_runtime_base,
                "api_mode": "anthropic_messages",
            },
        ),
        patch(
            "hermes_cli.models.validate_requested_model",
            return_value=_MOCK_VALIDATION,
        ),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key="sk-kimi-fake",
            explicit_provider=explicit_provider,
        )


class TestKimiCodingV1Strip:
    """kimi-coding: ``/model kimi-for-coding`` must strip /v1."""

    def test_switch_to_kimi_for_coding_strips_v1(self):
        """OpenRouter → Kimi: base_url loses trailing /v1."""
        result = _run_kimi_switch(
            raw_input="kimi-for-coding",
            current_provider="kimi-coding",
            current_model="kimi-for-coding",
            current_base_url="https://api.kimi.com/coding/v1",
        )

        assert result.success, f"switch_model failed: {result.error_message}"
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://api.kimi.com/coding", (
            f"Expected /v1 stripped for anthropic_messages; got {result.base_url}"
        )

    def test_switch_to_kimi_k2_turbo_strips_v1(self):
        """Same behavior for kimi-k2-turbo-preview."""
        result = _run_kimi_switch(
            raw_input="kimi-k2-turbo-preview",
            current_provider="kimi-coding",
            current_model="kimi-for-coding",
            current_base_url="https://api.kimi.com/coding/v1",
        )

        assert result.success
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://api.kimi.com/coding"

    def test_trailing_slash_also_stripped(self):
        """``/v1/`` with trailing slash is also stripped cleanly."""
        result = _run_kimi_switch(
            raw_input="kimi-for-coding",
            current_provider="kimi-coding",
            current_model="kimi-for-coding",
            current_base_url="https://api.kimi.com/coding/v1/",
        )

        assert result.success
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://api.kimi.com/coding"
