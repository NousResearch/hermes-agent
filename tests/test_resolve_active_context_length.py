"""Regression tests for _resolve_active_context_length() param forwarding.

Covers the fix for #37808 — the function was calling
``get_model_context_length(model_id)`` without passing ``base_url``,
``provider``, or ``config_context_length``, causing every cold start
to fall through to the OpenRouter live API fetch (~5-6s overhead).
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock


class TestResolveActiveContextLength:
    """Verify _resolve_active_context_length passes all available params."""

    def test_passes_config_context_length(self):
        """model.context_length in config should short-circuit the lookup."""
        from model_tools import _resolve_active_context_length

        mock_cfg = {
            "model": {
                "model": "gpt-4o",
                "context_length": 200_000,
            }
        }
        with (
            patch("hermes_cli.config.load_config", return_value=mock_cfg),
            patch("agent.model_metadata.get_model_context_length", return_value=200_000) as mock_gmcl,
        ):
            result = _resolve_active_context_length()

        assert result == 200_000
        mock_gmcl.assert_called_once_with(
            "gpt-4o",
            base_url="",
            provider="",
            config_context_length=200_000,
        )

    def test_passes_base_url_and_provider(self):
        """base_url and provider from config should be forwarded."""
        from model_tools import _resolve_active_context_length

        mock_cfg = {
            "model": {
                "model": "custom-model",
                "base_url": "https://my-endpoint.invalid/v1",
                "provider": "openai",
            }
        }
        with (
            patch("hermes_cli.config.load_config", return_value=mock_cfg),
            patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_gmcl,
        ):
            result = _resolve_active_context_length()

        assert result == 128_000
        mock_gmcl.assert_called_once_with(
            "custom-model",
            base_url="https://my-endpoint.invalid/v1",
            provider="openai",
            config_context_length=None,
        )

    def test_returns_zero_when_no_model(self):
        """Empty model config should return 0 without calling get_model_context_length."""
        from model_tools import _resolve_active_context_length

        with (
            patch("hermes_cli.config.load_config", return_value={}),
            patch("agent.model_metadata.get_model_context_length") as mock_gmcl,
        ):
            result = _resolve_active_context_length()

        assert result == 0
        mock_gmcl.assert_not_called()

    def test_returns_zero_on_exception(self):
        """Exceptions should be caught and return 0."""
        from model_tools import _resolve_active_context_length

        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            result = _resolve_active_context_length()

        assert result == 0

    def test_invalid_context_length_ignored(self):
        """Non-positive context_length should be treated as None."""
        from model_tools import _resolve_active_context_length

        mock_cfg = {
            "model": {
                "model": "gpt-4o",
                "context_length": -1,
            }
        }
        with (
            patch("hermes_cli.config.load_config", return_value=mock_cfg),
            patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_gmcl,
        ):
            _resolve_active_context_length()

        mock_gmcl.assert_called_once_with(
            "gpt-4o",
            base_url="",
            provider="",
            config_context_length=None,
        )
