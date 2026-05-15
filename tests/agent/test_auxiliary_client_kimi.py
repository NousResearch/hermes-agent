"""Tests for Kimi Coding /v1 stripping in auxiliary_client.

Refs: https://www.kimi.com/code/docs/en/
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux


class TestMaybeWrapAnthropicKimiStripping:
    """_maybe_wrap_anthropic strips /v1 for Kimi Coding endpoints."""

    def test_strips_v1_for_kimi_coding(self, monkeypatch):
        """kimi-coding with /coding/v1 → /v1 stripped before build_anthropic_client."""
        mock_client = MagicMock()
        monkeypatch.setattr(
            aux,
            "_endpoint_speaks_anthropic_messages",
            lambda url: True,
        )

        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            result = aux._maybe_wrap_anthropic(
                mock_client, "kimi-for-coding", "test-key",
                "https://api.kimi.com/coding/v1",
            )
            mock_build.assert_called_once_with("test-key", "https://api.kimi.com/coding")
            assert isinstance(result, aux.AnthropicAuxiliaryClient)

    def test_without_v1_unchanged(self, monkeypatch):
        """kimi-coding without /v1 → base_url preserved."""
        mock_client = MagicMock()
        monkeypatch.setattr(
            aux,
            "_endpoint_speaks_anthropic_messages",
            lambda url: True,
        )

        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            result = aux._maybe_wrap_anthropic(
                mock_client, "kimi-for-coding", "test-key",
                "https://api.kimi.com/coding",
            )
            mock_build.assert_called_once_with("test-key", "https://api.kimi.com/coding")

    def test_preserves_non_kimi_url(self, monkeypatch):
        """Non-Kimi Anthropic endpoints are untouched."""
        mock_client = MagicMock()
        monkeypatch.setattr(
            aux,
            "_endpoint_speaks_anthropic_messages",
            lambda url: True,
        )

        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            result = aux._maybe_wrap_anthropic(
                mock_client, "claude-sonnet", "test-key",
                "https://api.anthropic.com",
            )
            mock_build.assert_called_once_with("test-key", "https://api.anthropic.com")
