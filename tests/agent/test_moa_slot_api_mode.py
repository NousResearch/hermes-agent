"""Tests for MoA slot_runtime api_mode propagation (issue #54379).

Verify that _slot_runtime passes the resolved api_mode through to call_llm,
so reference slots using providers that require a specific API surface
(e.g. Copilot GPT-5.x → codex_responses) get routed correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSlotRuntimeApiMode:
    """_slot_runtime should include api_mode when resolve_runtime_provider returns it."""

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_includes_api_mode(self, mock_resolve):
        """api_mode from resolve_runtime_provider is forwarded in output dict."""
        mock_resolve.return_value = {
            "provider": "copilot",
            "model": "gpt-5.5",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "test-key",
            "api_mode": "codex_responses",
        }
        from agent.moa_loop import _slot_runtime

        result = _slot_runtime({"provider": "copilot", "model": "gpt-5.5"})
        assert result["api_mode"] == "codex_responses"
        assert result["base_url"] == "https://api.githubcopilot.com"
        assert result["api_key"] == "test-key"

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_omits_api_mode_when_absent(self, mock_resolve):
        """When resolve_runtime_provider does not return api_mode, output omits it."""
        mock_resolve.return_value = {
            "provider": "openai",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
        }
        from agent.moa_loop import _slot_runtime

        result = _slot_runtime({"provider": "openai", "model": "gpt-4o"})
        assert "api_mode" not in result

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_omits_api_mode_when_empty(self, mock_resolve):
        """Empty string api_mode is treated as absent."""
        mock_resolve.return_value = {
            "provider": "copilot",
            "model": "gpt-5.5",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "test-key",
            "api_mode": "",
        }
        from agent.moa_loop import _slot_runtime

        result = _slot_runtime({"provider": "copilot", "model": "gpt-5.5"})
        assert "api_mode" not in result


class TestCallLlmApiMode:
    """call_llm should accept and forward api_mode parameter."""

    def test_call_llm_accepts_api_mode_kwarg(self):
        """call_llm signature includes api_mode parameter."""
        import inspect
        from agent.auxiliary_client import call_llm

        sig = inspect.signature(call_llm)
        assert "api_mode" in sig.parameters
        assert sig.parameters["api_mode"].default is None
