"""Tests for llm.py — shared LLM helper for tiered memory composition."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestLlmComposeSuccess:
    def test_llm_compose_success(self):
        """Mock adapter: token + client work. Verify messages.create call shape."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Composed memory output")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_build = MagicMock(return_value=mock_client)
        mock_resolve = MagicMock(return_value="test-token-abc")

        with patch.dict("sys.modules", {
            "agent": MagicMock(),
            "agent.anthropic_adapter": MagicMock(
                build_anthropic_client=mock_build,
                resolve_anthropic_token=mock_resolve,
            ),
        }):
            # Re-import to pick up the mocked modules
            from plugins.memory.tiered.llm import llm_compose

            result = llm_compose("Compose memory", "Context data here")

        assert result == "Composed memory output"
        mock_resolve.assert_called_once()
        mock_build.assert_called_once_with("test-token-abc")
        mock_client.messages.create.assert_called_once()

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs[1]["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs[1]["max_tokens"] == 4096
        assert call_kwargs[1]["temperature"] == 0.3
        assert len(call_kwargs[1]["messages"]) == 1
        assert call_kwargs[1]["messages"][0]["role"] == "user"
        assert "Compose memory" in call_kwargs[1]["messages"][0]["content"]
        assert "Context data here" in call_kwargs[1]["messages"][0]["content"]


class TestLlmComposeNoToken:
    def test_llm_compose_no_token(self):
        """resolve_anthropic_token returns None. Verify returns None, no client created."""
        mock_build = MagicMock()
        mock_resolve = MagicMock(return_value=None)

        with patch.dict("sys.modules", {
            "agent": MagicMock(),
            "agent.anthropic_adapter": MagicMock(
                build_anthropic_client=mock_build,
                resolve_anthropic_token=mock_resolve,
            ),
        }):
            from plugins.memory.tiered.llm import llm_compose

            result = llm_compose("Compose memory", "Context data")

        assert result is None
        mock_resolve.assert_called_once()
        mock_build.assert_not_called()


class TestLlmComposeApiError:
    def test_llm_compose_api_error(self):
        """client.messages.create raises Exception. Verify returns None."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")

        mock_build = MagicMock(return_value=mock_client)
        mock_resolve = MagicMock(return_value="test-token-abc")

        with patch.dict("sys.modules", {
            "agent": MagicMock(),
            "agent.anthropic_adapter": MagicMock(
                build_anthropic_client=mock_build,
                resolve_anthropic_token=mock_resolve,
            ),
        }):
            from plugins.memory.tiered.llm import llm_compose

            result = llm_compose("Compose memory", "Context data")

        assert result is None


class TestLlmComposeAdapterImportError:
    def test_llm_compose_adapter_import_error(self):
        """Import of anthropic_adapter fails. Verify returns None."""
        # Setting modules to None causes ImportError on from ... import
        with patch.dict("sys.modules", {
            "agent": None,
            "agent.anthropic_adapter": None,
        }):
            from plugins.memory.tiered.llm import llm_compose

            result = llm_compose("Compose memory", "Context data")

        assert result is None
