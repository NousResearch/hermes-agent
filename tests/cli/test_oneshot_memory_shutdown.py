"""Regression test for #37632: oneshot mode must call shutdown_memory_provider.

Without this, Honcho memory daemon threads (honcho-sync, honcho-prewarm-dialectic,
honcho-context-prefetch) remain blocked in httpx I/O when the process exits.
CPython's Py_FinalizeEx force-terminates them via pthread_exit(), and glibc
converts that into abort() → SIGABRT → exit 134.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestOneshotMemoryShutdown:
    """shutdown_memory_provider() must be called after agent.chat() in oneshot mode."""

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    @patch("hermes_cli.config.load_config")
    @patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None)
    @patch("hermes_cli.oneshot.get_fallback_chain", return_value=None)
    @patch("run_agent.AIAgent")
    def test_run_agent_calls_shutdown_after_chat(
        self, MockAIAgent, mock_fb, mock_sdb, mock_load_config, mock_resolve
    ):
        """_run_agent must call agent.shutdown_memory_provider() after chat()."""
        mock_load_config.return_value = {"model": {"default": "test-model", "provider": "test"}}
        mock_resolve.return_value = {
            "api_key": "test-key",
            "base_url": "http://test",
            "provider": "test",
            "api_mode": "chat_completions",
            "credential_pool": None,
        }

        mock_agent = MagicMock()
        mock_agent.chat.return_value = "test response"
        MockAIAgent.return_value = mock_agent

        from hermes_cli.oneshot import _run_agent

        result = _run_agent("test prompt")

        assert result == "test response"
        mock_agent.chat.assert_called_once_with("test prompt")
        mock_agent.shutdown_memory_provider.assert_called_once()

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    @patch("hermes_cli.config.load_config")
    @patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None)
    @patch("hermes_cli.oneshot.get_fallback_chain", return_value=None)
    @patch("run_agent.AIAgent")
    def test_run_agent_shutdown_does_not_crash_on_error(
        self, MockAIAgent, mock_fb, mock_sdb, mock_load_config, mock_resolve
    ):
        """shutdown_memory_provider raising must not crash _run_agent."""
        mock_load_config.return_value = {"model": {"default": "test-model", "provider": "test"}}
        mock_resolve.return_value = {
            "api_key": "test-key",
            "base_url": "http://test",
            "provider": "test",
            "api_mode": "chat_completions",
            "credential_pool": None,
        }

        mock_agent = MagicMock()
        mock_agent.chat.return_value = "test response"
        mock_agent.shutdown_memory_provider.side_effect = RuntimeError("boom")
        MockAIAgent.return_value = mock_agent

        from hermes_cli.oneshot import _run_agent

        # Must not raise despite shutdown_memory_provider failing
        result = _run_agent("test prompt")
        assert result == "test response"
        mock_agent.shutdown_memory_provider.assert_called_once()
