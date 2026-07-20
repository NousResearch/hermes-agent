"""Tests for hermes_cli.oneshot MCP discovery wait — regression for #68137."""

from unittest.mock import MagicMock, patch


class TestOneshotMcpDiscoveryWait:
    """Verify that _run_agent waits for MCP discovery before building the agent."""

    @patch("hermes_cli.oneshot._create_session_db_for_oneshot")
    @patch("hermes_cli.oneshot.get_fallback_chain")
    @patch("hermes_cli.oneshot.load_config")
    @patch("hermes_cli.oneshot._normalize_toolsets")
    @patch("hermes_cli.oneshot._get_platform_tools")
    @patch("hermes_cli.oneshot.resolve_runtime_provider")
    @patch("hermes_cli.oneshot.detect_provider_for_model")
    @patch("hermes_cli.mcp_startup.wait_for_mcp_discovery")
    def test_run_agent_waits_for_mcp_discovery(
        self,
        mock_wait_mcp,
        mock_detect_provider,
        mock_resolve_runtime,
        mock_get_tools,
        mock_normalize_toolsets,
        mock_load_config,
        mock_get_fallback,
        mock_create_session_db,
    ):
        """_run_agent should call wait_for_mcp_discovery before constructing AIAgent."""
        from hermes_cli.oneshot import _run_agent

        # Setup minimal mocks
        mock_load_config.return_value = {"model": {"default": "test-model", "provider": "test"}}
        mock_normalize_toolsets.return_value = []
        mock_get_tools.return_value = []
        mock_resolve_runtime.return_value = {
            "api_key": "test",
            "base_url": "http://test",
            "provider": "test",
            "api_mode": "openai",
        }
        mock_detect_provider.return_value = None
        mock_get_fallback.return_value = []
        mock_create_session_db.return_value = MagicMock()

        # Mock AIAgent to avoid actually constructing it
        with patch("run_agent.AIAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "test"}
            mock_agent_class.return_value = mock_agent

            _run_agent("test prompt")

            # Verify wait_for_mcp_discovery was called
            mock_wait_mcp.assert_called_once()

    @patch("hermes_cli.oneshot._create_session_db_for_oneshot")
    @patch("hermes_cli.oneshot.get_fallback_chain")
    @patch("hermes_cli.oneshot.load_config")
    @patch("hermes_cli.oneshot._normalize_toolsets")
    @patch("hermes_cli.oneshot._get_platform_tools")
    @patch("hermes_cli.oneshot.resolve_runtime_provider")
    @patch("hermes_cli.oneshot.detect_provider_for_model")
    @patch("hermes_cli.mcp_startup.wait_for_mcp_discovery")
    def test_run_agent_does_not_crash_when_mcp_discovery_times_out(
        self,
        mock_wait_mcp,
        mock_detect_provider,
        mock_resolve_runtime,
        mock_get_tools,
        mock_normalize_toolsets,
        mock_load_config,
        mock_get_fallback,
        mock_create_session_db,
    ):
        """_run_agent should still proceed even if MCP discovery times out."""
        from hermes_cli.oneshot import _run_agent

        mock_load_config.return_value = {"model": {"default": "test-model", "provider": "test"}}
        mock_normalize_toolsets.return_value = []
        mock_get_tools.return_value = []
        mock_resolve_runtime.return_value = {
            "api_key": "test",
            "base_url": "http://test",
            "provider": "test",
            "api_mode": "openai",
        }
        mock_detect_provider.return_value = None
        mock_get_fallback.return_value = []
        mock_create_session_db.return_value = MagicMock()

        # wait_for_mcp_discovery should not raise on timeout — it just returns
        mock_wait_mcp.return_value = None

        with patch("run_agent.AIAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "test"}
            mock_agent_class.return_value = mock_agent

            result = _run_agent("test prompt")

            # Should still produce a result
            assert result is not None
            mock_wait_mcp.assert_called_once()