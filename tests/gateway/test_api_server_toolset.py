"""Tests for hermes-api-server toolset and API server tool availability."""
from unittest.mock import MagicMock, patch

class TestApiServerPlatformConfig:

    def test_default_api_server_includes_terminal_toolset(self):
        """Regression #49622: desktop-only read_terminal is registered into the
        'terminal' toolset (ships in-repo), so resolve_toolset('terminal') grows
        to include it after discovery. read_terminal is NOT in the
        hermes-api-server composite, so the old all-tools subset test dropped
        'terminal' entirely. Its static membership (terminal, process) IS in the
        composite, so it must stay enabled."""
        from tools.registry import discover_builtin_tools
        from hermes_cli.tools_config import _get_platform_tools
        discover_builtin_tools()
        assert "terminal" in _get_platform_tools({}, "api_server")


class TestApiServerAdapterToolset:
    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_intersects_allowed_tools_with_server_policy(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(PlatformConfig())
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model", return_value="test/model"), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("hermes_cli.tools_config._get_platform_tools") as mock_tools, \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": []}
            mock_tools.return_value = {"web_search", "terminal"}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(allowed_tools=("web_search",))

        assert mock_agent_cls.call_args.kwargs["enabled_toolsets"] == ["web_search"]

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_can_disable_all_tools(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(PlatformConfig())
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model", return_value="test/model"), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": []}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(disable_tools=True)

        assert mock_agent_cls.call_args.kwargs["enabled_toolsets"] == []
