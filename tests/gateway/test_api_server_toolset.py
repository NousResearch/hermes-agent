"""Tests for hermes-api-server toolset and API server tool availability."""
import os
import json
from unittest.mock import patch, MagicMock

import pytest

from toolsets import resolve_toolset, get_toolset, validate_toolset


class TestHermesApiServerToolset:
    """Tests for the hermes-api-server toolset definition."""

    def test_toolset_exists(self):
        ts = get_toolset("hermes-api-server")
        assert ts is not None

    def test_toolset_validates(self):
        assert validate_toolset("hermes-api-server")

    def test_toolset_includes_web_tools(self):
        tools = resolve_toolset("hermes-api-server")
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_toolset_includes_core_tools(self):
        tools = resolve_toolset("hermes-api-server")
        expected = [
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze", "image_generate",
            "execute_code", "delegate_task",
            "todo", "memory", "session_search", "cronjob",
        ]
        for tool in expected:
            assert tool in tools, f"Missing expected tool: {tool}"

    def test_toolset_includes_browser_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["browser_navigate", "browser_snapshot", "browser_click",
                      "browser_type", "browser_scroll", "browser_back",
                      "browser_press"]:
            assert tool in tools, f"Missing browser tool: {tool}"

    def test_toolset_includes_homeassistant_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"]:
            assert tool in tools, f"Missing HA tool: {tool}"

    def test_toolset_excludes_clarify(self):
        tools = resolve_toolset("hermes-api-server")
        assert "clarify" not in tools

    def test_toolset_excludes_send_message(self):
        tools = resolve_toolset("hermes-api-server")
        assert "send_message" not in tools

    def test_toolset_excludes_text_to_speech(self):
        tools = resolve_toolset("hermes-api-server")
        assert "text_to_speech" not in tools

    def test_web_toolset_excludes_local_execution_tools(self):
        tools = resolve_toolset("hermes-web")
        assert "web_search" in tools
        assert "vision_analyze" in tools
        assert "terminal" not in tools
        assert "execute_code" not in tools
        assert "read_file" not in tools

    def test_mobile_chat_toolset_is_no_tool_default(self):
        assert resolve_toolset("hermes-mobile-chat") == []

    def test_mobile_chat_platform_does_not_inherit_default_plugins_or_mcp(self, monkeypatch):
        from hermes_cli import tools_config

        monkeypatch.setattr(tools_config, "_get_plugin_toolset_keys", lambda: {"plugin_default"})
        config = {
            "mcp_servers": {
                "global_mcp": {"enabled": True},
            }
        }

        assert tools_config._get_platform_tools(config, "mobile_chat") == set()

    def test_mobile_chat_platform_allows_explicit_plugin_and_mcp(self, monkeypatch):
        from hermes_cli import tools_config

        monkeypatch.setattr(tools_config, "_get_plugin_toolset_keys", lambda: {"plugin_default"})
        config = {
            "platform_toolsets": {
                "mobile_chat": ["plugin_default", "explicit_mcp"],
            },
            "mcp_servers": {
                "explicit_mcp": {"enabled": True},
                "global_mcp": {"enabled": True},
            },
        }

        assert tools_config._get_platform_tools(config, "mobile_chat") == {"plugin_default", "explicit_mcp"}


class TestApiServerPlatformConfig:
    def test_platforms_dict_includes_api_server(self):
        from hermes_cli.tools_config import PLATFORMS
        assert "api_server" in PLATFORMS
        assert PLATFORMS["api_server"]["default_toolset"] == "hermes-api-server"

    def test_platforms_dict_includes_api_client_surfaces(self):
        from hermes_cli.tools_config import PLATFORMS
        assert PLATFORMS["web"]["default_toolset"] == "hermes-web"
        assert PLATFORMS["mobile_chat"]["default_toolset"] == "hermes-mobile-chat"


class TestApiServerAdapterToolset:
    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reads_config_toolsets(self):
        """API server resolves toolsets from config like all other platforms."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # No platform_toolsets override — should fall back to hermes-api-server default
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert isinstance(toolsets, list)
            assert len(toolsets) > 0
            assert call_kwargs.kwargs.get("platform") == "api_server"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_respects_config_override(self):
        """User can override API server toolsets via platform_toolsets in config.yaml."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # User overrides with just web and terminal
            mock_config.return_value = {
                "platform_toolsets": {"api_server": ["web", "terminal"]}
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert sorted(toolsets) == ["terminal", "web"]

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_respects_requested_mobile_chat_platform(self):
        """API clients can request a server-known platform before AIAgent construction."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "***", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            mock_config.return_value = {"platform_toolsets": {"mobile_chat": []}}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(api_platform="mobile_chat")

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            assert call_kwargs.kwargs.get("platform") == "mobile_chat"
            assert call_kwargs.kwargs.get("enabled_toolsets") == []

    def test_normalize_api_platform_accepts_known_platforms_only(self):
        from gateway.platforms.api_server import _normalize_api_platform

        assert _normalize_api_platform(None) == "api_server"
        assert _normalize_api_platform(" mobile_chat ") == "mobile_chat"
        assert _normalize_api_platform("web") == "web"

        with pytest.raises(ValueError):
            _normalize_api_platform("cli")

        with pytest.raises(ValueError):
            _normalize_api_platform("telegram")

        with pytest.raises(ValueError):
            _normalize_api_platform("terminal")

        with pytest.raises(ValueError):
            _normalize_api_platform("mobile_chat\nX-Evil: yes")
