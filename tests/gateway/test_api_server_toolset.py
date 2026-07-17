"""Tests for hermes-api-server toolset and API server tool availability."""
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


class TestApiServerPlatformConfig:
    def test_platforms_dict_includes_api_server(self):
        from hermes_cli.tools_config import PLATFORMS
        assert "api_server" in PLATFORMS
        assert PLATFORMS["api_server"]["default_toolset"] == "hermes-api-server"

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

    def test_registering_tool_into_toolset_does_not_drop_toolset_from_inference(self):
        """Class invariant (covers the delegate_cli overlay case): registering a
        NEW tool into an existing configurable toolset must never remove that
        toolset from a platform whose composite lists the toolset's static
        tools. Synthetic registration keeps the test hermetic in CI."""
        from tools.registry import registry
        from hermes_cli.tools_config import _get_platform_tools

        sentinel = "test_sentinel_delegation_tool"
        registry.register(
            name=sentinel,
            toolset="delegation",
            schema={"name": sentinel, "description": "test",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: "{}",
        )
        try:
            # delegation's static membership (delegate_task) is in the composite,
            # so the toolset must survive inference despite the extra registry tool.
            assert "delegation" in _get_platform_tools({}, "api_server"), (
                "registering a tool into 'delegation' dropped it from api_server"
            )
        finally:
            registry.deregister(sentinel)

    def test_default_off_and_restricted_toolsets_stay_off_on_api_server(self):
        """Negative contract: the static-membership comparison must NOT newly
        enable default-off or platform-restricted toolsets."""
        import os
        from unittest.mock import patch
        from hermes_cli.tools_config import _get_platform_tools
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HASS_TOKEN", None)
            os.environ.pop("XAI_API_KEY", None)
            enabled = _get_platform_tools({}, "api_server")
        assert "homeassistant" not in enabled
        assert "discord" not in enabled
        assert "discord_admin" not in enabled
        assert "x_search" not in enabled


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
    def test_create_agent_applies_request_runtime_controls(self):
        """Per-request reasoning and Fast settings reach the provider agent."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model", return_value="gpt-5.6-sol"), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key",
                "base_url": None,
                "provider": "openai-codex",
                "api_mode": None,
                "command": None,
                "args": [],
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(
                reasoning_effort_override="xhigh",
                service_tier_override="priority",
            )

            kwargs = mock_agent_cls.call_args.kwargs
            assert kwargs["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
            assert kwargs["service_tier"] == "priority"
            assert kwargs["request_overrides"] == {"service_tier": "priority"}

            mock_agent_cls.reset_mock()
            adapter._create_agent(service_tier_override="normal")
            assert mock_agent_cls.call_args.kwargs["service_tier"] is None
            assert mock_agent_cls.call_args.kwargs["request_overrides"] == {}

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_explicit_request_route_wins_over_session_model_override(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider") as mock_provider, \
             patch("gateway.run._resolve_gateway_model", return_value="global-model"), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch.object(adapter, "_session_model_override_for", return_value={"model": "session-model"}), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "global-key",
                "base_url": None,
                "provider": "global-provider",
                "api_mode": None,
                "command": None,
                "args": [],
            }
            mock_provider.return_value = {
                "api_key": "request-key",
                "base_url": None,
                "provider": "openai-codex",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(
                gateway_session_key="webui:session",
                request_route={"model": "gpt-5.6-sol", "provider": "openai-codex"},
            )

            kwargs = mock_agent_cls.call_args.kwargs
            assert kwargs["model"] == "gpt-5.6-sol"
            assert kwargs["provider"] == "openai-codex"
            assert kwargs["api_key"] == "request-key"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_session_model_override_wins_over_static_route_and_is_applied(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        session_override = {
            "model": "session-model",
            "provider": "session-provider",
            "api_mode": "responses",
        }
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider") as mock_provider, \
             patch("gateway.run._resolve_gateway_model", return_value="global-model"), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch.object(adapter, "_session_model_override_for", return_value=session_override), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "global-key",
                "base_url": None,
                "provider": "global-provider",
                "api_mode": None,
                "command": None,
                "args": [],
            }
            mock_provider.return_value = {
                "api_key": "session-key",
                "base_url": None,
                "provider": "session-provider",
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(
                gateway_session_key="webui:session",
                route={"model": "static-model", "provider": "static-provider"},
            )

            kwargs = mock_agent_cls.call_args.kwargs
            assert kwargs["model"] == "session-model"
            assert kwargs["provider"] == "session-provider"
            assert kwargs["api_key"] == "session-key"
            assert kwargs["api_mode"] == "responses"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_route_credentials_are_used_to_resolve_unconfigured_provider(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider") as mock_provider, \
             patch("gateway.run._resolve_gateway_model", return_value="global-model"), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "global-key",
                "base_url": None,
                "provider": "global-provider",
                "api_mode": None,
                "command": None,
                "args": [],
            }
            mock_provider.return_value = {
                "api_key": "route-key",
                "base_url": "https://route.invalid/v1",
                "provider": "anthropic",
                "api_mode": "anthropic_messages",
                "command": None,
                "args": [],
                "credential_pool": None,
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(
                request_route={
                    "model": "claude-route-model",
                    "provider": "anthropic",
                    "api_key": "route-key",
                    "base_url": "https://route.invalid/v1",
                }
            )

            mock_provider.assert_called_once_with(
                "anthropic",
                api_key="route-key",
                base_url="https://route.invalid/v1",
                target_model="claude-route-model",
            )
            kwargs = mock_agent_cls.call_args.kwargs
            assert kwargs["api_key"] == "route-key"
            assert kwargs["base_url"] == "https://route.invalid/v1"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_routed_provider_resolution_failure_does_not_reuse_default_credentials(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=RuntimeError("provider unavailable")), \
             patch("gateway.run._resolve_gateway_model", return_value="global-model"), \
             patch("gateway.run._load_gateway_config", return_value={}):
            mock_kwargs.return_value = {
                "api_key": "must-not-leak",
                "base_url": "https://default.invalid/v1",
                "provider": "default-provider",
                "api_mode": None,
                "command": None,
                "args": [],
            }

            with pytest.raises(RuntimeError, match="provider unavailable"):
                adapter._create_agent(
                    request_route={"model": "gpt-5.6-sol", "provider": "openai-codex"},
                )
