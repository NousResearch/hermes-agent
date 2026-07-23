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

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_reduced_direct_vision_route_bypasses_broken_default_runtime(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=AssertionError("broken default runtime must not be resolved"),
        ) as default_resolver, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value={
                "api_key": "route-key",
                "base_url": "https://route.invalid/v1",
                "provider": "openai-codex",
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        ) as route_resolver, patch(
            "gateway.run._resolve_gateway_model",
            return_value="broken-default-model",
        ), patch(
            "gateway.run._load_gateway_config",
            return_value={},
        ), patch(
            "agent.image_routing._lookup_supports_vision",
            return_value=True,
        ) as vision_lookup, patch(
            "run_agent.AIAgent",
        ) as agent_cls:
            agent_cls.return_value = MagicMock(tools=[], valid_tool_names=set())

            adapter._create_agent(
                reduced_authority=True,
                requires_vision=True,
                request_route={
                    "model": "gpt-5.6-sol",
                    "provider": "openai-codex",
                    "api_key": "route-key",
                    "base_url": "https://route.invalid/v1",
                },
            )

        default_resolver.assert_not_called()
        route_resolver.assert_called_once_with(
            "openai-codex",
            api_key="route-key",
            base_url="https://route.invalid/v1",
            target_model="gpt-5.6-sol",
        )
        vision_lookup.assert_called_once_with("openai-codex", "gpt-5.6-sol", {})
        assert agent_cls.call_args.kwargs["model"] == "gpt-5.6-sol"
        assert agent_cls.call_args.kwargs["api_key"] == "route-key"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_reduced_nonvision_route_rejects_before_agent_or_auxiliary_calls(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=AssertionError("broken default runtime must not be resolved"),
        ) as default_resolver, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value={
                "api_key": "route-key",
                "base_url": "https://route.invalid/v1",
                "provider": "openai-codex",
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        ), patch(
            "gateway.run._resolve_gateway_model",
            return_value="broken-default-model",
        ), patch(
            "gateway.run._load_gateway_config",
            return_value={},
        ), patch(
            "agent.image_routing._lookup_supports_vision",
            return_value=False,
        ), patch(
            "tools.vision_tools.vision_analyze_tool",
            side_effect=AssertionError("auxiliary vision must not run"),
        ) as auxiliary_vision, patch(
            "run_agent.AIAgent",
        ) as agent_cls:
            with pytest.raises(ValueError, match="vision-capable"):
                adapter._create_agent(
                    reduced_authority=True,
                    requires_vision=True,
                    request_route={
                        "model": "text-only-model",
                        "provider": "openai-codex",
                        "api_key": "route-key",
                        "base_url": "https://route.invalid/v1",
                    },
                )

        default_resolver.assert_not_called()
        agent_cls.assert_not_called()
        auxiliary_vision.assert_not_called()

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_invalid_reduced_explicit_route_never_falls_back_to_default_runtime(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=AssertionError("default runtime must not mask route failure"),
        ) as default_resolver, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            side_effect=RuntimeError("explicit route unavailable"),
        ) as route_resolver, patch(
            "gateway.run._resolve_gateway_model",
            return_value="broken-default-model",
        ), patch(
            "gateway.run._load_gateway_config",
            return_value={},
        ), patch(
            "run_agent.AIAgent",
        ) as agent_cls:
            with pytest.raises(RuntimeError, match="explicit route unavailable"):
                adapter._create_agent(
                    reduced_authority=True,
                    request_route={
                        "model": "invalid-route-model",
                        "provider": "invalid-route-provider",
                    },
                )

        default_resolver.assert_not_called()
        route_resolver.assert_called_once_with(
            "invalid-route-provider",
            api_key=None,
            base_url=None,
            target_model="invalid-route-model",
        )
        agent_cls.assert_not_called()

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_reduced_explicit_route_still_rejects_subprocess_runtime(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=AssertionError("default runtime must not be resolved"),
        ) as default_resolver, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value={
                "api_key": None,
                "base_url": None,
                "provider": "codex_app_server",
                "api_mode": "codex_app_server",
                "command": "codex",
                "args": ["app-server"],
                "credential_pool": None,
            },
        ), patch(
            "gateway.run._resolve_gateway_model",
            return_value="broken-default-model",
        ), patch(
            "gateway.run._load_gateway_config",
            return_value={},
        ), patch(
            "run_agent.AIAgent",
        ) as agent_cls:
            with pytest.raises(
                ValueError,
                match="not allowed for reduced-authority",
            ):
                adapter._create_agent(
                    reduced_authority=True,
                    request_route={
                        "model": "gpt-codex",
                        "provider": "codex_app_server",
                    },
                )

        default_resolver.assert_not_called()
        agent_cls.assert_not_called()

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reduced_authority_enforces_empty_final_tool_surface(
        self, monkeypatch
    ):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        injected_agent = MagicMock()
        injected_agent.tools = [
            {"type": "function", "function": {"name": "kanban_show"}}
        ]
        injected_agent.valid_tool_names = {"kanban_show"}
        monkeypatch.setenv("HERMES_KANBAN_TASK", "ambient-worker")

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs"
        ) as mock_kwargs, patch(
            "gateway.run._resolve_gateway_model"
        ) as mock_model, patch(
            "gateway.run._load_gateway_config"
        ) as mock_config, patch(
            "run_agent.AIAgent"
        ) as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key",
                "base_url": None,
                "provider": None,
                "api_mode": None,
                "command": None,
                "args": [],
            }
            mock_model.return_value = "test/model"
            mock_config.return_value = {}
            mock_agent_cls.return_value = injected_agent

            returned = adapter._create_agent(reduced_authority=True)

        call_kwargs = mock_agent_cls.call_args.kwargs
        assert call_kwargs["enabled_toolsets"] == []
        assert call_kwargs["skip_memory"] is True
        assert call_kwargs["skip_context_files"] is True
        assert call_kwargs["reduced_authority"] is True
        assert call_kwargs["session_db"] is None
        assert call_kwargs["fallback_model"] is None
        assert call_kwargs["max_iterations"] == 1
        assert returned.tools == []
        assert returned.valid_tool_names == set()
        assert returned._skip_mcp_refresh is True
        assert returned._skip_plugin_hooks is True
        assert returned._skip_extension_middleware is True

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reduced_authority_rejects_subprocess_runtime(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs"
        ) as mock_kwargs, patch(
            "gateway.run._resolve_gateway_model", return_value="gpt-codex"
        ), patch(
            "gateway.run._load_gateway_config", return_value={}
        ), patch(
            "run_agent.AIAgent"
        ) as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key",
                "base_url": None,
                "provider": "codex_app_server",
                "api_mode": None,
                "command": None,
                "args": [],
            }

            with pytest.raises(
                ValueError, match="not allowed for reduced-authority"
            ):
                adapter._create_agent(reduced_authority=True)

        mock_agent_cls.assert_not_called()

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reduced_authority_rejects_subprocess_api_mode(self):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs"
        ) as mock_kwargs, patch(
            "gateway.run._resolve_gateway_model", return_value="gpt-codex"
        ), patch(
            "gateway.run._load_gateway_config", return_value={}
        ), patch(
            "run_agent.AIAgent"
        ) as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key",
                "base_url": None,
                "provider": "openai",
                "api_mode": "codex_app_server",
                "command": None,
                "args": [],
            }

            with pytest.raises(
                ValueError, match="not allowed for reduced-authority"
            ):
                adapter._create_agent(reduced_authority=True)

        mock_agent_cls.assert_not_called()

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reduced_authority_disables_runtime_fallback_resolution(
        self,
    ):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs"
        ) as resolver, patch(
            "gateway.run._resolve_gateway_model", return_value="test/model"
        ), patch(
            "gateway.run._load_gateway_config", return_value={}
        ), patch(
            "run_agent.AIAgent"
        ) as agent_cls:
            resolver.return_value = {
                "api_key": "test-key",
                "base_url": "https://api.example/v1",
                "provider": "openai",
                "api_mode": None,
                "command": None,
                "args": [],
            }
            agent_cls.return_value = MagicMock(tools=[], valid_tool_names=set())

            adapter._create_agent(reduced_authority=True)

        resolver.assert_called_once_with(allow_fallback=False)

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    @pytest.mark.parametrize(
        "base_url", ["acp://copilot", "acp+tcp://127.0.0.1:9000"]
    )
    def test_create_agent_reduced_authority_rejects_acp_base_urls(
        self, base_url
    ):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs"
        ) as resolver, patch(
            "gateway.run._resolve_gateway_model", return_value="test/model"
        ), patch(
            "gateway.run._load_gateway_config", return_value={}
        ), patch(
            "run_agent.AIAgent"
        ) as agent_cls:
            resolver.return_value = {
                "api_key": "test-key",
                "base_url": base_url,
                "provider": "openai",
                "api_mode": None,
                "command": None,
                "args": [],
            }
            with pytest.raises(
                ValueError, match="not allowed for reduced-authority"
            ):
                adapter._create_agent(reduced_authority=True)

        agent_cls.assert_not_called()
