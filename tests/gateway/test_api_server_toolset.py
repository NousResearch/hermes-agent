"""Tests for hermes-api-server toolset and API server tool availability."""

from toolsets import resolve_toolset




class TestHermesApiServerToolset:
    """Tests for the hermes-api-server toolset definition."""


    def test_toolset_excludes_web_tools(self):
        tools = resolve_toolset("hermes-api-server")
        assert "web_search" not in tools
        assert "web_extract" not in tools

    def test_toolset_only_includes_reference_and_clarify_tools(self):
        tools = resolve_toolset("hermes-api-server")
        assert set(tools) == {"read_file", "search_files", "clarify"}

    def test_toolset_excludes_browser_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["browser_navigate", "browser_snapshot", "browser_click",
                      "browser_type", "browser_scroll", "browser_back",
                      "browser_press"]:
            assert tool not in tools

    def test_toolset_excludes_homeassistant_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"]:
            assert tool not in tools

    def test_toolset_includes_clarify(self):
        tools = resolve_toolset("hermes-api-server")
        assert "clarify" in tools

    def test_toolset_excludes_send_message(self):
        tools = resolve_toolset("hermes-api-server")
        assert "send_message" not in tools

    def test_toolset_excludes_text_to_speech(self):
        tools = resolve_toolset("hermes-api-server")
        assert "text_to_speech" not in tools


class TestApiServerPlatformConfig:

    def test_default_api_server_only_enables_reference_and_clarify(self):
        from tools.registry import discover_builtin_tools
        from hermes_cli.tools_config import _get_platform_tools
        discover_builtin_tools()
        assert _get_platform_tools({}, "api_server") == {"reference", "clarify"}

    def test_registering_tool_does_not_expand_minimal_api_defaults(self):
        """Registry additions must not silently expand the minimal API surface."""
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
            assert "delegation" not in _get_platform_tools({}, "api_server")
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



