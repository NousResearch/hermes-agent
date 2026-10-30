"""Live activation of plugin-provided MCP servers."""

from unittest.mock import patch

import pytest

from hermes_cli.plugins_activation_live import connect_plugin_mcp


@pytest.mark.parametrize("registered", [
    ["mcp__docs__list_resources", "mcp__docs__read_resource"],
    ["mcp__docs__list_prompts", "mcp__docs__get_prompt"],
])
def test_utility_only_mcp_server_is_reported_connected(registered) -> None:
    """Generated utilities prove the server connected even when it has no domain tools."""
    activation = {"deferred": {"mcp_servers": ["docs"]}}
    portable = {"docs": {"command": "docs-server"}}

    with (
        patch("tools.mcp_tool_config._load_mcp_config", return_value={}),
        patch(
            "tools.mcp_tool_config._filter_suspicious_mcp_servers",
            side_effect=lambda servers: servers,
        ),
        patch("tools.mcp_tool_discovery.register_mcp_servers"),
        patch(
            "tools.connectors.mcp._registered_tool_names",
            return_value=registered,
        ),
    ):
        rows = connect_plugin_mcp(activation, portable)

    assert rows == [{"name": "docs", "connected": True, "tools": []}]
