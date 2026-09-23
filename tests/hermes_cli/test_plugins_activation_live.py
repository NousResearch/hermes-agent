"""Live activation of plugin-provided MCP servers."""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.plugins_activation_live import connect_plugin_mcp


def test_resource_only_mcp_server_is_reported_connected() -> None:
    """Generated resource utilities prove the server connected even when it has no domain tools."""
    activation = {"deferred": {"mcp_servers": ["docs"]}}
    manager = SimpleNamespace(
        get_portable_mcp_servers=lambda: {
            "docs": {"command": "docs-server"},
        },
    )

    with (
        patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
        patch("tools.mcp_tool_config._load_mcp_config", return_value={}),
        patch(
            "tools.mcp_tool_config._filter_suspicious_mcp_servers",
            side_effect=lambda servers: servers,
        ),
        patch("tools.mcp_tool_discovery.register_mcp_servers"),
        patch(
            "tools.connectors.mcp._registered_tool_names",
            return_value=[
                "mcp__docs__list_resources",
                "mcp__docs__read_resource",
            ],
        ),
    ):
        rows = connect_plugin_mcp(activation)

    assert rows == [{"name": "docs", "connected": True, "tools": []}]
