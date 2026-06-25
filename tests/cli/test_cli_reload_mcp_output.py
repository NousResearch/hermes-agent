"""Regression tests for classic CLI /reload-mcp status output."""

from types import SimpleNamespace
from unittest.mock import patch


def _make_cli_without_agent():
    import cli as cli_mod

    cli_obj = object.__new__(cli_mod.HermesCLI)
    cli_obj._command_running = False
    cli_obj.agent = None
    cli_obj.enabled_toolsets = ["coding"]
    cli_obj.conversation_history = []
    return cli_obj


def test_reload_mcp_before_first_message_does_not_claim_agent_updated(capsys):
    """Before the first prompt, only the MCP registry exists; no agent was rebuilt."""
    cli_obj = _make_cli_without_agent()
    tools = [
        "mcp_langchaindocs_search",
        "mcp_langchaindocs_fetch",
        "mcp_langchaindocs_list_resources",
        "mcp_langchaindocs_read_resource",
    ]

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=tools),
        patch.dict(
            "tools.mcp_tool._servers",
            {"langchaindocs": SimpleNamespace(_registered_tool_names=tools)},
            clear=True,
        ),
    ):
        cli_obj._reload_mcp()

    output = capsys.readouterr().out
    assert "🔧 4 tool(s) available from 1 server(s)" in output
    assert "MCP registry updated — 4 MCP tool(s) available" in output
    assert "agent will load tools on the next message" in output
    assert "Agent updated — 0 tool(s) available" not in output
