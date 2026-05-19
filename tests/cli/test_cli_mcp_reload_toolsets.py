from types import SimpleNamespace
from unittest.mock import patch

from cli import HermesCLI


def test_reload_mcp_preserves_disabled_toolsets():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._command_running = True
    cli_obj.conversation_history = []
    cli_obj.agent = SimpleNamespace(
        enabled_toolsets=["browser", "mcp-lean_ctx"],
        disabled_toolsets=["messaging"],
        tools=[],
        valid_tool_names=set(),
        _persist_session=lambda *args, **kwargs: None,
    )

    fake_tools = [
        {
            "type": "function",
            "function": {
                "name": "mcp_lean_ctx_ctx_call",
                "description": "Invoke lean-ctx.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with patch("tools.mcp_tool.shutdown_mcp_servers"), \
         patch("tools.mcp_tool.discover_mcp_tools", return_value=fake_tools), \
         patch("cli.get_tool_definitions", return_value=fake_tools) as mock_defs:
        cli_obj._reload_mcp()

    mock_defs.assert_called_once_with(
        enabled_toolsets=["browser", "mcp-lean_ctx"],
        disabled_toolsets=["messaging"],
        quiet_mode=True,
    )
    assert cli_obj.agent.tools == fake_tools
    assert cli_obj.agent.valid_tool_names == {"mcp_lean_ctx_ctx_call"}
