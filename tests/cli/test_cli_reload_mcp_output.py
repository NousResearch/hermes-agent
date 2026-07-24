"""Regression tests for classic CLI /reload-mcp status output."""

from types import SimpleNamespace
from unittest.mock import patch


def _make_cli_without_agent(enabled_toolsets=None):
    import cli as cli_mod

    cli_obj = object.__new__(cli_mod.HermesCLI)
    cli_obj._command_running = False
    cli_obj.agent = None
    cli_obj.enabled_toolsets = enabled_toolsets or ["coding"]
    cli_obj.conversation_history = []
    return cli_obj


def test_reload_mcp_before_first_message_queues_mcp_alias_for_first_agent(capsys):
    """Before the first prompt, /reload-mcp prepares the next agent initialization."""
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
    assert "queued for next agent initialization" in output
    assert "Agent updated — 0 tool(s) available" not in output
    assert cli_obj.enabled_toolsets == ["coding", "langchaindocs"]


def test_reload_mcp_before_first_message_keeps_all_toolsets_unmodified():
    """`all` already includes connected MCP servers, so do not rewrite it."""
    cli_obj = _make_cli_without_agent(enabled_toolsets=["all"])

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=["mcp_langchaindocs_search"]),
        patch.dict(
            "tools.mcp_tool._servers",
            {"langchaindocs": SimpleNamespace(_registered_tool_names=["mcp_langchaindocs_search"])},
            clear=True,
        ),
    ):
        cli_obj._reload_mcp()

    assert cli_obj.enabled_toolsets == ["all"]


def test_first_init_agent_receives_mcp_alias_after_pre_agent_reload(monkeypatch):
    """The first AIAgent build must receive MCP aliases queued by /reload-mcp."""
    import cli as cli_mod
    from hermes_cli import mcp_startup

    cli_obj = cli_mod.HermesCLI(compact=True)
    cli_obj.agent = None
    cli_obj.enabled_toolsets = ["coding"]
    cli_obj.conversation_history = []
    cli_obj._session_db = object()
    cli_obj._resumed = False
    cli_obj._install_tool_callbacks = lambda: None
    cli_obj._ensure_tirith_security = lambda: None
    cli_obj._ensure_runtime_credentials = lambda: True

    tools = [
        "mcp_langchaindocs_search",
        "mcp_langchaindocs_fetch",
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

    captured = {}

    def _fake_agent(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", lambda: None)
    monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)

    assert cli_obj._init_agent() is True
    assert captured["enabled_toolsets"] == ["coding", "langchaindocs"]
