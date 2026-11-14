"""#134593: Blank Slate wrote the subset toolset `search` (only web_search) into agent.disabled_toolsets;
enabling `web` via `hermes tools` left it there, silently stripping web_search."""
from unittest.mock import patch

from hermes_cli.setup_quick import _blank_slate_minimal_toolsets
from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools, toolsets_partially_stripped


def test_blank_slate_list_excludes_search_but_keeps_kanban():
    config = {}
    _blank_slate_minimal_toolsets(config)
    disabled = config["agent"]["disabled_toolsets"]
    assert "search" not in disabled
    assert "kanban" in disabled and "web" in disabled
    for kept in ("file", "terminal", "vision", "skills"):
        assert kept not in disabled


def test_legacy_search_entry_is_reported_then_cleared_by_enabling_web():
    from model_tools import get_tool_definitions
    # Configs written by Blank Slate before this fix still carry `search` next to an enabled `web`; legacy
    # names (`terminal_tools`) strip tools the same way and must clear the same way.
    config = {"platform_toolsets": {"cli": ["file", "terminal", "web"]},
              "agent": {"disabled_toolsets": ["search", "terminal_tools", "browser", "memory"]}}
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert toolsets_partially_stripped(config, enabled) == {
        "terminal": (["terminal"], ["terminal_tools"], "run `hermes tools enable terminal` to restore"),
        "web": (["web_search"], ["search"], "run `hermes tools enable web` to restore"),
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", set(enabled))
    assert config["agent"]["disabled_toolsets"] == ["browser", "memory"]
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert toolsets_partially_stripped(config, enabled) == {}
    names = {t["function"]["name"] for t in get_tool_definitions(
        enabled_toolsets=sorted(enabled), disabled_toolsets=config["agent"]["disabled_toolsets"], quiet_mode=True)}
    assert {"web_search", "web_extract", "terminal"} <= names
