"""#134593: Blank Slate wrote the subset toolset `search` (only web_search) into agent.disabled_toolsets;
enabling `web` via `hermes tools` left it there, silently stripping web_search."""
from unittest.mock import patch

from hermes_cli.setup_quick import _blank_slate_minimal_toolsets
from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools
from toolsets import resolve_toolset


def _cli_tool_names(config):
    from model_tools import get_tool_definitions
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    return enabled, {t["function"]["name"] for t in get_tool_definitions(enabled_toolsets=sorted(enabled), quiet_mode=True)}


def test_search_is_subset_of_web():
    assert set(resolve_toolset("search")) <= set(resolve_toolset("web"))


def test_blank_slate_list_excludes_search_but_keeps_kanban():
    config = {}
    _blank_slate_minimal_toolsets(config)
    disabled = config["agent"]["disabled_toolsets"]
    assert "search" not in disabled
    assert "kanban" in disabled and "web" in disabled
    for kept in ("file", "terminal", "vision", "skills"):
        assert kept not in disabled


def test_enabling_web_clears_legacy_search_entry():
    # Configs written by Blank Slate before this fix still carry `search`.
    config = {"platform_toolsets": {"cli": ["file", "terminal"]},
              "agent": {"disabled_toolsets": ["web", "search", "browser", "memory"]}}
    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"file", "terminal", "web"})
    disabled = config["agent"]["disabled_toolsets"]
    assert "web" not in disabled and "search" not in disabled
    assert "browser" in disabled and "memory" in disabled


def test_blank_slate_then_enable_web_exposes_web_search():
    config = {}
    _blank_slate_minimal_toolsets(config)
    config["agent"]["disabled_toolsets"].append("search")  # legacy Blank Slate output
    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", set(config["platform_toolsets"]["cli"]) | {"web"})
    assert "search" not in config["agent"]["disabled_toolsets"]
    enabled, names = _cli_tool_names(config)
    assert "web" in enabled
    assert {"web_search", "web_extract"} <= names
