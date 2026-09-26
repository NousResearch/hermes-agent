"""Tests for hermes tools disable/enable/list command (backend)."""
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.platform_registry import platform_registry
from hermes_cli.tools_config import _get_platform_tools, tools_disable_enable_command
from model_tools import get_tool_definitions


# ── Built-in toolset disable ────────────────────────────────────────────────


class TestToolsDisableBuiltin:

    def test_disable_removes_toolset_from_platform(self):
        config = {"platform_toolsets": {"cli": ["web", "memory", "terminal"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="disable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]

# ── Search-only configuration contracts ─────────────────────────────────────


@pytest.mark.parametrize("initial", [[], ["web"]])
def test_search_config_round_trip_preserves_other_toolsets(initial, tmp_path, monkeypatch):
    from hermes_cli.config import load_config, save_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_config({"platform_toolsets": {"cli": initial}})
    tools_disable_enable_command(
        Namespace(tools_action="enable", names=["search"], platform="cli")
    )
    saved = load_config()
    assert set(saved["platform_toolsets"]["cli"]) == set(initial) | {"search"}
    enabled = _get_platform_tools(saved, "cli", include_default_mcp_servers=False)
    with patch("tools.registry._check_fn_cached", return_value=True):
        definitions = get_tool_definitions(
            enabled_toolsets=sorted(enabled), quiet_mode=False,
            skip_tool_search_assembly=True,
        )
    names = [definition["function"]["name"] for definition in definitions]
    assert names.count("web_search") == 1
    assert ("web_extract" in names) == ("web" in initial)

    tools_disable_enable_command(
        Namespace(tools_action="disable", names=["search"], platform="cli")
    )
    assert set(load_config()["platform_toolsets"]["cli"]) == set(initial)


# ── MCP tool disable ────────────────────────────────────────────────────────


class TestToolsDisableMcp:


    def test_disable_unknown_server_prints_error(self, capsys):
        config = {"mcp_servers": {}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config"):
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["unknown:tool"], platform="cli")
            )
        out = capsys.readouterr().out
        assert "MCP server 'unknown' not found in config" in out


# ── MCP tool enable ──────────────────────────────────────────────────────────


# ── Mixed targets ────────────────────────────────────────────────────────────


# ── List output ──────────────────────────────────────────────────────────────


class TestToolsList:


    def test_list_shows_mcp_excluded_tools(self, capsys):
        config = {
            "mcp_servers": {"github": {"tools": {"exclude": ["create_issue"]}}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config):
            tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
        out = capsys.readouterr().out
        assert "github" in out
        assert "create_issue" in out


# ── Validation ───────────────────────────────────────────────────────────────


class TestToolsValidation:


    def test_mixed_valid_and_invalid_applies_valid_only(self):
        config = {"platform_toolsets": {"cli": ["web", "memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["web", "bad_toolset"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]


@pytest.mark.parametrize("action", ["list", "enable", "disable"])
def test_tools_action_accepts_deferred_plugin_without_materializing(action, capsys):
    platform = "deferred-tools-test"
    loader = MagicMock()
    configured_tools = ["memory", "web"] if action == "disable" else ["memory"]
    config = {"platform_toolsets": {platform: configured_tools}}
    args = Namespace(tools_action=action, platform=platform)
    if action != "list":
        args.names = ["web"]

    def discover_deferred_platform():
        platform_registry.register_deferred(platform, loader)

    try:
        with patch(
            "hermes_cli.plugins.discover_plugins",
            side_effect=discover_deferred_platform,
        ) as discover, \
             patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as save:
            tools_disable_enable_command(args)

        out = capsys.readouterr().out
        assert "Unknown platform" not in out
        discover.assert_called()
        loader.assert_not_called()
        if action == "list":
            assert f"Built-in toolsets ({platform}):" in out
            save.assert_not_called()
        else:
            save.assert_called()
            saved_tools = save.call_args.args[0]["platform_toolsets"][platform]
            assert ("web" in saved_tools) is (action == "enable")
    finally:
        platform_registry.unregister(platform)
