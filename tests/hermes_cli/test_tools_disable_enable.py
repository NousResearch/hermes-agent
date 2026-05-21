"""Tests for hermes tools disable/enable/list command (backend)."""
from argparse import Namespace
from unittest.mock import patch

from hermes_cli.tools_config import resolve_mcp_excludes, tools_disable_enable_command


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

    def test_disable_multiple_toolsets(self):
        config = {"platform_toolsets": {"cli": ["web", "memory", "terminal"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="disable", names=["web", "memory"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" not in saved["platform_toolsets"]["cli"]
        assert "terminal" in saved["platform_toolsets"]["cli"]

    def test_disable_already_absent_is_idempotent(self):
        config = {"platform_toolsets": {"cli": ["memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="disable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]


# ── Built-in toolset enable ─────────────────────────────────────────────────


class TestToolsEnableBuiltin:

    def test_enable_adds_toolset_to_platform(self):
        config = {"platform_toolsets": {"cli": ["memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="enable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" in saved["platform_toolsets"]["cli"]

    def test_enable_already_present_is_idempotent(self):
        config = {"platform_toolsets": {"cli": ["web"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="enable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert saved["platform_toolsets"]["cli"].count("web") == 1


# ── MCP tool disable ────────────────────────────────────────────────────────


class TestToolsDisableMcp:

    def test_disable_adds_to_exclude_list(self):
        config = {"mcp_servers": {"github": {"command": "npx"}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["github:create_issue"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "create_issue" in saved["mcp_servers"]["github"]["tools"]["exclude"]

    def test_disable_already_excluded_is_idempotent(self):
        config = {"mcp_servers": {"github": {"tools": {"exclude": ["create_issue"]}}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["github:create_issue"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert saved["mcp_servers"]["github"]["tools"]["exclude"].count("create_issue") == 1

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


class TestToolsEnableMcp:

    def test_enable_removes_from_exclude_list(self):
        config = {"mcp_servers": {"github": {"tools": {"exclude": ["create_issue", "delete_branch"]}}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="enable", names=["github:create_issue"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "create_issue" not in saved["mcp_servers"]["github"]["tools"]["exclude"]
        assert "delete_branch" in saved["mcp_servers"]["github"]["tools"]["exclude"]


# ── Platform-scoped MCP tool disable (--platform <non-cli>) ──────────────────


class TestToolsDisableMcpPlatformScoped:
    """Bug 1: `hermes tools disable --platform cron openbb:*` must write a
    platform-scoped exclusion, NOT a global one."""

    def test_disable_with_platform_writes_scoped_not_global(self):
        config = {"mcp_servers": {"openbb": {"command": "uvx"}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["openbb:get_quote"], platform="cron")
            )
        saved = mock_save.call_args[0][0]
        # Scoped exclusion written under platform_mcp_excludes.
        assert saved["platform_mcp_excludes"]["cron"]["openbb"] == ["get_quote"]
        # Global exclude list NOT touched.
        assert "tools" not in saved["mcp_servers"]["openbb"] or \
               not saved["mcp_servers"]["openbb"].get("tools", {}).get("exclude")

    def test_disable_wildcard_with_platform_scoped(self):
        config = {"mcp_servers": {"openbb": {"command": "uvx"}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["openbb:*"], platform="cron")
            )
        saved = mock_save.call_args[0][0]
        assert saved["platform_mcp_excludes"]["cron"]["openbb"] == ["*"]

    def test_disable_default_platform_still_writes_global(self):
        """Backward compat: no --platform (default cli) writes the GLOBAL list."""
        config = {"mcp_servers": {"openbb": {"command": "uvx"}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["openbb:get_quote"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "get_quote" in saved["mcp_servers"]["openbb"]["tools"]["exclude"]
        assert "platform_mcp_excludes" not in saved

    def test_disable_scoped_unknown_server_prints_error(self, capsys):
        config = {"mcp_servers": {}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config"):
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["ghost:tool"], platform="cron")
            )
        out = capsys.readouterr().out
        assert "MCP server 'ghost' not found in config" in out


class TestToolsEnableMcpPlatformScoped:
    """Inverse path: `hermes tools enable --platform cron` must remove a
    platform-scoped exclusion."""

    def test_enable_removes_scoped_exclusion(self):
        config = {
            "mcp_servers": {"openbb": {"command": "uvx"}},
            "platform_mcp_excludes": {"cron": {"openbb": ["get_quote", "get_news"]}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="enable", names=["openbb:get_quote"], platform="cron")
            )
        saved = mock_save.call_args[0][0]
        assert saved["platform_mcp_excludes"]["cron"]["openbb"] == ["get_news"]

    def test_enable_last_scoped_exclusion_prunes_empty_containers(self):
        config = {
            "mcp_servers": {"openbb": {"command": "uvx"}},
            "platform_mcp_excludes": {"cron": {"openbb": ["get_quote"]}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="enable", names=["openbb:get_quote"], platform="cron")
            )
        saved = mock_save.call_args[0][0]
        # Empty leaves are pruned so the config fully reverts.
        assert "platform_mcp_excludes" not in saved

    def test_enable_scoped_leaves_other_platforms_untouched(self):
        config = {
            "mcp_servers": {"openbb": {"command": "uvx"}},
            "platform_mcp_excludes": {
                "cron": {"openbb": ["get_quote"]},
                "telegram": {"openbb": ["get_quote"]},
            },
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="enable", names=["openbb:get_quote"], platform="cron")
            )
        saved = mock_save.call_args[0][0]
        assert "cron" not in saved["platform_mcp_excludes"]
        assert saved["platform_mcp_excludes"]["telegram"]["openbb"] == ["get_quote"]


# ── Mixed targets ────────────────────────────────────────────────────────────


class TestToolsMixedTargets:

    def test_disable_builtin_and_mcp_together(self):
        config = {
            "platform_toolsets": {"cli": ["web", "memory"]},
            "mcp_servers": {"github": {"command": "npx"}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(
                tools_action="disable",
                names=["web", "github:create_issue"],
                platform="cli",
            ))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "create_issue" in saved["mcp_servers"]["github"]["tools"]["exclude"]

    def test_builtin_toggle_does_not_persist_implicit_mcp_defaults(self):
        config = {
            "platform_toolsets": {"cli": ["web", "memory"]},
            "mcp_servers": {"exa": {"url": "https://mcp.exa.ai/mcp"}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(
                tools_action="disable",
                names=["web"],
                platform="cli",
            ))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]
        assert "exa" not in saved["platform_toolsets"]["cli"]


# ── List output ──────────────────────────────────────────────────────────────


class TestToolsList:

    def test_list_shows_enabled_toolsets(self, capsys):
        config = {"platform_toolsets": {"cli": ["web", "memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config):
            tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
        out = capsys.readouterr().out
        assert "web" in out
        assert "memory" in out

    def test_list_shows_mcp_excluded_tools(self, capsys):
        config = {
            "mcp_servers": {"github": {"tools": {"exclude": ["create_issue"]}}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config):
            tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
        out = capsys.readouterr().out
        assert "github" in out
        assert "create_issue" in out

    def test_list_shows_platform_scoped_mcp_excludes(self, capsys):
        """Bug 3: `hermes tools list --platform cron` must reflect the
        platform-scoped exclusion, not just the global config."""
        config = {
            "mcp_servers": {"openbb": {"command": "uvx"}},
            "platform_mcp_excludes": {"cron": {"openbb": ["get_quote"]}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config):
            tools_disable_enable_command(Namespace(tools_action="list", platform="cron"))
        out = capsys.readouterr().out
        assert "openbb" in out
        assert "cron-only excluded" in out
        assert "get_quote" in out

    def test_list_cli_does_not_show_other_platform_scoped_excludes(self, capsys):
        config = {
            "mcp_servers": {"openbb": {"command": "uvx"}},
            "platform_mcp_excludes": {"cron": {"openbb": ["get_quote"]}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config):
            tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
        out = capsys.readouterr().out
        assert "cron-only excluded" not in out


# ── resolve_mcp_excludes (runtime resolver helper) ──────────────────────────


class TestResolveMcpExcludes:

    def test_global_exclude_applies_to_all_platforms(self):
        config = {"mcp_servers": {"openbb": {"tools": {"exclude": ["get_quote"]}}}}
        for platform in ("cli", "cron", "telegram"):
            resolved = resolve_mcp_excludes(config, platform)
            assert resolved == {"openbb": {"get_quote"}}

    def test_scoped_exclude_applies_only_to_named_platform(self):
        config = {
            "mcp_servers": {"openbb": {"command": "uvx"}},
            "platform_mcp_excludes": {"cron": {"openbb": ["get_quote"]}},
        }
        assert resolve_mcp_excludes(config, "cron") == {"openbb": {"get_quote"}}
        assert resolve_mcp_excludes(config, "cli") == {}

    def test_global_and_scoped_are_unioned(self):
        config = {
            "mcp_servers": {"openbb": {"tools": {"exclude": ["get_news"]}}},
            "platform_mcp_excludes": {"cron": {"openbb": ["get_quote"]}},
        }
        assert resolve_mcp_excludes(config, "cron") == {"openbb": {"get_news", "get_quote"}}
        # cli still sees only the global exclude.
        assert resolve_mcp_excludes(config, "cli") == {"openbb": {"get_news"}}

    def test_wildcard_preserved(self):
        config = {"platform_mcp_excludes": {"cron": {"openbb": ["*"]}}}
        assert resolve_mcp_excludes(config, "cron") == {"openbb": {"*"}}

    def test_empty_config_returns_empty(self):
        assert resolve_mcp_excludes({}, "cron") == {}

    def test_string_exclude_value_normalized_to_set(self):
        config = {"mcp_servers": {"openbb": {"tools": {"exclude": "get_quote"}}}}
        assert resolve_mcp_excludes(config, "cli") == {"openbb": {"get_quote"}}


# ── Validation ───────────────────────────────────────────────────────────────


class TestToolsValidation:

    def test_unknown_platform_prints_error(self, capsys):
        config = {}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config"):
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["web"], platform="invalid_platform")
            )
        out = capsys.readouterr().out
        assert "Unknown platform 'invalid_platform'" in out

    def test_unknown_toolset_prints_error(self, capsys):
        config = {"platform_toolsets": {"cli": ["web"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config"):
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["nonexistent_toolset"], platform="cli")
            )
        out = capsys.readouterr().out
        assert "Unknown toolset 'nonexistent_toolset'" in out

    def test_unknown_toolset_does_not_corrupt_config(self):
        config = {"platform_toolsets": {"cli": ["web", "memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["nonexistent_toolset"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "web" in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]

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
