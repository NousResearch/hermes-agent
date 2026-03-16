"""Tests for /tools slash command handler in the interactive CLI."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _NullCtx:
    """No-op context manager (stand-in for _busy_command)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _make_cli(enabled_toolsets=None):
    """Build a minimal HermesCLI stub without running __init__."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.enabled_toolsets = set(enabled_toolsets or ["web", "memory"])
    cli_obj._command_running = False
    cli_obj._busy_command = MagicMock(return_value=_NullCtx())
    return cli_obj


# ── /tools (no subcommand) ─────────────────────────────────────────────────

class TestToolsSlashNoSubcommand:

    def test_bare_tools_shows_tool_list(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "show_tools") as mock_show:
            cli_obj._handle_tools_command("/tools")
        mock_show.assert_called_once()

    def test_unknown_subcommand_falls_back_to_show_tools(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "show_tools") as mock_show:
            cli_obj._handle_tools_command("/tools foobar")
        mock_show.assert_called_once()


# ── /tools list ────────────────────────────────────────────────────────────

class TestToolsSlashList:

    def test_list_calls_backend(self, capsys):
        cli_obj = _make_cli()
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["web"]}}), \
             patch("hermes_cli.tools_config.save_config"):
            cli_obj._handle_tools_command("/tools list")
        out = capsys.readouterr().out
        assert "web" in out

    def test_list_does_not_modify_enabled_toolsets(self):
        """List is read-only — self.enabled_toolsets must not change."""
        cli_obj = _make_cli(["web", "memory"])
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["web"]}}):
            cli_obj._handle_tools_command("/tools list")
        assert cli_obj.enabled_toolsets == {"web", "memory"}

    def test_list_does_not_call_reload_mcp(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_reload_mcp") as mock_reload, \
             patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["web"]}}):
            cli_obj._handle_tools_command("/tools list")
        mock_reload.assert_not_called()


# ── /tools disable <builtin> ───────────────────────────────────────────────

class TestToolsSlashDisableBuiltin:

    def test_disable_reloads_enabled_toolsets(self):
        cli_obj = _make_cli(["web", "memory"])
        # _get_platform_tools is fully mocked; load_config return value is irrelevant
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["web", "memory"]}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory"}), \
             patch("hermes_cli.config.load_config", return_value={}):
            cli_obj._handle_tools_command("/tools disable web")
        assert "web" not in cli_obj.enabled_toolsets
        assert "memory" in cli_obj.enabled_toolsets

    def test_disable_does_not_call_reload_mcp(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_reload_mcp") as mock_reload, \
             patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": []}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value=set()), \
             patch("hermes_cli.config.load_config", return_value={}):
            cli_obj._handle_tools_command("/tools disable web")
        mock_reload.assert_not_called()

    def test_disable_missing_name_prints_usage(self, capsys):
        cli_obj = _make_cli()
        cli_obj._handle_tools_command("/tools disable")
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_disable_missing_name_does_not_call_reload_mcp(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_reload_mcp") as mock_reload:
            cli_obj._handle_tools_command("/tools disable")
        mock_reload.assert_not_called()


# ── /tools disable <server:tool> ──────────────────────────────────────────

class TestToolsSlashDisableMcp:

    def test_disable_mcp_calls_reload_mcp(self):
        cli_obj = _make_cli()
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"mcp_servers": {"github": {"command": "npx"}}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch.object(cli_obj, "_reload_mcp") as mock_reload:
            cli_obj._handle_tools_command("/tools disable github:create_issue")
        mock_reload.assert_called_once()

    def test_disable_mcp_does_not_touch_enabled_toolsets(self):
        """MCP-only targets must not modify self.enabled_toolsets."""
        cli_obj = _make_cli(["web", "memory"])
        original = cli_obj.enabled_toolsets.copy()
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"mcp_servers": {"github": {"command": "npx"}}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch.object(cli_obj, "_reload_mcp"):
            cli_obj._handle_tools_command("/tools disable github:create_issue")
        assert cli_obj.enabled_toolsets == original


# ── /tools enable <builtin> ───────────────────────────────────────────────

class TestToolsSlashEnableBuiltin:

    def test_enable_reloads_enabled_toolsets(self):
        cli_obj = _make_cli(["memory"])
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["memory"]}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory", "web"}), \
             patch("hermes_cli.config.load_config", return_value={}):
            cli_obj._handle_tools_command("/tools enable web")
        assert "web" in cli_obj.enabled_toolsets

    def test_enable_missing_name_prints_usage(self, capsys):
        cli_obj = _make_cli()
        cli_obj._handle_tools_command("/tools enable")
        out = capsys.readouterr().out
        assert "Usage" in out


# ── Mixed built-in + MCP ──────────────────────────────────────────────────

class TestToolsSlashMixed:

    def test_mixed_disable_reloads_toolsets_and_mcp(self):
        cli_obj = _make_cli(["web", "memory"])
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["memory"]},
                                 "mcp_servers": {"github": {"command": "npx"}}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory"}), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(cli_obj, "_reload_mcp") as mock_reload:
            cli_obj._handle_tools_command("/tools disable web github:create_issue")
        mock_reload.assert_called_once()
        assert "web" not in cli_obj.enabled_toolsets
