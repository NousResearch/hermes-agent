"""Tests for /tools slash command handler in the interactive CLI."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli(enabled_toolsets=None):
    """Build a minimal HermesCLI stub without running __init__."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.enabled_toolsets = set(enabled_toolsets or ["web", "memory"])
    cli_obj._command_running = False
    cli_obj.console = MagicMock()
    return cli_obj


def _tool_def(name):
    return {"type": "function", "function": {"name": name, "description": f"{name} tool"}}


# ── show_tools: static config vs. live agent state (#56461) ─────────────────


class TestShowToolsPrefersLiveAgentState:
    """#56461: /tools independently recomputed from static config, fully
    decoupled from the live agent.tools payload actually sent to the model.
    After agent.tools gets mutated in-place (grammar-retry stripping, a
    toolless max-iterations fallback, etc.), /tools kept reporting the
    original static list -- so a user could be told a tool is available
    right as the model itself is told there are no tools at all.
    """

    def test_no_agent_falls_back_to_static_config(self, capsys):
        """Before any agent exists (e.g. at CLI startup) there is nothing
        live to prefer -- the static recomputation is correct and expected."""
        cli_obj = _make_cli(["web"])
        cli_obj.agent = None
        with patch("cli.get_tool_definitions", return_value=[_tool_def("web_search")]) as mock_get:
            cli_obj.show_tools()
        mock_get.assert_called_once_with(enabled_toolsets=cli_obj.enabled_toolsets, quiet_mode=True)
        assert "web_search" in capsys.readouterr().out

    def test_missing_agent_attr_falls_back_to_static_config(self, capsys):
        """Some HermesCLI stubs (e.g. bypassing __init__) never set .agent
        at all -- must not raise AttributeError."""
        cli_obj = _make_cli(["web"])
        assert not hasattr(cli_obj, "agent")
        with patch("cli.get_tool_definitions", return_value=[_tool_def("web_search")]):
            cli_obj.show_tools()  # must not raise
        assert "web_search" in capsys.readouterr().out

    def test_agent_with_no_tools_falls_back_to_static_config(self, capsys):
        cli_obj = _make_cli(["web"])
        cli_obj.agent = MagicMock(tools=None)
        with patch("cli.get_tool_definitions", return_value=[_tool_def("web_search")]):
            cli_obj.show_tools()
        assert "web_search" in capsys.readouterr().out

    def test_live_agent_tools_used_when_present(self, capsys):
        """The core fix: once an agent exists, its actual (possibly
        mutated) tools list wins over the static recomputation."""
        cli_obj = _make_cli(["web", "memory"])
        cli_obj.agent = MagicMock(tools=[_tool_def("memory")])
        with patch("cli.get_tool_definitions", return_value=[_tool_def("web_search"), _tool_def("memory")]) as mock_get:
            cli_obj.show_tools()
        mock_get.assert_not_called()
        out = capsys.readouterr().out
        assert "memory" in out
        assert "web_search" not in out

    def test_live_agent_tools_stripped_to_empty_shows_no_tools(self, capsys):
        """If agent.tools was mutated down to genuinely empty (e.g. the
        max-iterations toolless fallback), /tools must reflect that truth --
        not silently fall back to showing the original static list, which
        would recreate exactly the #56461 contradiction (a tool reported
        as available while the model itself is being told there are none)."""
        cli_obj = _make_cli(["web", "memory"])
        cli_obj.agent = MagicMock(tools=[])
        with patch("cli.get_tool_definitions", return_value=[_tool_def("web_search"), _tool_def("memory")]) as mock_get:
            cli_obj.show_tools()
        mock_get.assert_not_called()
        out = capsys.readouterr().out
        assert "No tools available" in out
        assert "web_search" not in out


# ── /tools (no subcommand) ──────────────────────────────────────────────────


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


# ── /tools list ─────────────────────────────────────────────────────────────


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


# ── /tools disable (session reset) ──────────────────────────────────────────


class TestToolsSlashDisableWithReset:

    def test_disable_applies_directly_and_resets_session(self):
        """Disable applies immediately (no confirmation prompt) and resets session."""
        cli_obj = _make_cli(["web", "memory"])
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["web", "memory"]}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory"}), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(cli_obj, "new_session") as mock_reset:
            cli_obj._handle_tools_command("/tools disable web")
        mock_reset.assert_called_once()
        assert "web" not in cli_obj.enabled_toolsets

    def test_disable_does_not_prompt_for_confirmation(self):
        """Disable no longer uses input() — it applies directly."""
        cli_obj = _make_cli(["web", "memory"])
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["web", "memory"]}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory"}), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(cli_obj, "new_session"), \
             patch("builtins.input") as mock_input:
            cli_obj._handle_tools_command("/tools disable web")
        mock_input.assert_not_called()

    def test_disable_always_resets_session(self):
        """Even without a confirmation prompt, disable always resets the session."""
        cli_obj = _make_cli(["web", "memory"])
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["web", "memory"]}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory"}), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(cli_obj, "new_session") as mock_reset:
            cli_obj._handle_tools_command("/tools disable web")
        mock_reset.assert_called_once()

    def test_disable_missing_name_prints_usage(self, capsys):
        cli_obj = _make_cli()
        cli_obj._handle_tools_command("/tools disable")
        out = capsys.readouterr().out
        assert "Usage" in out


# ── /tools enable (session reset) ───────────────────────────────────────────


class TestToolsSlashEnableWithReset:

    def test_enable_applies_directly_and_resets_session(self):
        """Enable applies immediately (no confirmation prompt) and resets session."""
        cli_obj = _make_cli(["memory"])
        with patch("hermes_cli.tools_config.load_config",
                   return_value={"platform_toolsets": {"cli": ["memory"]}}), \
             patch("hermes_cli.tools_config.save_config"), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory", "web"}), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(cli_obj, "new_session") as mock_reset:
            cli_obj._handle_tools_command("/tools enable web")
        mock_reset.assert_called_once()
        assert "web" in cli_obj.enabled_toolsets

    def test_enable_missing_name_prints_usage(self, capsys):
        cli_obj = _make_cli()
        cli_obj._handle_tools_command("/tools enable")
        out = capsys.readouterr().out
        assert "Usage" in out
