"""Gateway help surfaces plugin-registered slash commands."""

from unittest.mock import patch

from hermes_cli.commands import gateway_help_lines, should_bypass_active_session
from hermes_cli.slash_exec import CommandContext, execute_command


_PLUGIN_COMMANDS = [
    ("kb", "Search the Thor Forge knowledge base", "<query>"),
    ("mission", "Thor Mission Control", "[status|agents|route|ask|incidents|tasks|create]"),
]


def test_gateway_help_lines_include_plugin_commands_with_usage():
    with patch(
        "hermes_cli.commands._iter_plugin_command_entries",
        return_value=_PLUGIN_COMMANDS,
    ):
        lines = gateway_help_lines()

    assert "`/kb <query>` -- Search the Thor Forge knowledge base" in lines
    assert (
        "`/mission [status|agents|route|ask|incidents|tasks|create]` -- "
        "Thor Mission Control"
    ) in lines


def test_gateway_help_executor_includes_plugin_commands():
    with patch(
        "hermes_cli.commands._iter_plugin_command_entries",
        return_value=_PLUGIN_COMMANDS,
    ):
        reply = execute_command("help", CommandContext(surface="gateway"))

    assert "`/kb <query>` -- Search the Thor Forge knowledge base" in reply.text
    assert "`/mission [status|agents|route|ask|incidents|tasks|create]`" in reply.text


def test_gateway_help_deduplicates_plugin_name_that_matches_builtin():
    with patch(
        "hermes_cli.commands._iter_plugin_command_entries",
        return_value=[("status", "Shadow status", ""), *_PLUGIN_COMMANDS],
    ):
        lines = gateway_help_lines()

    assert sum(line.startswith("`/status") for line in lines) == 1
    assert not any("Shadow status" in line for line in lines)


def test_plugin_commands_bypass_active_session_guard_for_direct_dispatch():
    with patch(
        "hermes_cli.commands._iter_plugin_command_entries",
        return_value=_PLUGIN_COMMANDS,
    ):
        assert should_bypass_active_session("kb") is True
        assert should_bypass_active_session("mission") is True
