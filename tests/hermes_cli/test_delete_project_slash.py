"""Tests for /delete-project slash command resolution and CLI dispatch."""

from __future__ import annotations

from hermes_cli.commands import COMMAND_REGISTRY, resolve_command


def test_delete_project_command_registration():
    cmd = resolve_command("delete-project")
    assert cmd is not None
    assert cmd.name == "delete-project"
    assert cmd.category == "Project"
    assert "deleteproject" in cmd.aliases
    assert "rmproject" in cmd.aliases


def test_delete_project_alias_resolution():
    cmd1 = resolve_command("deleteproject")
    assert cmd1 is not None
    assert cmd1.name == "delete-project"

    cmd2 = resolve_command("rmproject")
    assert cmd2 is not None
    assert cmd2.name == "delete-project"


def test_tui_gateway_pending_commands():
    from tui_gateway.server import _PENDING_INPUT_COMMANDS
    assert "delete-project" in _PENDING_INPUT_COMMANDS
    assert "deleteproject" in _PENDING_INPUT_COMMANDS
    assert "rmproject" in _PENDING_INPUT_COMMANDS

