"""Tests for CLI subcommand aliases (config list, mcp status, memory list/show)."""

import argparse
from unittest.mock import patch

from hermes_cli.subcommands.config import build_config_parser
from hermes_cli.subcommands.mcp import build_mcp_parser
from hermes_cli.subcommands.memory import build_memory_parser


def test_config_show_aliases():
    """Verify that hermes config list and hermes config ls parse to the show action."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_config_parser(subparsers, cmd_config=lambda args: None)

    for alias in ["show", "list", "ls"]:
        args = parser.parse_args(["config", alias])
        assert args.command == "config"
        assert args.config_command == alias


def test_mcp_list_aliases():
    """Verify that hermes mcp status and hermes mcp ls parse to mcp list actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_mcp_parser(subparsers, cmd_mcp=lambda args: None)

    for alias in ["list", "ls", "status"]:
        args = parser.parse_args(["mcp", alias])
        assert args.command == "mcp"
        assert args.mcp_action == alias


def test_memory_status_aliases():
    """Verify that hermes memory list, ls, and show parse to memory status actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers, cmd_memory=lambda args: None)

    for alias in ["status", "list", "ls", "show"]:
        args = parser.parse_args(["memory", alias])
        assert args.command == "memory"
        assert args.memory_command == alias


def test_config_command_dispatcher_with_list_alias():
    """Verify that config_command dispatches list to show_config."""
    from hermes_cli import config as config_mod

    with patch.object(config_mod, "show_config") as mock_show:
        config_mod.config_command(argparse.Namespace(config_command="list"))
        mock_show.assert_called_once()

    with patch.object(config_mod, "show_config") as mock_show:
        config_mod.config_command(argparse.Namespace(config_command="ls"))
        mock_show.assert_called_once()


def test_mcp_command_dispatcher_with_status_alias():
    """Verify that mcp_command dispatches status to cmd_mcp_list."""
    from hermes_cli import mcp_config

    with patch.object(mcp_config, "cmd_mcp_list") as mock_list:
        mcp_config.mcp_command(argparse.Namespace(mcp_action="status"))
        mock_list.assert_called_once()


def test_memory_command_dispatcher_with_list_alias():
    """Verify that memory_command dispatches list to cmd_status."""
    from hermes_cli import memory_setup

    with patch.object(memory_setup, "cmd_status") as mock_status:
        memory_setup.memory_command(argparse.Namespace(memory_command="list"))
        mock_status.assert_called_once()
