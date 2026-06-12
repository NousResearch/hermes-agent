import argparse
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import memory_setup
from hermes_cli.subcommands.memory import build_memory_parser


def test_memory_workspace_parser_accepts_add_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers, cmd_memory=lambda args: None)

    args = parser.parse_args(
        [
            "memory",
            "workspace",
            "add",
            "--title",
            "Research Plan",
            "--bucket",
            "Projects",
            "--tag",
            "research",
            "durable project note with enough content to save",
        ]
    )

    assert args.command == "memory"
    assert args.memory_command == "workspace"
    assert args.workspace_command == "add"
    assert args.title == "Research Plan"
    assert args.bucket == "Projects"
    assert args.tag == ["research"]


def test_memory_procedural_parser_accepts_distill_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers, cmd_memory=lambda args: None)

    args = parser.parse_args(
        [
            "memory",
            "procedural",
            "distill",
            "--name",
            "Dataset Triage",
            "--description",
            "Triage datasets.",
            "--trigger",
            "When a dataset arrives.",
            "--step",
            "Inspect the files.",
        ]
    )

    assert args.memory_command == "procedural"
    assert args.procedural_command == "distill"
    assert args.trigger == ["When a dataset arrives."]
    assert args.step == ["Inspect the files."]


def test_memory_setup_router_delegates_dual_memory_commands():
    workspace_args = SimpleNamespace(memory_command="workspace")
    procedural_args = SimpleNamespace(memory_command="procedural")

    with patch("hermes_cli.dual_memory.cmd_workspace") as workspace:
        memory_setup.memory_command(workspace_args)
    workspace.assert_called_once_with(workspace_args)

    with patch("hermes_cli.dual_memory.cmd_procedural") as procedural:
        memory_setup.memory_command(procedural_args)
    procedural.assert_called_once_with(procedural_args)
