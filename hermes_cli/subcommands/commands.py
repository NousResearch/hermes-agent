"""``hermes commands`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_commands_parser(subparsers, *, cmd_commands: Callable, parser) -> None:
    """Attach the ``commands`` subcommand; the handler needs the root ``parser`` to walk."""
    commands_parser = subparsers.add_parser(
        "commands",
        help="Print every command and subcommand as one tree (agent-friendly; --json for tooling)",
    )
    commands_parser.add_argument(
        "--json", action="store_true", help="Machine-readable tree instead of indented text")
    commands_parser.set_defaults(func=lambda args: cmd_commands(args, parser))
