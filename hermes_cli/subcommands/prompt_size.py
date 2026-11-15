"""``hermes prompt-size`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_prompt_size_parser(subparsers, *, cmd_prompt_size: Callable) -> None:
    """Attach the ``prompt-size`` subcommand to ``subparsers``."""
    prompt_size_parser = subparsers.add_parser(
        "prompt-size", help="Show a byte breakdown of the system prompt + tool schemas",
        description="Report the fixed prompt budget for a fresh session: system "
            "prompt total, skills index, memory, user profile, and tool-schema "
            "JSON. Runs offline (no API call).")
    prompt_size_parser.add_argument(
        "--platform", default="cli",
        help="Platform to simulate (cli, telegram, discord, ...). Default: cli")
    add_json_flag(prompt_size_parser, "Emit the breakdown as JSON")
    prompt_size_parser.set_defaults(func=cmd_prompt_size)
