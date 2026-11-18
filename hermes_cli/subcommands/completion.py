"""``hermes completion`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_completion_parser(subparsers, *, cmd_completion: Callable, parser) -> None:
    """Attach the ``completion`` subcommand to ``subparsers``."""
    completion_parser = subparsers.add_parser(
        "completion", help="Print shell completion script (bash, zsh, or fish)")
    completion_parser.add_argument(
        "shell", nargs="?", default="bash", choices=["bash", "zsh", "fish"],
        help="Shell type (default: bash)")
    completion_parser.set_defaults(func=lambda args: cmd_completion(args, parser))
