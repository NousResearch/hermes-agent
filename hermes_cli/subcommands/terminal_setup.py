"""Parser builder for ``hermes terminal-setup``."""

from __future__ import annotations

from typing import Callable


def build_terminal_setup_parser(subparsers, *, cmd_terminal_setup: Callable) -> None:
    """Attach the safe ``terminal-setup`` subcommand."""
    parser = subparsers.add_parser(
        "terminal-setup",
        help="Show classic CLI multiline-input terminal guidance",
        description="Show safe, non-destructive terminal guidance for the classic Hermes CLI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the managed file change without writing it",
    )
    parser.add_argument(
        "--print",
        dest="print_config",
        action="store_true",
        help="Print a pasteable mapping without writing it",
    )
    parser.set_defaults(func=cmd_terminal_setup)
