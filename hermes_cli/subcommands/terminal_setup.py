"""Parser for `hermes terminal-setup`."""

from __future__ import annotations

from typing import Callable


def build_terminal_setup_parser(subparsers, *, cmd_terminal_setup: Callable) -> None:
    """Attach the ``terminal-setup`` subcommand to ``subparsers``."""
    ts_parser = subparsers.add_parser(
        "terminal-setup",
        help="Configure your terminal for native Shift+Enter newline support",
        description="Configure terminal newline support for Hermes.",
    )
    ts_parser.set_defaults(func=cmd_terminal_setup)
