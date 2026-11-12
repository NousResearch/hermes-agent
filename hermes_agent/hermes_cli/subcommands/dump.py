"""``hermes dump`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_dump_parser(subparsers, *, cmd_dump: Callable) -> None:
    """Attach the ``dump`` subcommand to ``subparsers``."""
    dump_parser = subparsers.add_parser(
        "dump", help="Dump setup summary for support/debugging",
        description="Output a compact, plain-text summary of your Hermes setup "
        "that can be copy-pasted into Discord/GitHub for support context")
    dump_parser.add_argument(
        "--show-keys", action="store_true",
        help="Show redacted API key prefixes (first/last 4 chars) instead of just set/not set")
    dump_parser.set_defaults(func=cmd_dump)
