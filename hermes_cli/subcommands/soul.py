"""``hermes soul`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_soul_parser(subparsers, *, cmd_soul: Callable) -> None:
    """Attach the non-interactive ``soul`` subcommand to ``subparsers``."""
    soul_parser = subparsers.add_parser(
        "soul", help="Manage the profile's SOUL.md persona file",
        description="Set the profile's SOUL.md persona without opening an editor.",
    )
    soul_subparsers = soul_parser.add_subparsers(dest="soul_action")

    soul_set = soul_subparsers.add_parser(
        "set", help="Replace SOUL.md with text or the contents of a file")
    source = soul_set.add_mutually_exclusive_group(required=True)
    source.add_argument("text", nargs="?", help="Persona text to write")
    source.add_argument("--file", metavar="PATH", help="Read persona text from PATH")

    soul_parser.set_defaults(func=cmd_soul)
