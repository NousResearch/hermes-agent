"""``hermes skin`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_skin_parser(subparsers, *, cmd_skin: Callable) -> None:
    """Attach the ``skin`` subcommand to ``subparsers``."""
    skin_parser = subparsers.add_parser(
        "skin",
        help="List, switch, and tweak skins",
        description="Manage Hermes skins. `set` tweaks one color of the active skin in place.",
    )
    skin_subparsers = skin_parser.add_subparsers(dest="skin_command")

    skin_subparsers.add_parser("list", help="List available skins")

    skin_use = skin_subparsers.add_parser("use", help="Switch the active skin")
    skin_use.add_argument("name", help="Skin name")

    # skin set — change ONE color of the active skin in place (bg untouched).
    skin_set = skin_subparsers.add_parser(
        "set", help="Set one color of the active skin (e.g. `skin set ui_tool '#00FFFF'`)"
    )
    skin_set.add_argument("key", help="Color key (e.g. ui_tool, ui_accent, background)")
    skin_set.add_argument("value", help="Hex color (#rrggbb)")
    skin_set.add_argument("--skin", help="Target a specific skin instead of the active one")

    skin_parser.set_defaults(func=cmd_skin)
