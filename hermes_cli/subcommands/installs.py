"""``hermes installs`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag, add_yes_flag


def build_installs_parser(subparsers, *, cmd_installs: Callable) -> None:
    """Attach the ``installs`` subcommand to ``subparsers``."""
    installs_parser = subparsers.add_parser(
        "installs", help="List or remove other Hermes installs on this machine",
        description="Show every Hermes install this machine has and which one answers `hermes`. "
        "Remove another install's files without touching your data or the installs that stay.")
    actions = installs_parser.add_subparsers(dest="installs_command")
    list_parser = actions.add_parser("list", help="List installs and the `hermes` launchers on PATH (default)")
    add_json_flag(list_parser, "Print the result as JSON")
    remove_parser = actions.add_parser(
        "remove", help="Remove one other install",
        description="Remove one install's folder and the launchers and shortcuts that point into it. "
        "Your Hermes data, User PATH, User environment variables, and the gateway stay as they are.")
    remove_parser.add_argument("install_id", help="Install id from `hermes installs` (a unique prefix is enough)")
    add_yes_flag(remove_parser, "Skip the confirmation question")
    remove_parser.add_argument(
        "--dry-run", action="store_true", help="Show what this removes. Change nothing.")
    actions.add_parser("dismiss", help="Hide the launch notice until the set of other installs changes")
    installs_parser.set_defaults(func=cmd_installs)
