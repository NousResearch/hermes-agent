"""``hermes uninstall`` subcommand parser."""

from __future__ import annotations

import argparse
from typing import Callable

from hermes_cli.subcommands._shared import add_yes_flag


def build_uninstall_parser(subparsers, *, cmd_uninstall: Callable) -> None:
    """Attach the ``uninstall`` subcommand to ``subparsers``."""
    uninstall_parser = subparsers.add_parser(
        "uninstall", help="Uninstall Hermes Agent",
        description="Remove Hermes Agent from your system. Can keep configs/data for reinstall.")
    modes = uninstall_parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--full", action="store_true",
        help="Full uninstall - remove everything including configs and data")
    modes.add_argument(
        "--gui", action="store_true",
        help="Uninstall only the desktop Chat GUI, leaving the agent intact")
    modes.add_argument(
        "--data",
        action="store_true",
        help="Remove only the active Hermes home's user data, leaving installed code "
        "intact — the one mode that works on Nix / bundled-app / Docker installs",
    )
    uninstall_parser.add_argument(
        "--gui-summary",
        action="store_true",
        help="Print a JSON summary of installed GUI/agent artifacts and exit "
        "(used by the desktop app to gate uninstall options)")
    add_yes_flag(uninstall_parser, "Skip confirmation prompts")
    uninstall_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what uninstall would remove without changing anything")
    uninstall_parser.add_argument("--desktop-userdata", help=argparse.SUPPRESS)
    uninstall_parser.set_defaults(func=cmd_uninstall)
