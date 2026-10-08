"""Parser for explicitly enabled, user-level auto-update schedules."""

from __future__ import annotations

import argparse


def _dispatch(args) -> None:
    from hermes_cli.update_auto import cmd_update_auto

    cmd_update_auto(args)


def build_auto_parser(update_parser) -> None:
    children = update_parser.add_subparsers(dest="update_subcommand")
    auto = children.add_parser(
        "auto", help="Opt-in scheduled updates (no model calls)",
        description="User-level launchd/systemd scheduling around the transactional updater. Disabled by default.",
    )
    commands = auto.add_subparsers(dest="auto_subcommand", required=True)
    for name, help_text in (
        ("status", "Show scheduler and last-run status without checking the network"),
        ("plan", "Check the selected update target; save and print an advisory plan"),
        ("run-now", "Run the updater now with mandatory backups and terminal receipt verification"),
        ("run-scheduled", "Internal entrypoint; requires valid saved scheduler activation"),
        ("enable", "Enable a user schedule; no administrator privileges"),
        ("disable", "Disable and remove this installation's schedules"),
        ("migrate", "Consolidate matching legacy profile schedules into one installation schedule"),
    ):
        child = commands.add_parser(name, help=help_text)
        child.set_defaults(func=_dispatch)
        if name == "run-scheduled":
            child.add_argument("--scheduler-identity", default=None, help=argparse.SUPPRESS)
        if name == "enable":
            child.add_argument("--time", required=True, metavar="HH:MM", help="Daily local update time")
            child.add_argument("--plan-time", action="append", default=[], metavar="HH:MM",
                               help="Optional check-only time; repeat for multiple times")
        if name in {"plan", "run-now"}:
            child.add_argument("--branch", default=argparse.SUPPRESS, metavar="NAME")
            child.add_argument("--channel", default=argparse.SUPPRESS, metavar="CHANNEL")
