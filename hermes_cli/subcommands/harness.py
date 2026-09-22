"""``hermes harness`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_harness_parser(subparsers, *, cmd_harness: Callable) -> None:
    parser = subparsers.add_parser(
        "harness", help="Inspect and change typed harness overlays",
        description="Manage typed, reversible overlays over the stock Hermes runtime")
    commands = parser.add_subparsers(dest="harness_command")

    show = commands.add_parser("show", help="Show harness identity and overlay lineage")
    add_json_flag(show, "Print machine-readable JSON")

    diff = commands.add_parser("diff", help="Diff the effective harness against stock")
    add_json_flag(diff, "Print machine-readable JSON")

    explain = commands.add_parser("explain", help="Explain one tunable harness option")
    explain.add_argument("key", help="Dotted option name")
    add_json_flag(explain, "Print machine-readable JSON")

    set_cmd = commands.add_parser("set", help="Set a value in a named overlay")
    set_cmd.add_argument("overlay", help="Overlay name")
    set_cmd.add_argument("key", help="Dotted option name")
    set_cmd.add_argument("value", help="YAML scalar/list/object value")
    set_cmd.add_argument("--reason", required=True, help="Why this harness change is being made")

    revert = commands.add_parser("revert", help="Remove a named overlay (latest when omitted)")
    revert.add_argument("overlay", nargs="?", help="Overlay name")
    revert.add_argument("--reason", required=True, help="Why this overlay is being reverted")

    parser.set_defaults(func=cmd_harness)
