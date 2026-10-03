"""``hermes harness`` command parser."""
from __future__ import annotations
from typing import Callable
from hermes_cli.subcommands._shared import add_json_flag


def build_harness_parser(subparsers, *, cmd_harness: Callable) -> None:
    parser = subparsers.add_parser(
        "harness", help="Inspect and change typed reversible harness overlays",
        description="Manage bounded harness overlays without editing source or config.yaml",
    )
    children = parser.add_subparsers(dest="harness_command")
    for name, help_text in (("show", "Show the effective harness manifest"), ("diff", "Show tunable values changed from stock")):
        child = children.add_parser(name, help=help_text)
        add_json_flag(child, "Print machine-readable JSON")
    explain = children.add_parser("explain", help="Explain one tunable key")
    explain.add_argument("key")
    add_json_flag(explain, "Print machine-readable JSON")
    set_parser = children.add_parser("set", help="Set a value in a named overlay")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    set_parser.add_argument("--overlay", required=True, help="Overlay id to create or update")
    set_parser.add_argument("--reason", required=True, help="Human/evolver provenance for the change")
    revert = children.add_parser("revert", help="Remove a named overlay")
    revert.add_argument("overlay")
    revert.add_argument("--yes", action="store_true", help="Confirm removal without an interactive prompt")
    parser.set_defaults(func=cmd_harness)
