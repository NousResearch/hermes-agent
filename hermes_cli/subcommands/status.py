"""``hermes status`` subcommand parser."""

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_status_parser(subparsers, *, cmd_status: Callable) -> None:
    """Attach the ``status`` subcommand to ``subparsers``."""
    status_parser = subparsers.add_parser(
        "status", help="Show status of all components",
        description="Display status of Hermes Agent components")
    status_parser.add_argument(
        "--all", action="store_true", help="Show all details (redacted for sharing)")
    status_parser.add_argument(
        "--deep", action="store_true", help="Run deep checks (may take longer)")
    add_json_flag(status_parser, "Output status as JSON")
    status_parser.set_defaults(func=cmd_status)
