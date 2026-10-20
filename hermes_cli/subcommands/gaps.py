"""``hermes gaps`` subcommand parser."""

from __future__ import annotations


def _cmd_gaps(args) -> None:
    from hermes_cli.affordance_gaps import cmd_gaps

    cmd_gaps(args)


def build_gaps_parser(subparsers) -> None:
    """Attach the ``gaps`` subcommand to ``subparsers``."""
    gaps_parser = subparsers.add_parser(
        "gaps",
        help="Review missing-capability affordance gaps",
        description="Record and review missing affordances discovered during task execution.",
    )
    gaps_sub = gaps_parser.add_subparsers(dest="gaps_action")
    gaps_list = gaps_sub.add_parser("list", help="List recent affordance gaps")
    gaps_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent gaps to show (default: 20)",
    )
    gaps_log = gaps_sub.add_parser("log", help="Manually log an affordance gap")
    gaps_log.add_argument("--goal", required=True, help="Task goal that could not be completed")
    gaps_log.add_argument(
        "--missing-capability",
        required=True,
        help="Tool, skill, knowledge, or permission that was missing",
    )
    gaps_log.add_argument("--failure", default="", help="Short failure description")
    gaps_log.add_argument("--tool", action="append", default=[], help="Available tool at time of failure")
    gaps_log.add_argument("--session-id", help="Session id associated with the gap")
    gaps_parser.set_defaults(func=_cmd_gaps)
