"""``hermes proactive`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_proactive_parser(subparsers, *, cmd_proactive: Callable) -> None:
    """Attach the ``proactive`` subcommand to ``subparsers``."""
    proactive_parser = subparsers.add_parser(
        "proactive",
        help="Preview proactive opportunities from repeated session patterns",
        description=(
            "Analyze recent session history and suggest reusable skills, quick "
            "commands, scheduled jobs, Kanban templates, or workflows."
        ),
    )
    proactive_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze (default: 30)"
    )
    proactive_parser.add_argument(
        "--source", help="Filter by platform (cli, telegram, discord, etc.)"
    )
    proactive_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum opportunities to show (default: 5)"
    )
    proactive_parser.add_argument(
        "--min-messages",
        type=int,
        default=2,
        help="Minimum similar user asks before proposing an opportunity (default: 2)",
    )
    proactive_parser.add_argument(
        "--json", action="store_true", help="Print the raw structured report as JSON"
    )
    proactive_parser.set_defaults(func=cmd_proactive)
