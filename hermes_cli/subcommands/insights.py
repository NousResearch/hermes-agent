"""``hermes insights`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_insights_parser(subparsers, *, cmd_insights: Callable) -> None:
    """Attach the ``insights`` subcommand to ``subparsers``."""
    insights_parser = subparsers.add_parser(
        "insights",
        help="Show usage insights and analytics",
        description="Analyze session history to show token usage, costs, tool patterns, and activity trends",
    )
    insights_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze (default: 30)"
    )
    insights_parser.add_argument(
        "--source", help="Filter by platform (cli, telegram, discord, etc.)"
    )
    insights_parser.add_argument(
        "--provider", help="Filter usage by billing provider (e.g. openai-codex)"
    )
    insights_parser.add_argument(
        "--by-credential",
        action="store_true",
        help="Show per-credential token usage when credential telemetry is available",
    )
    insights_parser.set_defaults(func=cmd_insights)
