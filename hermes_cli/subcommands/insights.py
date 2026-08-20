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
    insights_parser.set_defaults(func=cmd_insights)


def build_models_usage_parser(subparsers, *, cmd_models_usage: Callable) -> None:
    """Attach the ``models-usage`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "models-usage",
        help="Show per-model token/cost usage with daily charts",
        description=(
            "Break down local session usage by model: estimated cost, tokens, "
            "API calls, and a per-day cost series. Human-readable bar charts "
            "by default; --json emits the machine-readable report."
        ),
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze (default: 30)"
    )
    parser.add_argument(
        "--source", help="Filter by platform (cli, telegram, discord, etc.)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the machine-readable JSON report"
    )
    parser.add_argument(
        "--top", type=int, default=5, help="Top N models to chart (default: 5)"
    )
    parser.set_defaults(func=cmd_models_usage)
