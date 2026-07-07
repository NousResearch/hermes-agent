"""``hermes behavior`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_behavior_parser(subparsers, *, cmd_behavior: Callable) -> None:
    """Attach the ``behavior`` subcommand to ``subparsers``."""
    behavior_parser = subparsers.add_parser(
        "behavior",
        help="Show behavioral analysis and insight cards",
        description="Analyze session history to show 5-axis behavioral scores and personality-driven insight cards",
    )
    behavior_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze (default: 30)"
    )
    behavior_parser.add_argument(
        "--source", help="Filter by platform (cli, telegram, discord, etc.)"
    )
    behavior_parser.set_defaults(func=cmd_behavior)