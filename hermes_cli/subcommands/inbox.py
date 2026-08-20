"""``hermes inbox`` subcommand parser.

One view of everything in flight and needing attention — background
processes, async delegations, cron jobs, and open chat surfaces.
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_inbox_parser(subparsers, *, cmd_inbox: Callable) -> None:
    """Attach the ``inbox`` subcommand to ``subparsers``."""
    inbox_parser = subparsers.add_parser(
        "inbox",
        help="Show everything in flight: background work, results, cron, sessions",
        description=(
            "A unified, read-only view of Hermes activity: what needs your "
            "attention (stalled delegations, failed cron runs, orphaned "
            "processes), what is in progress (background processes, async "
            "delegations), finished results not yet delivered, upcoming "
            "scheduled jobs, and open chat surfaces."
        ),
    )
    inbox_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the inbox snapshot as JSON",
    )
    inbox_parser.set_defaults(func=cmd_inbox)
