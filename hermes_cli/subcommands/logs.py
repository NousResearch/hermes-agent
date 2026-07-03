"""``hermes logs`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import argparse
from typing import Callable


def build_logs_parser(subparsers, *, cmd_logs: Callable) -> None:
    """Attach the ``logs`` subcommand to ``subparsers``."""
    # =========================================================================
    # logs command
    # =========================================================================
    logs_parser = subparsers.add_parser(
        "logs",
        help="View and filter Hermes log files",
        description="View, tail, and filter agent.log / errors.log / gateway.log / gui.log / desktop.log / codex-service-tier.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    hermes logs                    Show last 50 lines of agent.log
    hermes logs -f                 Follow agent.log in real time
    hermes logs errors             Show last 50 lines of errors.log
    hermes logs gateway -n 100     Show last 100 lines of gateway.log
    hermes logs gui -f             Follow gui.log in real time
    hermes logs desktop -f         Follow desktop.log (Electron app boot/backend)
    hermes logs codex-tier -n 20   Show requested/effective Codex service_tier records
    hermes logs --level WARNING    Only show WARNING and above
    hermes logs --session abc123   Filter by session ID
    hermes logs --component tools  Only show tool-related lines
    hermes logs --since 1h         Lines from the last hour
    hermes logs --since 30m -f     Follow, starting from 30 min ago
    hermes logs list               List available log files with sizes
""",
    )
    logs_parser.add_argument(
        "log_name",
        nargs="?",
        default="agent",
        help="Log to view: agent (default), errors, gateway, gui, desktop, codex-tier, or 'list' to show available files",
    )
    logs_parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=50,
        help="Number of lines to show (default: 50)",
    )
    logs_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Follow the log in real time (like tail -f)",
    )
    logs_parser.add_argument(
        "--level",
        metavar="LEVEL",
        help="Minimum log level to show (DEBUG, INFO, WARNING, ERROR)",
    )
    logs_parser.add_argument(
        "--session",
        metavar="ID",
        help="Filter lines containing this session ID substring",
    )
    logs_parser.add_argument(
        "--since",
        metavar="TIME",
        help="Show lines since TIME ago (e.g. 1h, 30m, 2d)",
    )
    logs_parser.add_argument(
        "--component",
        metavar="NAME",
        help="Filter by component: gateway, agent, tools, cli, cron, gui",
    )
    logs_parser.set_defaults(func=cmd_logs)
