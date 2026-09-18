"""``hermes health`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_health_parser(subparsers, *, cmd_health: Callable) -> None:
    """Attach the ``health`` subcommand to ``subparsers``."""
    health_parser = subparsers.add_parser(
        "health",
        help="Run automation-friendly offline health checks",
        description=(
            "Run low-cost Hermes health checks with stable exit-code semantics: "
            "0=healthy, 1=warning/degraded, 2=critical. No provider or network "
            "probes are run. Use `hermes doctor` for interactive diagnostics."
        ),
    )
    health_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    health_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print nothing when healthy; useful for cron, watchdogs, and service health checks",
    )
    health_parser.set_defaults(func=cmd_health)
