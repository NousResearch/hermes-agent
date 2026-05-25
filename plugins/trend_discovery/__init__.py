"""Trend Discovery Center plugin.

Provides an operator CLI and slash command for a reliable trend-scanning
workflow. The implementation is local-first and does not depend on n8n or
Open Crawl; those can be attached later as optional source adapters.
"""

from __future__ import annotations

from .cli import register_cli, trend_discovery_command


def register(ctx) -> None:
    ctx.register_cli_command(
        name="trend-discovery",
        help="Operate the Hermes Trend Discovery Center",
        setup_fn=register_cli,
        handler_fn=trend_discovery_command,
        description=(
            "Persistent phase tracking, reliable multi-source scanning, "
            "watchdog alerts, knowledge writeback, and numeric compliance."
        ),
    )
    ctx.register_command(
        "trend-discovery",
        lambda raw_args: trend_discovery_command.from_slash(raw_args),
        description="Show Trend Discovery Center status",
        args_hint="[status|comply|watchdog]",
    )
