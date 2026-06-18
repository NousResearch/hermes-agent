"""Unified OSINT agent — SitDeck + WM Free + scrapling + PDB cron."""

from __future__ import annotations

from . import core
from .cli import osint_agent_command, register_cli

_TOOLS = (
    ("osint_agent_status", core.STATUS_SCHEMA, core.handle_status, "🛰️"),
    ("osint_agent_brief", core.BRIEF_SCHEMA, core.handle_brief, "📡"),
)


def register(ctx) -> None:
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="osint_agent",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            emoji=emoji,
        )

    ctx.register_command(
        "osint-agent",
        handler=core.handle_slash,
        description="Unified OSINT brief (SitDeck + WM Free + gov RSS + MHLW).",
        args_hint="[status|brief]",
    )
    ctx.register_cli_command(
        name="osint-agent",
        help="Unified OSINT agent briefs and cron",
        setup_fn=register_cli,
        handler_fn=osint_agent_command,
        description=(
            "Integrate SitDeck browser crawl, World Monitor Free PDB, "
            "scrapling-feeds government RSS, and MHLW monitor."
        ),
    )
