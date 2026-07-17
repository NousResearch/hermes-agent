"""Unified OSINT agent — SitDeck + WM Free + scrapling + Computer Use + web."""

from __future__ import annotations

from . import core
from .cli import osint_agent_command, register_cli

_TOOLS = (
    ("osint_agent_status", core.STATUS_SCHEMA, core.handle_status, "🛰️"),
    ("osint_agent_brief", core.BRIEF_SCHEMA, core.handle_brief, "📡"),
    (
        "osint_agent_computer_use_plan",
        core.CU_PLAN_SCHEMA,
        core.handle_computer_use_plan,
        "🖱️",
    ),
    (
        "osint_agent_multilayer_collect",
        core.MULTILAYER_SCHEMA,
        core.handle_multilayer_collect,
        "🕸️",
    ),
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
        description="Unified OSINT (SitDeck + WM + CU + web multilayer).",
        args_hint="[status|brief|cu|multilayer]",
    )
    ctx.register_cli_command(
        name="osint-agent",
        help="Unified OSINT agent briefs, Computer Use playbooks, cron",
        setup_fn=register_cli,
        handler_fn=osint_agent_command,
        description=(
            "Integrate SitDeck, World Monitor Free, Computer Use playbooks, "
            "multilayer web_search, government RSS, and MHLW monitor."
        ),
    )
