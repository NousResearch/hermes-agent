"""ShinkaEvolve-OSINT Hermes plugin — MILSPEC OSINT for daily world-affairs briefings."""

from __future__ import annotations

from . import core
from .cli import register_cli, shinka_osint_command

_TOOLS = (
    ("shinka_osint_status", core.STATUS_SCHEMA, core.handle_status, "O"),
    ("shinka_osint_list_scenarios", core.LIST_SCENARIOS_SCHEMA, core.handle_list_scenarios, "O"),
    ("shinka_osint_analyze", core.ANALYZE_SCHEMA, core.handle_analyze, "O"),
    ("shinka_osint_briefing", core.BRIEFING_SCHEMA, core.handle_briefing, "O"),
    ("shinka_osint_verify", core.VERIFY_SCHEMA, core.handle_verify, "O"),
    ("shinka_osint_audit", core.AUDIT_SCHEMA, core.handle_audit, "O"),
)


def register(ctx) -> None:
    """Register ShinkaEvolve-OSINT tools, slash command, and CLI."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="shinka_osint",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            emoji=emoji,
        )

    ctx.register_command(
        "shinka-osint",
        handler=core.handle_slash,
        description="Run MILSPEC OSINT briefings via ShinkaEvolve-OSINT.",
        args_hint="[status|scenarios|analyze|briefing|verify|audit]",
    )
    ctx.register_cli_command(
        name="shinka-osint",
        help="ShinkaEvolve-OSINT world-affairs and security intelligence",
        setup_fn=register_cli,
        handler_fn=shinka_osint_command,
        description=(
            "Bridge to ShinkaEvolve-OSINT for MILSPEC-grade OSINT reports, "
            "daily security briefings, corpus verification, and audit logs."
        ),
    )
