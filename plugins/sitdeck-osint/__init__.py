"""SitDeck OSINT Hermes plugin — browser crawl replacement for World Monitor Pro MCP."""

from __future__ import annotations

from . import core
from .cli import register_cli, sitdeck_osint_command

_TOOLS = (
    ("sitdeck_status", core.STATUS_SCHEMA, core.handle_status, "🎴"),
    ("sitdeck_crawl", core.CRAWL_SCHEMA, core.handle_crawl, "🌐"),
    ("sitdeck_osint_digest", core.DIGEST_SCHEMA, core.handle_digest, "📋"),
)


def register(ctx) -> None:
    """Register SitDeck OSINT tools, slash command, and CLI."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="sitdeck_osint",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            emoji=emoji,
        )

    ctx.register_command(
        "sitdeck-osint",
        handler=core.handle_slash,
        description="SitDeck browser OSINT (no World Monitor Pro MCP).",
        args_hint="[status|crawl|digest]",
    )
    ctx.register_cli_command(
        name="sitdeck-osint",
        help="SitDeck browser OSINT crawl",
        setup_fn=register_cli,
        handler_fn=sitdeck_osint_command,
        description=(
            "Log into app.sitdeck.com with ~/.hermes/.env credentials and "
            "crawl dashboard intelligence for Hermes OSINT workflows."
        ),
    )
