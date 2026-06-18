"""Scrapling-feeds — government RSS/Atom direct read for Hermes OSINT."""

from __future__ import annotations

from . import core
from .cli import register_cli, scrapling_feeds_command

_TOOLS = (
    ("gov_feed_status", core.STATUS_SCHEMA, core.handle_status, "🏛️"),
    ("gov_feed_fetch", core.FETCH_SCHEMA, core.handle_fetch, "📰"),
    ("gov_feed_digest", core.DIGEST_SCHEMA, core.handle_digest, "📋"),
    ("mhlw_designated_check", core.MHLW_CHECK_SCHEMA, core.handle_mhlw_check, "💊"),
)


def register(ctx) -> None:
    """Register government feed tools, slash command, and CLI."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="scrapling_feeds",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            emoji=emoji,
        )

    ctx.register_command(
        "scrapling-feeds",
        handler=core.handle_slash,
        description="Official government RSS feeds + MHLW 指定薬物 monitor.",
        args_hint="[status|fetch|digest|mhlw-check]",
    )
    ctx.register_cli_command(
        name="scrapling-feeds",
        help="Government RSS/Atom feeds (Scrapling + urllib)",
        setup_fn=register_cli,
        handler_fn=scrapling_feeds_command,
        description=(
            "Direct-read PRIMARY government feeds and MHLW 指定薬物部会 / "
            "施行・指定公表の定期監視。"
        ),
    )
