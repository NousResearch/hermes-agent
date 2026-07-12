"""Warashibe Reselling Plugin — わらしべ長者式せどり"""
from __future__ import annotations

__version__ = "1.1.0"


def register(ctx) -> None:
    """Register price research, CLI, and gateway slash command with Hermes."""
    from . import price_research
    from .slash import handle_warashibe
    from . import cli as cli_module

    ctx.register_tool(
        name=price_research.PRICE_RESEARCH_SCHEMA["name"],
        toolset="warashibe-reselling",
        schema=price_research.PRICE_RESEARCH_SCHEMA,
        handler=price_research.handle_price_research,
        check_fn=price_research.check_available,
        description=price_research.PRICE_RESEARCH_SCHEMA["description"],
    )
    ctx.register_command(
        "warashibe",
        handler=handle_warashibe,
        description="わらしべ長者式せどり: 公開価格調査・利益計算・台帳",
    )
    ctx.register_cli_command(
        name="warashibe",
        help="わらしべ長者式せどりCLI",
        setup_fn=cli_module.register_cli,
        handler_fn=cli_module.main,
        description="わらしべ長者式せどりCLI",
    )
