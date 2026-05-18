from __future__ import annotations

from .schemas import TOOL_SCHEMAS
from .tools import (
    TOOL_HANDLERS,
    crypto_bot_pm_status_slash,
)


TOOLSET = "crypto_bot_pm"


def check_requirements() -> bool:
    return True


def register(ctx) -> None:
    for name, schema in TOOL_SCHEMAS.items():
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=TOOL_HANDLERS[name],
            check_fn=check_requirements,
            description=schema.get("description", ""),
        )

    ctx.register_command(
        "crypto-bot-pm-status",
        handler=crypto_bot_pm_status_slash,
        description="Run read-only crypto_bot PM status through the safe PM bridge.",
        args_hint="[text]",
    )
