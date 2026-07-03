from __future__ import annotations

from plugins.openclaw_bridge.tools import (
    OPENCLAW_DELEGATE_SCHEMA,
    handle_openclaw_delegate,
    handle_openclaw_dry_run,
    pre_gateway_dispatch,
)


def register(ctx) -> None:
    ctx.register_tool(
        name="openclaw_delegate",
        toolset="openclaw",
        schema=OPENCLAW_DELEGATE_SCHEMA,
        handler=handle_openclaw_delegate,
        description="Delegate an approved execution-only task to OpenClaw and return results to Hermes.",
        emoji="OC",
    )
    ctx.register_command(
        "openclaw-dry-run",
        handler=handle_openclaw_dry_run,
        description="Validate Hermes-to-OpenClaw bridge routing without giving OpenClaw conversation control.",
        args_hint="<objective>",
    )
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
