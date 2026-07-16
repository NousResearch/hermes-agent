from __future__ import annotations

from . import core
from .cli import register_cli, sillytavern_command


def register(ctx) -> None:
    ctx.register_tool(
        name="sillytavern_capabilities",
        toolset=core.TOOLSET,
        schema=core.CAPABILITIES_SCHEMA,
        handler=core.handle_capabilities,
        check_fn=lambda: True,
        description=core.CAPABILITIES_SCHEMA["description"],
        emoji="ST",
    )
    ctx.register_tool(
        name="sillytavern_status",
        toolset=core.TOOLSET,
        schema=core.STATUS_SCHEMA,
        handler=core.handle_status,
        check_fn=lambda: True,
        description=core.STATUS_SCHEMA["description"],
        emoji="ST",
    )
    ctx.register_tool(
        name="sillytavern_start",
        toolset=core.TOOLSET,
        schema=core.START_SCHEMA,
        handler=core.handle_start,
        check_fn=core.check_available,
        description=core.START_SCHEMA["description"],
        emoji="ST",
    )
    ctx.register_tool(
        name="sillytavern_stop",
        toolset=core.TOOLSET,
        schema=core.STOP_SCHEMA,
        handler=core.handle_stop,
        check_fn=lambda: True,
        description=core.STOP_SCHEMA["description"],
        emoji="ST",
    )
    ctx.register_tool(
        name="sillytavern_generate",
        toolset=core.TOOLSET,
        schema=core.GENERATE_SCHEMA,
        handler=core.handle_generate,
        check_fn=lambda: True,
        description=core.GENERATE_SCHEMA["description"],
        emoji="ST",
    )
    ctx.register_command(
        "sillytavern",
        handler=core.handle_slash,
        description="Inspect or use the pinned SillyTavern bridge.",
        args_hint="[status|start|stop|generate <prompt>]",
    )
    ctx.register_cli_command(
        name="sillytavern",
        help="Use the pinned SillyTavern frontend and local generation bridge",
        setup_fn=register_cli,
        handler_fn=sillytavern_command,
        description=(
            "Start, stop, inspect, and send a bounded chat-completions request "
            "through the pinned SillyTavern submodule."
        ),
    )
