from __future__ import annotations

from . import core
from .cli import openmanus_command, register_cli


def register(ctx) -> None:
    ctx.register_tool(
        name="openmanus_capabilities",
        toolset=core.TOOLSET,
        schema=core.CAPABILITIES_SCHEMA,
        handler=core.handle_capabilities,
        check_fn=lambda: True,
        description=core.CAPABILITIES_SCHEMA["description"],
        emoji="OM",
    )
    ctx.register_tool(
        name="openmanus_run",
        toolset=core.TOOLSET,
        schema=core.RUN_SCHEMA,
        handler=core.handle_run,
        check_fn=core.check_available,
        description=core.RUN_SCHEMA["description"],
        emoji="OM",
    )

    def wide_handler(args=None, **kwargs):
        try:
            host_llm = ctx.llm
        except Exception:
            host_llm = None
        return core.handle_wide_research(args, host_llm=host_llm, **kwargs)

    ctx.register_tool(
        name="openmanus_wide_research",
        toolset=core.TOOLSET,
        schema=core.WIDE_RESEARCH_SCHEMA,
        handler=wide_handler,
        check_fn=core.check_available,
        description=core.WIDE_RESEARCH_SCHEMA["description"],
        emoji="WR",
    )
    ctx.register_command(
        "openmanus",
        handler=core.handle_slash,
        description="Inspect the OpenManus bridge and its safety policy.",
        args_hint="[status|capabilities]",
    )
    ctx.register_cli_command(
        name="openmanus",
        help="Run the pinned OpenManus agent through Hermes safety gates",
        setup_fn=register_cli,
        handler_fn=openmanus_command,
        description=(
            "Use OpenManus from Hermes or MoA workers with bounded steps, an "
            "authorised workspace, optional isolated parallel research, and "
            "durable redacted receipts."
        ),
    )
