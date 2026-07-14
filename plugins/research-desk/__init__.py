from __future__ import annotations

from . import cli, core


def register(ctx) -> None:
    handlers = core.make_handlers(ctx)

    ctx.register_tool(
        name="research_desk_status",
        toolset=core.TOOLSET,
        schema=core.STATUS_SCHEMA,
        handler=handlers["status"],
        check_fn=lambda: True,
        description=core.STATUS_SCHEMA["description"],
        emoji="RD",
    )
    ctx.register_tool(
        name="research_desk_plan",
        toolset=core.TOOLSET,
        schema=core.PLAN_SCHEMA,
        handler=handlers["plan"],
        check_fn=lambda: True,
        description=core.PLAN_SCHEMA["description"],
        emoji="RD",
    )
    ctx.register_tool(
        name="research_desk_run",
        toolset=core.TOOLSET,
        schema=core.RUN_SCHEMA,
        handler=handlers["run"],
        check_fn=lambda: True,
        description=core.RUN_SCHEMA["description"],
        emoji="RD",
    )
    ctx.register_tool(
        name="research_desk_export",
        toolset=core.TOOLSET,
        schema=core.EXPORT_SCHEMA,
        handler=handlers["export"],
        check_fn=lambda: True,
        description=core.EXPORT_SCHEMA["description"],
        emoji="RD",
    )
    ctx.register_command(
        "research-desk",
        handler=handlers["slash"],
        description="Inspect or operate the active Research Desk profile.",
        args_hint="[status]",
    )
    ctx.register_cli_command(
        name="research-desk",
        help="Create and run public-evidence Research Desk reports",
        setup_fn=cli.register_cli,
        handler_fn=lambda args: cli.research_desk_command(args, ctx=ctx),
        description=(
            "Use the active profile and configured Private Runner workspace for "
            "allowlisted public research, approval, and redacted receipts."
        ),
    )
