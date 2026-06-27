"""Local OpenClaw delegation bridge plugin."""

from __future__ import annotations

from .schemas import OPENCLAW_DELEGATE
from .tools import (
    _env_ready,
    handle_clawops_approve_command,
    handle_clawops_command,
    handle_clawops_run_command,
    handle_openclaw_dry_run_command,
    openclaw_delegate,
    pre_gateway_dispatch,
    set_clawops_host_llm,
)


def register(ctx) -> None:
    """Register the OpenClaw delegation tool."""
    try:
        set_clawops_host_llm(ctx.llm)
    except Exception:
        set_clawops_host_llm(None)

    ctx.register_tool(
        name="openclaw_delegate",
        toolset="openclaw_bridge",
        schema=OPENCLAW_DELEGATE,
        handler=openclaw_delegate,
        check_fn=_env_ready,
        description="Delegate approved mock-safe dry-run tasks to local OpenClaw.",
        emoji="OC",
    )
    ctx.register_command(
        "openclaw-dry-run",
        handle_openclaw_dry_run_command,
        description="Delegate an approved OpenClaw dry-run task.",
        args_hint="<request>",
    )
    ctx.register_command(
        "clawops",
        handle_clawops_command,
        description="Assign a task to ClawOps agents without external side effects.",
        args_hint="<request>",
    )
    ctx.register_command(
        "clawops-run",
        handle_clawops_run_command,
        description="Alias for /clawops.",
        args_hint="<request>",
    )
    ctx.register_command(
        "clawops-approve",
        handle_clawops_approve_command,
        description="Execute a pending ClawOps approval by id.",
        args_hint="<approval_id>",
    )
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
