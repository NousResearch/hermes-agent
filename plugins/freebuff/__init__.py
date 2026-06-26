"""Hermes plugin bridge for Codebuff Freebuff."""

from __future__ import annotations

from .cli import freebuff_command, register_cli
from . import core


def register(ctx) -> None:
    """Register Freebuff CLI, gateway slash command, and service-gated tools."""
    ctx.register_cli_command(
        name="freebuff",
        help="Install and run Codebuff Freebuff free terminal coding agent",
        setup_fn=register_cli,
        handler_fn=freebuff_command,
        description=(
            "Wraps the Freebuff npm CLI (CodebuffAI/codebuff). Installs via npm, "
            "probes ~/.config/manicode/freebuff native binary, and launches the "
            "interactive TUI. Use freebuff_status/doctor tools from chat after "
            "enabling plugins.enabled: freebuff and tools.cli toolset freebuff."
        ),
    )

    ctx.register_command(
        "freebuff",
        handler=core.handle_slash,
        description="Install, doctor, or run Freebuff coding agent.",
        args_hint="[status|doctor|setup|install|run|skill]",
    )

    ctx.register_tool(
        name="freebuff_status",
        toolset="freebuff",
        schema=core.STATUS_SCHEMA,
        handler=core.handle_status,
        check_fn=lambda: True,
        description=core.STATUS_SCHEMA["description"],
        emoji="🆓",
    )
    ctx.register_tool(
        name="freebuff_doctor",
        toolset="freebuff",
        schema=core.DOCTOR_SCHEMA,
        handler=core.handle_doctor,
        check_fn=lambda: True,
        description=core.DOCTOR_SCHEMA["description"],
        emoji="🩺",
    )
    ctx.register_tool(
        name="freebuff_launch",
        toolset="freebuff",
        schema=core.LAUNCH_SCHEMA,
        handler=core.handle_launch,
        check_fn=lambda: True,
        description=core.LAUNCH_SCHEMA["description"],
        emoji="🚀",
    )
