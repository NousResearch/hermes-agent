"""Unsloth Studio bridge for Hermes."""

from __future__ import annotations

from . import core
from .cli import register_cli, unsloth_studio_command


def _json_handler(fn):
    def handler(values=None, **kwargs):
        payload = values if isinstance(values, dict) else {}
        payload.update(kwargs)
        return core.to_json(fn(payload))

    return handler


def register(ctx) -> None:
    """Register Unsloth Studio tools, slash command, and CLI command."""
    ctx.register_tool(
        name="unsloth_studio_status",
        toolset="unsloth-studio",
        schema=core.STATUS_SCHEMA,
        handler=_json_handler(core.status_payload),
        check_fn=lambda: True,
        description=core.STATUS_SCHEMA["description"],
    )
    ctx.register_tool(
        name="unsloth_studio_start",
        toolset="unsloth-studio",
        schema=core.START_SCHEMA,
        handler=_json_handler(core.start_studio),
        check_fn=core.check_available,
        description=core.START_SCHEMA["description"],
    )
    ctx.register_tool(
        name="unsloth_studio_stop",
        toolset="unsloth-studio",
        schema=core.STOP_SCHEMA,
        handler=_json_handler(core.stop_studio),
        check_fn=lambda: True,
        description=core.STOP_SCHEMA["description"],
    )
    ctx.register_tool(
        name="unsloth_studio_install_info",
        toolset="unsloth-studio",
        schema=core.INSTALL_INFO_SCHEMA,
        handler=_json_handler(core.install_info),
        check_fn=lambda: True,
        description=core.INSTALL_INFO_SCHEMA["description"],
    )
    ctx.register_command(
        "unsloth-studio",
        handler=lambda raw_args: core.handle_slash(raw_args),
        description="Inspect, launch, and stop the local Unsloth Studio UI.",
        args_hint="[status|start|stop|install-info]",
    )
    ctx.register_cli_command(
        name="unsloth-studio",
        help="Local Unsloth Studio launcher",
        setup_fn=register_cli,
        handler_fn=unsloth_studio_command,
        description="Inspect, launch, and stop Unsloth Studio without reimplementing its UI.",
    )
