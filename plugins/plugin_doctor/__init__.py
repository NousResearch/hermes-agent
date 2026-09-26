"""Hermes plugin manifest/import validator."""

from __future__ import annotations

from . import core
from .cli import plugin_doctor_command, register_cli


def _json_handler(fn):
    def handler(values=None, **kwargs):
        payload = values if isinstance(values, dict) else {}
        payload.update(kwargs)
        return core.to_json(fn(payload))

    return handler


def register(ctx) -> None:
    """Register Plugin Doctor tools, slash command, and CLI command."""
    ctx.register_tool(
        name="plugin_doctor_scan",
        toolset="plugin-doctor",
        schema=core.SCAN_SCHEMA,
        handler=_json_handler(core.scan_plugins),
        check_fn=lambda: True,
        description=core.SCAN_SCHEMA["description"],
    )
    ctx.register_command(
        "plugin-doctor",
        handler=lambda raw_args: core.handle_slash(raw_args),
        description="Validate Hermes plugin manifests and import/register entry points.",
        args_hint="[plugins_dir]",
    )
    ctx.register_cli_command(
        name="plugin-doctor",
        help="Validate Hermes plugin manifests and imports",
        setup_fn=register_cli,
        handler_fn=plugin_doctor_command,
        description="Scan plugin.yaml metadata, imports, register(ctx), and duplicate names.",
    )
