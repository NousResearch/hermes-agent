from __future__ import annotations

from . import core
from .cli import register_cli, tookie_osint_command


def _json_handler(fn):
    def handler(values=None, **kwargs):
        payload = values if isinstance(values, dict) else {}
        payload.update(kwargs)
        return core.to_json(fn(payload))

    return handler


def register(ctx) -> None:
    ctx.register_tool(
        name="tookie_status",
        toolset="tookie_osint",
        schema=core.STATUS_SCHEMA,
        handler=_json_handler(core.status_payload),
        check_fn=lambda: True,
        description=core.STATUS_SCHEMA["description"],
        emoji="O",
    )
    ctx.register_tool(
        name="tookie_scan_username",
        toolset="tookie_osint",
        schema=core.SCAN_SCHEMA,
        handler=_json_handler(core.scan_username),
        check_fn=core.check_available,
        description=core.SCAN_SCHEMA["description"],
        emoji="O",
    )
    ctx.register_command(
        "tookie-osint",
        handler=lambda raw_args: core.handle_slash(raw_args),
        description="Run Tookie-OSINT username scans.",
        args_hint="[status|scan <username>]",
    )
    ctx.register_cli_command(
        name="tookie-osint",
        help="Tookie-OSINT username scanner",
        setup_fn=register_cli,
        handler_fn=tookie_osint_command,
        description="Configure and run Alfredredbird/tookie-osint from Hermes.",
    )
