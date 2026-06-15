"""Google Colab CLI bridge for Hermes."""

from __future__ import annotations

from . import core
from .cli import google_colab_command, register_cli


def _json_handler(fn):
    def handler(values=None, **kwargs):
        payload = values if isinstance(values, dict) else {}
        payload.update(kwargs)
        return core.to_json(fn(payload))

    return handler


def register(ctx) -> None:
    """Register Colab tools, slash command, and CLI command."""
    ctx.register_tool(
        name="google_colab_status",
        toolset="google-colab",
        schema=core.STATUS_SCHEMA,
        handler=_json_handler(core.status_payload),
        check_fn=lambda: True,
        description=core.STATUS_SCHEMA["description"],
    )
    ctx.register_tool(
        name="google_colab_sessions",
        toolset="google-colab",
        schema=core.SESSIONS_SCHEMA,
        handler=_json_handler(core.sessions_payload),
        check_fn=core.check_available,
        description=core.SESSIONS_SCHEMA["description"],
    )
    ctx.register_tool(
        name="google_colab_run",
        toolset="google-colab",
        schema=core.RUN_SCHEMA,
        handler=_json_handler(core.run_job),
        check_fn=core.check_available,
        description=core.RUN_SCHEMA["description"],
    )
    ctx.register_tool(
        name="google_colab_sft_template",
        toolset="google-colab",
        schema=core.SFT_TEMPLATE_SCHEMA,
        handler=_json_handler(core.write_sft_template),
        check_fn=lambda: True,
        description=core.SFT_TEMPLATE_SCHEMA["description"],
    )
    ctx.register_command(
        "colab",
        handler=lambda raw_args: core.handle_slash(raw_args),
        description="Inspect and run Google Colab CLI accelerator jobs.",
        args_hint="[status|sessions|run|sft-template]",
    )
    ctx.register_cli_command(
        name="google-colab",
        help="Google Colab CLI accelerator job runner",
        setup_fn=register_cli,
        handler_fn=google_colab_command,
        description="Inspect Colab CLI state, run confirmed jobs, and generate SFT templates.",
    )
