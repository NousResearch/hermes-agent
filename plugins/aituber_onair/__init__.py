"""AITuber OnAir bridge for Hermes."""

from __future__ import annotations

from . import core
from .cli import aituber_onair_command, register_cli

_TOOLS = (
    ("aituber_onair_status", core.STATUS_SCHEMA, core.handle_status, "A"),
    (
        "aituber_onair_configure_hakua",
        core.CONFIGURE_HAKUA_SCHEMA,
        core.handle_configure_hakua,
        "A",
    ),
    ("aituber_onair_prepare", core.PREPARE_SCHEMA, core.handle_prepare, "A"),
    ("aituber_onair_start", core.START_SCHEMA, core.handle_start, "A"),
    ("aituber_onair_stop", core.STOP_SCHEMA, core.handle_stop, "A"),
    ("aituber_onair_tts_status", core.TTS_STATUS_SCHEMA, core.handle_tts_status, "A"),
    ("aituber_onair_start_tts", core.START_TTS_SCHEMA, core.handle_start_tts, "A"),
    ("aituber_onair_speak", core.SPEAK_SCHEMA, core.handle_speak, "A"),
    ("aituber_onair_say", core.SAY_SCHEMA, core.handle_say, "A"),
    ("aituber_onair_smoke", core.SMOKE_SCHEMA, core.handle_smoke, "A"),
)


def register(ctx) -> None:
    """Register AITuber OnAir tools, slash command, and CLI command."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="aituber-onair",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "aituber",
        handler=core.handle_slash,
        description="Run AITuber OnAir FBX streaming and Hakua Codex character chat.",
        args_hint="[status|configure|prepare|start|stop|say|smoke]",
    )
    ctx.register_cli_command(
        name="aituber-onair",
        help="AITuber OnAir FBX and Hakua Codex bridge",
        setup_fn=register_cli,
        handler_fn=aituber_onair_command,
        description=(
            "Configure Hakua, prepare AITuber OnAir's Codex SDK example, "
            "start the FBX app, run Codex-auth character prompts, and route Hakua voice."
        ),
    )
