"""Hermes plugin bridge for tegnike/aituber-kit."""

from __future__ import annotations

from . import core
from .cli import aituber_kit_command, register_cli

_CTX = None


def _ctx_llm():
    if _CTX is None:
        raise RuntimeError("aituber-kit plugin is not registered yet")
    return _CTX.llm


_TOOLS = (
    ("aituber_kit_status", core.STATUS_SCHEMA, core.handle_status, "🎭"),
    ("aituber_kit_configure", core.CONFIGURE_SCHEMA, core.handle_configure, "⚙️"),
    ("aituber_kit_install", core.INSTALL_SCHEMA, core.handle_install, "📦"),
    ("aituber_kit_prepare", core.PREPARE_SCHEMA, core.handle_prepare, "🛠️"),
    ("aituber_kit_start", core.START_SCHEMA, core.handle_start, "▶️"),
    ("aituber_kit_stop", core.STOP_SCHEMA, core.handle_stop, "⏹️"),
    ("aituber_kit_speak", core.SPEAK_SCHEMA, core.handle_speak, "🗣️"),
    ("aituber_kit_chat", core.CHAT_SCHEMA, core.handle_chat, "💬"),
    ("aituber_kit_stop_playback", core.STOP_PLAYBACK_SCHEMA, core.handle_stop_playback, "🛑"),
    ("aituber_kit_bridge_start", core.BRIDGE_START_SCHEMA, core.handle_bridge_start, "🔗"),
    ("aituber_kit_bridge_stop", core.BRIDGE_STOP_SCHEMA, core.handle_bridge_stop, "🔌"),
)


def register(ctx) -> None:
    """Register AITuberKit tools, slash command, and CLI."""
    global _CTX
    _CTX = ctx
    core.bind_llm_factory(_ctx_llm)

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="aituber_kit",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "aituber-kit",
        handler=core.handle_slash,
        description="Install and operate tegnike/aituber-kit (dev server, v1 API, Hermes bridge).",
        args_hint="[status|install|prepare|start|speak|chat|bridge-start]",
    )
    ctx.register_cli_command(
        name="aituber-kit",
        help="AITuberKit install, dev server, API control, and External Linkage bridge",
        setup_fn=register_cli,
        handler_fn=aituber_kit_command,
        description=(
            "Clone https://github.com/tegnike/aituber-kit, run the Next.js dev server, "
            "drive /api/v1 speak/chat/stop endpoints, and optionally expose a Hermes "
            "External Linkage WebSocket bridge on ws://127.0.0.1:8000/ws."
        ),
    )
