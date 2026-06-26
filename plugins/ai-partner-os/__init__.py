"""Hermes plugin bridge for AI Partner OS."""

from __future__ import annotations

from . import core
from .cli import ai_partner_os_command, register_cli

_CTX = None


def _ctx_llm():
    if _CTX is None:
        raise RuntimeError("ai-partner-os plugin is not registered yet")
    return _CTX.llm


_TOOLS = (
    ("ai_partner_os_status", core.STATUS_SCHEMA, core.handle_status, "🖥️"),
    ("ai_partner_os_configure", core.CONFIGURE_SCHEMA, core.handle_configure, "⚙️"),
    ("ai_partner_os_connect_gui", core.CONNECT_GUI_SCHEMA, core.handle_connect_gui, "🪟"),
    ("ai_partner_os_start", core.START_SCHEMA, core.handle_start, "▶️"),
    ("ai_partner_os_stop", core.STOP_SCHEMA, core.handle_stop, "⏹️"),
    ("ai_partner_os_enable_lan", core.ENABLE_LAN_SCHEMA, core.handle_enable_lan, "📱"),
    ("ai_partner_os_discover_lan", core.DISCOVER_LAN_SCHEMA, core.handle_discover_lan, "🔍"),
    ("ai_partner_os_tts_status", core.TTS_STATUS_SCHEMA, core.handle_tts_status, "🔊"),
    ("ai_partner_os_start_tts", core.START_TTS_SCHEMA, core.handle_start_tts, "🎙️"),
    ("ai_partner_os_chat", core.CHAT_SCHEMA, core.handle_chat, "💬"),
    ("ai_partner_os_gui_say", core.GUI_SAY_SCHEMA, core.handle_gui_say, "🎭"),
    ("ai_partner_os_speak", core.SPEAK_SCHEMA, core.handle_speak, "🗣️"),
    ("ai_partner_os_action", core.ACTION_SCHEMA, core.handle_action, "🎮"),
    ("ai_partner_os_bridge_start", core.BRIDGE_START_SCHEMA, core.handle_bridge_start, "🔗"),
    ("ai_partner_os_bridge_stop", core.BRIDGE_STOP_SCHEMA, core.handle_bridge_stop, "🔌"),
)


def register(ctx) -> None:
    """Register AI Partner OS tools, slash command, CLI, and GUI conversation surface."""
    global _CTX
    _CTX = ctx
    core.bind_llm_factory(_ctx_llm)

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="ai_partner_os",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "ai-partner-os",
        handler=core.handle_slash,
        description="AI Partner OS as Hermes GUI — connect, chat, Hermes TTS, LAN/OS actions.",
        args_hint="[connect-gui|chat|gui-say|discover-lan|tts-status]",
    )
    ctx.register_cli_command(
        name="ai-partner-os",
        help="AI Partner OS Hermes GUI bridge (LLM + VOICEVOX/irodori TTS + avatar UI)",
        setup_fn=register_cli,
        handler_fn=ai_partner_os_command,
        description=(
            "Use AI Partner OS as the Hermes Agent visual desktop GUI. "
            "Voice uses Hermes irodoriTTS/VOICEVOX via play_tts_on_pc, not in-app TTS."
        ),
    )
    ctx.register_conversation_plugin(
        "ai-partner-os",
        prompt_builder=core.conversation_prompt,
        matcher=core.matches_conversation_event,
    )
