"""Hermes plugin: autonomous VRChat chatbox, conversation loop, and movement."""

from __future__ import annotations

from .cli import register_cli, vrchat_autonomy_command
from . import core

def _always_available() -> bool:
    return True


_TOOLS = (
    ("vrchat_autonomy_plugin_status", core.STATUS_SCHEMA, core.handle_status, "🌐", core.check_available),
    ("vrchat_autonomy_plugin_chatbox", core.CHATBOX_SCHEMA, core.handle_chatbox, "💬", core.check_available),
    ("vrchat_autonomy_plugin_move", core.MOVE_SCHEMA, core.handle_move, "🚶", core.check_available),
    ("vrchat_autonomy_plugin_tick", core.TICK_SCHEMA, core.handle_tick, "🔁", core.check_available),
    ("vrchat_autonomy_plugin_enqueue", core.ENQUEUE_SCHEMA, core.handle_enqueue, "📥", core.check_available),
    (
        "vrchat_autonomy_plugin_neuro_status",
        core.NEURO_STATUS_SCHEMA,
        core.handle_neuro_status,
        "🧠",
        _always_available,
    ),
    (
        "vrchat_autonomy_plugin_neuro_bootstrap",
        core.NEURO_BOOTSTRAP_SCHEMA,
        core.handle_neuro_bootstrap,
        "🧠",
        _always_available,
    ),
    (
        "vrchat_autonomy_plugin_neuro_handle_action",
        core.NEURO_HANDLE_SCHEMA,
        core.handle_neuro_action,
        "🧠",
        core.check_available,
    ),
)


def register(ctx) -> None:
    """Register VRChat autonomy tools, slash command, and CLI."""
    for name, schema, handler, emoji, check_fn in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="vrchat_autonomy",
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "vrchat-autonomy",
        handler=core.handle_slash,
        description="Autonomous VRChat chatbox, conversation loop, and movement.",
        args_hint="[status|doctor|tick|start|stop]",
    )
    ctx.register_cli_command(
        name="vrchat-autonomy",
        help="VRChat autonomous chatbox, conversation loop, and OSC movement",
        setup_fn=register_cli,
        handler_fn=vrchat_autonomy_command,
        description=(
            "Wraps the Hermes VRChat autonomy stack (profile tick, observation queue, "
            "ChatBox/TTS actuation, OSC movement) plus VedalAI Neuro API bridge helpers "
            "(neuro-sdk vendor, bootstrap, action routing). Live OSC requires python-osc."
        ),
    )
