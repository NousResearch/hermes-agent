"""Bundled ElevenLabs outbound-calling plugin."""

from __future__ import annotations

from plugins.elevenlabs_calls.tools import (
    ELEVENLABS_OUTBOUND_CALL_SCHEMA,
    _check_available,
    _handle_outbound_call,
)


def register(ctx) -> None:
    """Register the explicitly authorized outbound-call tool."""
    ctx.register_tool(
        name="elevenlabs_outbound_call",
        toolset="elevenlabs_calls",
        schema=ELEVENLABS_OUTBOUND_CALL_SCHEMA,
        handler=_handle_outbound_call,
        check_fn=_check_available,
        requires_env=["ELEVENLABS_API_KEY"],
        emoji="☎",
    )
