"""Voice calling plugin — guarded phone calls through a FastAPI voice service."""

from __future__ import annotations

from .config import voice_call_available
from .schemas import VOICE_CALL_SCHEMA
from .tools import voice_call


def register(ctx) -> None:
    ctx.register_tool(
        name="voice_call",
        toolset="voice_call",
        schema=VOICE_CALL_SCHEMA,
        handler=voice_call,
        check_fn=voice_call_available,
        emoji="☎️",
    )
