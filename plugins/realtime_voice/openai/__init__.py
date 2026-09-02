"""OpenAI Realtime speech-to-speech plugin — bundled, auto-loaded.

Registers :class:`OpenAIRealtimeProvider` through the same
``ctx.register_realtime_voice_provider()`` hook a user plugin would use.
Selected with ``hermes realtime --provider openai`` (the default).
"""

from __future__ import annotations

from plugins.realtime_voice.openai.provider import OpenAIRealtimeProvider


def register(ctx) -> None:
    """Plugin entry point — wire the OpenAI Realtime provider into the registry."""
    ctx.register_realtime_voice_provider(OpenAIRealtimeProvider())
