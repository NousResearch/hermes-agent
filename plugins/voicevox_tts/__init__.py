from __future__ import annotations

from .core import VoicevoxTTSProvider


def register(ctx) -> None:
    ctx.register_tts_provider(VoicevoxTTSProvider())
