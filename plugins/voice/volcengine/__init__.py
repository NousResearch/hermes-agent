"""Volcengine (Doubao) voice plugin."""

from .tts import VolcengineTtsProvider


def register(ctx):
    """Register Volcengine TTS provider."""
    ctx.register_tts_provider(VolcengineTtsProvider())
