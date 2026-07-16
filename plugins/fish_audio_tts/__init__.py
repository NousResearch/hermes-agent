from __future__ import annotations

import json
from typing import Any

from .core import FishAudioTTSProvider, status_payload, synthesize_text


def _status_handler(_: Any = None, **__: Any) -> str:
    return json.dumps(status_payload(), ensure_ascii=False, indent=2)


def _synthesize_handler(
    text: str,
    output_path: str | None = None,
    voice: str | None = None,
    model: str | None = None,
    format: str | None = None,
    speed: float | None = None,
    **_: Any,
) -> str:
    result = synthesize_text(
        text=text,
        output_path=output_path,
        voice=voice,
        model=model,
        output_format=format,
        speed=speed,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def register(ctx) -> None:
    ctx.register_tts_provider(FishAudioTTSProvider())

    ctx.register_tool(
        name="fish_audio_tts_status",
        toolset="tts",
        schema={"type": "object", "properties": {}},
        handler=_status_handler,
        check_fn=lambda: status_payload()["available"],
        requires_env=["FISH_AUDIO_API_KEY"],
        description="Report Fish Audio TTS API credential and provider readiness without revealing the key.",
    )

    ctx.register_tool(
        name="fish_audio_tts_synthesize",
        toolset="tts",
        schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to synthesize."},
                "output_path": {"type": "string", "description": "Optional destination audio path."},
                "voice": {"type": "string", "description": "Fish Audio reference_id voice model."},
                "model": {"type": "string", "description": "Fish Audio model header, e.g. s2.1-pro-free."},
                "format": {"type": "string", "description": "Audio format: mp3, wav, opus, or pcm."},
                "speed": {"type": "number", "description": "Speech speed from 0.5 to 2.0."},
            },
            "required": ["text"],
        },
        handler=_synthesize_handler,
        check_fn=lambda: status_payload()["available"],
        requires_env=["FISH_AUDIO_API_KEY"],
        description="Synthesize speech with Fish Audio through the official REST API.",
    )

    from .cli import register_cli

    ctx.register_cli_command(
        name="fish-audio-tts",
        help="Fish Audio cloud TTS backend",
        setup_fn=register_cli,
        description="Manage and invoke the Fish Audio TTS REST API backend.",
    )
