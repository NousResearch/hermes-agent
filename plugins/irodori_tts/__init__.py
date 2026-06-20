from __future__ import annotations

import json
from typing import Any

from .core import IrodoriScriptTTSProvider, synthesize_text, status_payload


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
    provider = IrodoriScriptTTSProvider()
    ctx.register_tts_provider(provider)

    ctx.register_tool(
        name="irodori_tts_status",
        toolset="tts",
        schema={
            "type": "object",
            "properties": {},
        },
        handler=_status_handler,
        check_fn=lambda: status_payload()["available"],
        description="Report local Irodori TTS script, server, and provider status.",
    )

    ctx.register_tool(
        name="irodori_tts_synthesize",
        toolset="tts",
        schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to synthesize.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional destination audio path.",
                },
                "voice": {
                    "type": "string",
                    "description": "Irodori voice id; defaults to none.",
                },
                "model": {
                    "type": "string",
                    "description": "Irodori model id.",
                },
                "format": {
                    "type": "string",
                    "description": "Audio format: wav, mp3, flac, opus, aac, or pcm.",
                },
                "speed": {
                    "type": "number",
                    "description": "Speech speed multiplier.",
                },
            },
            "required": ["text"],
        },
        handler=_synthesize_handler,
        check_fn=lambda: status_payload()["available"],
        description="Synthesize speech with local Irodori TTS through the Windows script harness.",
    )

    from .cli import register_cli

    ctx.register_cli_command(
        name="irodori-tts",
        help="Local Irodori TTS script backend",
        setup_fn=register_cli,
        description="Manage and invoke the local Irodori TTS script backend.",
    )
