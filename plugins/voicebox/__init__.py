from __future__ import annotations

import json
from typing import Any

from .core import (
    VoiceboxTTSProvider,
    ensure_hakua_profile,
    status_payload,
    synthesize_text,
    transcribe_audio,
)


def _tool_args(args: Any) -> dict[str, Any]:
    return args if isinstance(args, dict) else {}


def _status_handler(args: Any = None, **__: Any) -> str:
    del args
    return json.dumps(status_payload(), ensure_ascii=False, indent=2)


def _synthesize_handler(args: Any = None, **__: Any) -> str:
    data = _tool_args(args)
    result = synthesize_text(
        text=str(data.get("text") or ""),
        output_path=data.get("output_path"),
        voice=data.get("voice"),
        model=data.get("model"),
        language=data.get("language"),
        personality=data.get("personality"),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def _transcribe_handler(args: Any = None, **__: Any) -> str:
    data = _tool_args(args)
    result = transcribe_audio(
        audio_path=str(data.get("audio_path") or ""),
        language=data.get("language"),
        model=data.get("model"),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def register(ctx) -> None:
    provider = VoiceboxTTSProvider()
    ctx.register_tts_provider(provider)

    ctx.register_tool(
        name="voicebox_status",
        toolset="tts",
        schema={
            "type": "object",
            "properties": {},
        },
        handler=_status_handler,
        check_fn=lambda: status_payload()["available"],
        description="Report local Voicebox server health, profiles, and provider status.",
    )

    ctx.register_tool(
        name="voicebox_synthesize",
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
                    "description": "Voicebox profile name or id.",
                },
                "model": {
                    "type": "string",
                    "description": "Voicebox engine (qwen, kokoro, chatterbox_turbo, ...).",
                },
                "language": {
                    "type": "string",
                    "description": "Language code (en, ja, ...).",
                },
                "personality": {
                    "type": "boolean",
                    "description": "Rewrite text in the profile personality before TTS.",
                },
            },
            "required": ["text"],
        },
        handler=_synthesize_handler,
        check_fn=lambda: status_payload()["available"],
        description="Synthesize speech with local Voicebox via POST /speak.",
    )

    ctx.register_tool(
        name="voicebox_transcribe",
        toolset="tts",
        schema={
            "type": "object",
            "properties": {
                "audio_path": {
                    "type": "string",
                    "description": "Path to an audio file to transcribe.",
                },
                "language": {
                    "type": "string",
                    "description": "Optional language hint for Whisper.",
                },
                "model": {
                    "type": "string",
                    "description": "Whisper model size (base, small, medium, large, turbo).",
                },
            },
            "required": ["audio_path"],
        },
        handler=_transcribe_handler,
        check_fn=lambda: status_payload()["available"],
        description="Transcribe audio with local Voicebox Whisper via POST /transcribe.",
    )

    from .cli import register_cli

    ctx.register_cli_command(
        name="voicebox",
        help="Local Voicebox AI voice studio backend",
        setup_fn=register_cli,
        description="Manage and invoke the local Voicebox REST API backend.",
    )
