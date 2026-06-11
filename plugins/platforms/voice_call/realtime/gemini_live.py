"""Gemini Live API session (speech-to-speech over websockets).

Same surface as the OpenAI session, different wire protocol: a ``setup``
message configures the model, caller audio streams as ``realtime_input``
media chunks (PCM16 @ 16 kHz), model audio arrives as inline data parts
(PCM16 @ 24 kHz), and ``interrupted`` server content signals barge-in.
"""

import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

from ..config import RealtimeConfig
from .base import (
    AGENT_CONSULT_TOOL,
    DEFAULT_INSTRUCTIONS,
    RealtimeEvent,
    RealtimeVoiceSession,
)

logger = logging.getLogger(__name__)

LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview"
DEFAULT_VOICE = "Puck"


class GeminiLiveSession(RealtimeVoiceSession):
    name = "gemini"
    input_sample_rate = 16000
    output_sample_rate = 24000

    def __init__(self, config: RealtimeConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Gemini Live requires GEMINI_API_KEY")
        self.model = config.model or DEFAULT_MODEL
        self.voice = config.voice or DEFAULT_VOICE
        self.instructions = config.instructions or DEFAULT_INSTRUCTIONS
        self._ws = None
        self._closed = False

    async def connect(self) -> None:
        import websockets

        self._ws = await websockets.connect(
            f"{LIVE_URL}?key={self.api_key}",
            max_size=16 * 1024 * 1024,
        )
        model = self.model if self.model.startswith("models/") else f"models/{self.model}"
        await self._send({
            "setup": {
                "model": model,
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": self.voice}
                        }
                    },
                },
                "system_instruction": {
                    "parts": [{"text": self.instructions}]
                },
                "tools": [{"function_declarations": [AGENT_CONSULT_TOOL]}],
            }
        })

    async def _send(self, message: Dict[str, Any]) -> None:
        if self._ws is None or self._closed:
            return
        await self._ws.send(json.dumps(message))

    async def send_audio(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        await self._send({
            "realtime_input": {
                "media_chunks": [
                    {
                        "mime_type": f"audio/pcm;rate={self.input_sample_rate}",
                        "data": base64.b64encode(pcm16).decode(),
                    }
                ]
            }
        })

    async def inject_text(self, text: str) -> None:
        await self._send({
            "client_content": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": f"Say this to the caller now: {text}"}],
                    }
                ],
                "turn_complete": True,
            }
        })

    async def cancel_response(self) -> None:
        """Gemini Live cancels generation automatically on caller speech
        (server-side VAD); there is no explicit cancel message."""

    async def send_tool_result(self, tool_call_id: str, result: str) -> None:
        await self._send({
            "tool_response": {
                "function_responses": [
                    {
                        "id": tool_call_id,
                        "name": AGENT_CONSULT_TOOL["name"],
                        "response": {"result": result},
                    }
                ]
            }
        })

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                logger.debug("gemini live close failed", exc_info=True)
            self._ws = None

    def _translate(self, frame: Dict[str, Any]) -> list:
        events = []
        content = frame.get("serverContent") or frame.get("server_content") or {}
        if content.get("interrupted"):
            self.response_active = False
            events.append(RealtimeEvent(type="speech_started"))
        model_turn = content.get("modelTurn") or content.get("model_turn") or {}
        if model_turn.get("parts"):
            self.response_active = True
        for part in model_turn.get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data")
            if data:
                try:
                    events.append(
                        RealtimeEvent(type="audio", audio=base64.b64decode(data))
                    )
                except Exception:  # noqa: BLE001
                    continue
            elif part.get("text"):
                events.append(
                    RealtimeEvent(
                        type="transcript", role="assistant", text=str(part["text"])
                    )
                )
        if content.get("turnComplete") or content.get("turn_complete"):
            self.response_active = False
            events.append(RealtimeEvent(type="response_done"))

        tool_call = frame.get("toolCall") or frame.get("tool_call") or {}
        for fc in tool_call.get("functionCalls", tool_call.get("function_calls", [])):
            args = fc.get("args") or {}
            events.append(
                RealtimeEvent(
                    type="tool_call",
                    tool_call_id=str(fc.get("id", "")),
                    tool_name=str(fc.get("name", "")),
                    tool_args=args if isinstance(args, dict) else {},
                )
            )
        return events

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        import websockets

        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                for event in self._translate(frame):
                    yield event
        except websockets.ConnectionClosed:
            pass
        except Exception as e:  # noqa: BLE001
            if not self._closed:
                yield RealtimeEvent(type="error", text=str(e))
        yield RealtimeEvent(type="closed")


def create_realtime_session(config: RealtimeConfig) -> RealtimeVoiceSession:
    """Factory used by the bridge: realtime.provider → session."""
    if config.provider == "openai":
        from .openai_realtime import OpenAIRealtimeSession

        return OpenAIRealtimeSession(config)
    if config.provider == "gemini":
        return GeminiLiveSession(config)
    raise ValueError(f"unknown realtime provider {config.provider!r}")
