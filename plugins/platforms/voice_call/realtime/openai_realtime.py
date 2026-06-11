"""OpenAI Realtime API session (speech-to-speech over websockets).

Async sibling of ``plugins/google_meet/realtime/openai_client.py`` —
duplex instead of text-to-speech-only: caller PCM16 streams in through
``input_audio_buffer.append``, server VAD detects turns, audio deltas
stream back out, and ``agent_consult`` tool calls round-trip to the full
Hermes agent. PCM16 @ 24 kHz both directions.
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

REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime"
DEFAULT_VOICE = "marin"


class OpenAIRealtimeSession(RealtimeVoiceSession):
    name = "openai"
    input_sample_rate = 24000
    output_sample_rate = 24000

    def __init__(self, config: RealtimeConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI Realtime requires OPENAI_API_KEY")
        self.model = config.model or DEFAULT_MODEL
        self.voice = config.voice or DEFAULT_VOICE
        self.instructions = config.instructions or DEFAULT_INSTRUCTIONS
        self._ws = None
        self._closed = False
        # Tool-call argument accumulation: call_id → {"name", "args"}
        self._pending_tools: Dict[str, Dict[str, str]] = {}

    def _session_update(self) -> Dict[str, Any]:
        """GA realtime session shape (no OpenAI-Beta header, nested audio
        config). Accounts on the GA API reject the 2024 beta shape with
        ``beta_api_shape_disabled``."""
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "output_modalities": ["audio"],
                "instructions": self.instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
                        "turn_detection": {"type": "server_vad"},
                        "transcription": {"model": "whisper-1"},
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm", "rate": self.output_sample_rate,
                        },
                        "voice": self.voice,
                    },
                },
                "tools": [{"type": "function", **AGENT_CONSULT_TOOL}],
                "tool_choice": "auto",
            },
        }

    async def connect(self) -> None:
        import websockets

        self._ws = await websockets.connect(
            f"{REALTIME_URL}?model={self.model}",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=16 * 1024 * 1024,
        )
        await self._send(self._session_update())

    async def _send(self, message: Dict[str, Any]) -> None:
        if self._ws is None or self._closed:
            return
        await self._ws.send(json.dumps(message))

    async def send_audio(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode(),
        })

    async def inject_text(self, text: str) -> None:
        await self._send({
            "type": "response.create",
            "response": {"instructions": f"Say this to the caller now: {text}"},
        })

    async def cancel_response(self) -> None:
        await self._send({"type": "response.cancel"})

    async def send_tool_result(self, tool_call_id: str, result: str) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": result,
            },
        })
        await self._send({"type": "response.create"})

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                logger.debug("openai realtime close failed", exc_info=True)
            self._ws = None

    def _translate(self, frame: Dict[str, Any]) -> Optional[RealtimeEvent]:
        ftype = str(frame.get("type", ""))
        # Audio deltas (the API renamed response.audio.* → response.output_audio.*).
        if ftype in ("response.audio.delta", "response.output_audio.delta"):
            try:
                audio = base64.b64decode(frame.get("delta") or "")
            except Exception:  # noqa: BLE001
                return None
            return RealtimeEvent(type="audio", audio=audio)
        if ftype == "input_audio_buffer.speech_started":
            return RealtimeEvent(type="speech_started")
        if ftype == "conversation.item.input_audio_transcription.completed":
            return RealtimeEvent(
                type="transcript", role="user", text=str(frame.get("transcript", ""))
            )
        if ftype in (
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            return RealtimeEvent(
                type="transcript", role="assistant",
                text=str(frame.get("transcript", "")),
            )
        if ftype == "response.function_call_arguments.done":
            call_id = str(frame.get("call_id", ""))
            try:
                args = json.loads(frame.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            return RealtimeEvent(
                type="tool_call",
                tool_call_id=call_id,
                tool_name=str(frame.get("name") or "agent_consult"),
                tool_args=args if isinstance(args, dict) else {},
            )
        if ftype in ("response.done", "response.completed", "response.cancelled"):
            return RealtimeEvent(type="response_done")
        if ftype == "error":
            detail = (frame.get("error") or {}).get("message", "unknown error")
            return RealtimeEvent(type="error", text=str(detail))
        return None

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        import websockets

        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                event = self._translate(frame)
                if event is not None:
                    yield event
        except websockets.ConnectionClosed:
            pass
        except Exception as e:  # noqa: BLE001
            if not self._closed:
                yield RealtimeEvent(type="error", text=str(e))
        yield RealtimeEvent(type="closed")
