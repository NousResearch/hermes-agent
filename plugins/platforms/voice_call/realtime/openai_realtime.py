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
    input_sample_rate = 8000
    output_sample_rate = 8000
    # The GA realtime API speaks G.711 µ-law natively (audio/pcmu @ 8 kHz)
    # — the phone line's own codec — so the bridge passes carrier frames
    # straight through with no transcoding.
    audio_wire_format = "ulaw"

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
        # call_ids already surfaced as tool_call events — GA can deliver a
        # function call both as function_call_arguments.done and inside
        # response.done output items.
        self._seen_tool_calls: set = set()

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
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": {"type": "server_vad"},
                        "transcription": {"model": "whisper-1"},
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
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
        # Optimistic: the server's response.created event lags our create;
        # without this, a tool result landing in that window would issue a
        # colliding response.create ("already has an active response").
        self.response_active = True

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
        self.response_active = True  # optimistic — see inject_text

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                logger.debug("openai realtime close failed", exc_info=True)
            self._ws = None

    def _tool_call_event(
        self, call_id: str, name: str, arguments: Any
    ) -> Optional[RealtimeEvent]:
        if not call_id or call_id in self._seen_tool_calls:
            return None
        self._seen_tool_calls.add(call_id)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
        return RealtimeEvent(
            type="tool_call",
            tool_call_id=call_id,
            tool_name=name or "agent_consult",
            tool_args=arguments if isinstance(arguments, dict) else {},
        )

    def _translate(self, frame: Dict[str, Any]) -> list:
        ftype = str(frame.get("type", ""))
        if ftype == "response.created":
            self.response_active = True
            return []
        # Audio deltas (the API renamed response.audio.* → response.output_audio.*).
        if ftype in ("response.audio.delta", "response.output_audio.delta"):
            self.response_active = True
            try:
                audio = base64.b64decode(frame.get("delta") or "")
            except Exception:  # noqa: BLE001
                return []
            return [RealtimeEvent(type="audio", audio=audio)]
        if ftype == "input_audio_buffer.speech_started":
            return [RealtimeEvent(type="speech_started")]
        if ftype == "conversation.item.input_audio_transcription.completed":
            return [RealtimeEvent(
                type="transcript", role="user", text=str(frame.get("transcript", ""))
            )]
        if ftype in (
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            return [RealtimeEvent(
                type="transcript", role="assistant",
                text=str(frame.get("transcript", "")),
            )]
        if ftype == "response.function_call_arguments.done":
            event = self._tool_call_event(
                str(frame.get("call_id", "")),
                str(frame.get("name") or ""),
                frame.get("arguments"),
            )
            return [event] if event else []
        if ftype in ("response.done", "response.completed", "response.cancelled"):
            self.response_active = False
            # GA delivers function calls as response.done output items (the
            # standalone arguments.done event is not guaranteed) — scan and
            # dedupe against ones already surfaced.
            events = []
            output = (frame.get("response") or {}).get("output") or []
            for item in output:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    event = self._tool_call_event(
                        str(item.get("call_id", "")),
                        str(item.get("name") or ""),
                        item.get("arguments"),
                    )
                    if event:
                        events.append(event)
            events.append(RealtimeEvent(type="response_done"))
            return events
        if ftype == "error":
            detail = (frame.get("error") or {}).get("message", "unknown error")
            return [RealtimeEvent(type="error", text=str(detail))]
        return []

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
                for event in self._translate(frame):
                    yield event
        except websockets.ConnectionClosed:
            pass
        except Exception as e:  # noqa: BLE001
            if not self._closed:
                yield RealtimeEvent(type="error", text=str(e))
        yield RealtimeEvent(type="closed")
