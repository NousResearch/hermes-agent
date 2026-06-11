"""xAI Grok Voice Agent realtime provider session."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlencode

from gateway.realtime_voice.config import RealtimeVoiceConfig
from gateway.realtime_voice.tool_bridge import hermes_realtime_tool_definitions
from gateway.realtime_voice.session import (
    RealtimeAudioDelta,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeVoiceEvent,
    RealtimeVoiceSession,
)

logger = logging.getLogger(__name__)

RealtimeEventCallback = Callable[[RealtimeVoiceEvent], Awaitable[None] | None]

_XAI_REALTIME_ENDPOINT = "wss://api.x.ai/v1/realtime"
_XAI_DEFAULT_MODEL = "grok-voice-latest"
_XAI_DEFAULT_VOICE = "ara"
_XAI_DEFAULT_SAMPLE_RATE = 24000
_XAI_BUILTIN_TOOLS = {"web_search", "x_search"}


def _provider_overrides(config: RealtimeVoiceConfig) -> dict[str, Any]:
    providers = getattr(config, "providers", {}) or {}
    raw = providers.get("xai") or providers.get("xai-oauth") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _config_value(config: RealtimeVoiceConfig, key: str, default: Any) -> Any:
    overrides = _provider_overrides(config)
    value = overrides.get(key)
    if value is not None and value != "":
        return value
    value = getattr(config, key, None)
    if value is not None and value != "":
        return value
    return default


def _json_loads_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class XAIRealtimeVoiceSession(RealtimeVoiceSession):
    """Realtime voice session backed by xAI's WebSocket Voice Agent API.

    The class intentionally accepts Hermes xAI OAuth credentials via the shared
    xAI HTTP resolver. That resolver prefers OAuth and falls back to XAI_API_KEY,
    which keeps the voice path aligned with existing xAI TTS/STT integrations.
    """

    def __init__(
        self,
        config: RealtimeVoiceConfig,
        *,
        instructions: str = "",
        on_event: Optional[RealtimeEventCallback] = None,
    ) -> None:
        self.config = config
        self.instructions = instructions
        self.on_event = on_event
        self.model = str(_config_value(config, "model", _XAI_DEFAULT_MODEL)).strip() or _XAI_DEFAULT_MODEL
        self.voice = str(_config_value(config, "voice", _XAI_DEFAULT_VOICE)).strip() or _XAI_DEFAULT_VOICE
        self.sample_rate = int(_config_value(config, "sample_rate", _XAI_DEFAULT_SAMPLE_RATE) or _XAI_DEFAULT_SAMPLE_RATE)
        self.endpoint = str(_config_value(config, "endpoint", _XAI_REALTIME_ENDPOINT)).strip() or _XAI_REALTIME_ENDPOINT
        self.turn_detection = _config_value(
            config,
            "turn_detection",
            {
                "type": "server_vad",
                "threshold": 0.75,
                "silence_duration_ms": 650,
                "prefix_padding_ms": 333,
            },
        )

        self._ws: Any = None
        self._recv_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        """Open the xAI realtime WebSocket and configure the session."""
        if self._ws is not None:
            return
        token, base_url = self._resolve_credentials()
        endpoint = self._endpoint_for_base_url(base_url)
        separator = "&" if "?" in endpoint else "?"
        url = f"{endpoint}{separator}{urlencode({'model': self.model})}"

        try:
            import websockets
        except Exception as exc:  # pragma: no cover - depends on optional dep state
            raise RuntimeError("websockets is required for xAI realtime voice") from exc

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self._user_agent(),
        }
        connect_kwargs: dict[str, Any] = {
            "open_timeout": 20,
            "close_timeout": 5,
            "max_size": 8_000_000,
        }
        signature = inspect.signature(websockets.connect)
        if "additional_headers" in signature.parameters:
            connect_kwargs["additional_headers"] = headers
        else:  # websockets < 14
            connect_kwargs["extra_headers"] = headers

        self._ws = await websockets.connect(url, **connect_kwargs)
        self._closed = False
        await self._send_json(self._session_update_payload())
        self._recv_task = asyncio.create_task(self._receive_loop(), name="xai-realtime-voice-recv")

    async def stop(self) -> None:
        """Close the provider session and release associated resources."""
        self._closed = True
        task = self._recv_task
        self._recv_task = None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("xAI realtime receive task ended with error", exc_info=True)
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                logger.debug("xAI realtime WebSocket close failed", exc_info=True)

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        """Return a tool result to the realtime conversation.

        xAI's realtime API follows the OpenAI-style event vocabulary for custom
        function call outputs. This is best-effort and harmless when no tool call
        was active.
        """
        if self._ws is None or self._closed:
            return
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }
        )
        await self._send_json({"type": "response.create"})

    async def send_audio_pcm16(self, data: bytes, sample_rate: int) -> None:
        """Append PCM16 mono input audio to the xAI input buffer."""
        if not data or self._ws is None or self._closed:
            return
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"xAI realtime session expects {self.sample_rate}Hz PCM16 mono; got {sample_rate}Hz"
            )
        await self._send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(data).decode("ascii"),
            }
        )

    async def interrupt(self) -> None:
        """Cancel current provider output after user barge-in."""
        if self._ws is None or self._closed:
            return
        try:
            await self._send_json({"type": "response.cancel"})
        except Exception:
            logger.debug("xAI realtime interrupt failed", exc_info=True)

    async def update_instructions(self, instructions: str) -> None:
        """Update provider session instructions without reconnecting."""
        self.instructions = instructions
        if self._ws is None or self._closed:
            return
        await self._send_json(self._session_update_payload())

    async def send_text_turn(self, text: str) -> None:
        """Send a text turn and explicitly request a response.

        Useful for smoke tests and for future non-audio command bridges.
        """
        clean = text.strip()
        if not clean:
            return
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": clean}],
                },
            }
        )
        await self._send_json({"type": "response.create"})

    async def _send_json(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise RuntimeError("xAI realtime session is not connected")
        async with self._send_lock:
            await ws.send(json.dumps(payload, separators=(",", ":")))

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("xAI realtime ignored non-JSON frame")
                continue
            await self._handle_server_event(event)

    async def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "response.output_audio.delta":
            encoded = str(event.get("delta") or "")
            if encoded:
                try:
                    pcm16 = base64.b64decode(encoded)
                except Exception:
                    logger.debug("xAI realtime audio delta was not valid base64")
                    return
                await self._emit(RealtimeAudioDelta(pcm16=pcm16, sample_rate=self.sample_rate))
            return

        if event_type == "response.output_audio_transcript.delta":
            text = str(event.get("delta") or "")
            if text:
                await self._emit(RealtimeTranscriptDelta(role="assistant", text=text, final=False))
            return

        if event_type == "response.output_audio_transcript.done":
            text = str(event.get("transcript") or "")
            if text:
                await self._emit(RealtimeTranscriptDelta(role="assistant", text=text, final=True))
            return

        if event_type == "response.text.delta":
            text = str(event.get("delta") or "")
            if text:
                await self._emit(RealtimeTranscriptDelta(role="assistant", text=text, final=False))
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            text = str(event.get("transcript") or "")
            if text:
                await self._emit(RealtimeTranscriptDelta(role="user", text=text, final=True))
            return

        if event_type == "response.function_call_arguments.done":
            await self._emit(
                RealtimeToolCall(
                    name=str(event.get("name") or ""),
                    arguments=_json_loads_object(event.get("arguments")),
                    call_id=str(event.get("call_id") or ""),
                )
            )
            return

        if event_type == "error":
            raw_error = event.get("error")
            error = raw_error if isinstance(raw_error, dict) else {}
            logger.warning(
                "xAI realtime server error: %s: %s",
                error.get("code") or "unknown",
                error.get("message") or "",
            )

    async def _emit(self, event: RealtimeVoiceEvent) -> None:
        callback = self.on_event
        if callback is None:
            return
        result = callback(event)
        if inspect.isawaitable(result):
            await result

    def _session_update_payload(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "model": self.model,
            "voice": self.voice,
            "instructions": self.instructions,
            "turn_detection": self.turn_detection,
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                "output": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
            },
        }
        tools = self._tool_definitions()
        if tools:
            session["tools"] = tools
        return {"type": "session.update", "session": session}

    def _tool_definitions(self) -> list[dict[str, Any]]:
        """Return xAI realtime built-in and Hermes custom tools enabled by config."""
        tools: list[dict[str, Any]] = []
        for definition in hermes_realtime_tool_definitions(getattr(self.config, "allow_tools", ())):
            tools.append(
                {
                    "type": "function",
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                }
            )
        for tool_name in getattr(self.config, "allow_tools", ()) or ():
            normalized = str(tool_name).strip().lower()
            if normalized in _XAI_BUILTIN_TOOLS:
                tools.append({"type": normalized})
        return tools

    def _resolve_credentials(self) -> tuple[str, str]:
        from tools.xai_http import resolve_xai_http_credentials

        creds = resolve_xai_http_credentials()
        token = str(creds.get("api_key") or "").strip()
        if not token:
            raise RuntimeError("No xAI credentials found. Configure xAI OAuth or set XAI_API_KEY.")
        base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")
        return token, base_url

    def _endpoint_for_base_url(self, base_url: str) -> str:
        if self.endpoint != _XAI_REALTIME_ENDPOINT:
            return self.endpoint
        clean = (base_url or "https://api.x.ai/v1").rstrip("/")
        if clean.startswith("https://"):
            clean = "wss://" + clean[len("https://"):]
        elif clean.startswith("http://"):
            clean = "ws://" + clean[len("http://"):]
        if clean.endswith("/v1"):
            return f"{clean}/realtime"
        return f"{clean}/v1/realtime"

    @staticmethod
    def _user_agent() -> str:
        try:
            from tools.xai_http import hermes_xai_user_agent

            return hermes_xai_user_agent()
        except Exception:
            return "Hermes-Agent/unknown"
