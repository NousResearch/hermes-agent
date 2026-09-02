"""OpenAI Realtime (GA) speech-to-speech provider over WebSocket.

The first built-in :class:`agent.realtime_voice_provider.RealtimeVoiceProvider`.
It speaks the GA Realtime wire (``session.update`` with ``audio.input`` /
``audio.output`` blocks, ``response.output_audio.delta``,
``response.function_call_arguments.done``, ...) directly over an API-key
WebSocket — no ephemeral client secret, no browser relay — using the
``websockets`` package Hermes already depends on.

Everything the wire cannot do is declared, not faked: the capability set
below is exactly what the orchestrator may call. The API key is resolved
through the same profile-scoped helper the built-in OpenAI TTS/STT use,
travels only in the ``Authorization`` header, and is never logged.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from agent.realtime_voice_provider import (
    PCM16_24K,
    InputAudioCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeAudioFormat,
    RealtimeCapability,
    RealtimeTool,
    RealtimeToolResult,
    RealtimeVoiceEvent,
    RealtimeVoiceProvider,
    RealtimeVoiceSession,
    RealtimeVoiceSetup,
    ResponseCompleted,
    ResponseStarted,
    SessionClosed,
    SessionFailure,
    SessionReady,
    ToolCall,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai"
REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2.1"
DEFAULT_VOICE = "marin"
INPUT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
CONNECT_TIMEOUT_S = 30.0
PING_INTERVAL_S = 20.0
#: One audio delta is ~0.1 s of pcm16 (about 6 KiB base64); tool-call
#: arguments and transcripts are the only other large frames.
MAX_FRAME_BYTES = 16 * 1024 * 1024

_MODELS: tuple[Mapping[str, str], ...] = (
    {"id": "gpt-realtime-2.1", "display": "GPT Realtime 2.1"},
    {"id": "gpt-realtime-2.1-mini", "display": "GPT Realtime 2.1 Mini"},
    {"id": "gpt-realtime", "display": "GPT Realtime"},
    {"id": "gpt-realtime-mini", "display": "GPT Realtime Mini"},
)
_VOICES: tuple[str, ...] = (
    "marin",
    "cedar",
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
)

CAPABILITIES = frozenset(
    {
        RealtimeCapability.TOOL_CALLING,
        RealtimeCapability.INPUT_TRANSCRIPTION,
        RealtimeCapability.OUTPUT_TRANSCRIPTION,
        RealtimeCapability.INPUT_COMMIT_EVENTS,
        RealtimeCapability.EXPLICIT_RESPONSE,
        RealtimeCapability.RESPONSE_CANCELLATION,
        RealtimeCapability.OUTPUT_TRUNCATION,
        RealtimeCapability.DYNAMIC_CONTEXT,
    }
)

#: ``connector(url, headers)`` returns a connected socket exposing
#: ``send(str)``, ``close()``, and async iteration over inbound frames —
#: the subset of :class:`websockets.asyncio.client.ClientConnection` the
#: session uses. Injected by tests; the default dials the real endpoint.
WebSocketConnector = Callable[[str, Mapping[str, str]], Awaitable[Any]]


async def _default_connector(url: str, headers: Mapping[str, str]) -> Any:
    import websockets

    return await websockets.connect(
        url,
        additional_headers=dict(headers),
        open_timeout=CONNECT_TIMEOUT_S,
        ping_interval=PING_INTERVAL_S,
        max_size=MAX_FRAME_BYTES,
    )


def _resolve_api_key() -> str:
    """Profile-scoped OpenAI key (VOICE_TOOLS_OPENAI_KEY, then OPENAI_API_KEY)."""
    from tools.tool_backend_helpers import resolve_openai_audio_api_key

    return (resolve_openai_audio_api_key() or "").strip()


# -- wire encoding -----------------------------------------------------------


def _tool_wire(tool: RealtimeTool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": _plain(tool.parameters),
    }


def _plain(value: Any) -> Any:
    """Turn the contract's frozen mappings/tuples back into JSON-able objects."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        return [_plain(item) for item in value]
    return value


def build_session_update(setup: RealtimeVoiceSetup, *, voice: str) -> dict[str, Any]:
    """The GA ``session.update`` for one setup: server VAD, barge-in, transcripts.

    The model rides the connection URL and cannot change afterwards, so it
    is deliberately absent here.
    """
    session: dict[str, Any] = {
        "type": "realtime",
        "instructions": setup.instructions,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": PCM16_24K.sample_rate_hz},
                "noise_reduction": {"type": "near_field"},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": setup.automatic_response,
                    "interrupt_response": True,
                },
                "transcription": {"model": INPUT_TRANSCRIPTION_MODEL},
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": PCM16_24K.sample_rate_hz},
                "voice": voice,
            },
        },
    }
    if setup.tools:
        session["tools"] = [_tool_wire(tool) for tool in setup.tools]
        session["tool_choice"] = "auto"
    return {"type": "session.update", "session": session}


# -- wire decoding -----------------------------------------------------------


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_ms(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def _error_detail(error: Any) -> tuple[str, str]:
    if isinstance(error, dict):
        code = _identifier(error.get("code")) or _identifier(error.get("type")) or "error"
        message = str(error.get("message") or error.get("type") or "")
        return code, message
    return "error", str(error or "")


def decode_event(event: dict[str, Any]) -> RealtimeVoiceEvent | None:
    """Map one GA server event to a contract event; ``None`` = nothing to say.

    A malformed identifier from the wire becomes a non-terminal failure
    instead of a crash: one bad frame must never end a conversation.
    """
    event_type = str(event.get("type") or "")
    try:
        return _decode(event_type, event)
    except ValueError as exc:
        return SessionFailure(
            code="protocol",
            message=f"provider sent a malformed {event_type or 'event'}: {exc}",
            terminal=False,
        )


def _decode(event_type: str, event: dict[str, Any]) -> RealtimeVoiceEvent | None:
    if event_type == "session.created":
        session_id = _identifier(_mapping(event.get("session")).get("id")) or "session"
        return SessionReady(session_id=session_id)
    if event_type == "input_audio_buffer.speech_started":
        return InputSpeechStarted(
            item_id=_identifier(event.get("item_id")),
            audio_start_ms=_optional_ms(event.get("audio_start_ms")),
        )
    if event_type == "input_audio_buffer.speech_stopped":
        return InputSpeechStopped(
            item_id=_identifier(event.get("item_id")),
            audio_end_ms=_optional_ms(event.get("audio_end_ms")),
        )
    if event_type == "input_audio_buffer.committed":
        return InputAudioCommitted(item_id=_identifier(event.get("item_id")))
    if event_type == "response.created":
        response = _mapping(event.get("response"))
        metadata = {
            str(key): str(value) for key, value in _mapping(response.get("metadata")).items()
        }
        return ResponseStarted(response_id=_identifier(response.get("id")), metadata=metadata)
    if event_type == "response.output_audio.delta":
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return SessionFailure(
                code="audio", message="provider sent an empty audio delta", terminal=False
            )
        try:
            pcm = base64.b64decode(delta, validate=True)
        except (binascii.Error, ValueError):
            return SessionFailure(
                code="audio", message="provider sent a malformed audio delta", terminal=False
            )
        return OutputAudio(
            data=pcm,
            item_id=_identifier(event.get("item_id")),
            response_id=_identifier(event.get("response_id")),
        )
    if event_type in ("response.output_audio_transcript.delta", "response.output_text.delta"):
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return None
        return OutputTranscript(
            text=delta,
            final=False,
            item_id=_identifier(event.get("item_id")),
            response_id=_identifier(event.get("response_id")),
        )
    if event_type in ("response.output_audio_transcript.done", "response.output_text.done"):
        completed = event.get("transcript") if "transcript" in event else event.get("text")
        return OutputTranscript(
            text=completed if isinstance(completed, str) else "",
            final=True,
            item_id=_identifier(event.get("item_id")),
            response_id=_identifier(event.get("response_id")),
        )
    if event_type == "conversation.item.input_audio_transcription.delta":
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return None
        return InputTranscript(text=delta, final=False, item_id=_identifier(event.get("item_id")))
    if event_type == "conversation.item.input_audio_transcription.completed":
        transcript = event.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            return None
        return InputTranscript(
            text=transcript.strip(), final=True, item_id=_identifier(event.get("item_id"))
        )
    if event_type == "response.function_call_arguments.done":
        arguments = event.get("arguments")
        return ToolCall(
            call_id=event.get("call_id"),
            name=event.get("name"),
            arguments=arguments if isinstance(arguments, str) else "",
            response_id=_identifier(event.get("response_id")),
            item_id=_identifier(event.get("item_id")),
        )
    if event_type == "response.done":
        response = _mapping(event.get("response"))
        return ResponseCompleted(
            response_id=_identifier(response.get("id")),
            status=str(response.get("status") or ""),
        )
    if event_type == "error":
        code, message = _error_detail(event.get("error"))
        if "no active response" in message.lower():
            # A cancel that lost the race with response completion: the
            # response is already over, which is what the cancel wanted.
            return None
        return SessionFailure(
            code=code, message=message or "provider reported a session error", terminal=False
        )
    return None


# -- session -----------------------------------------------------------------


class OpenAIRealtimeSession(RealtimeVoiceSession):
    """One connected GA Realtime WebSocket, translated to contract events."""

    def __init__(self, socket: Any) -> None:
        super().__init__(CAPABILITIES, input_audio=PCM16_24K, output_audio=PCM16_24K)
        self._socket = socket
        self._send_lock = asyncio.Lock()

    async def _send(self, *messages: dict[str, Any]) -> None:
        """Write messages contiguously; a batch is never interleaved."""
        async with self._send_lock:
            for message in messages:
                await self._socket.send(json.dumps(message))

    async def send_audio(self, audio: bytes) -> None:
        if not audio:
            return
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(bytes(audio)).decode("ascii"),
            }
        )

    async def _commit_audio(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})

    async def _create_response(self, metadata: Mapping[str, str]) -> None:
        message: dict[str, Any] = {"type": "response.create"}
        if metadata:
            message["response"] = {"metadata": dict(metadata)}
        await self._send(message)

    async def _cancel_response(self, response_id: str | None) -> None:
        message: dict[str, Any] = {"type": "response.cancel"}
        if response_id is not None:
            message["response_id"] = response_id
        await self._send(message)

    async def _truncate_output(self, item_id: str, audio_end_ms: int) -> None:
        await self._send(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": audio_end_ms,
            }
        )

    async def _add_context(self, item_id: str, text: str) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def _remove_context(self, item_id: str) -> None:
        await self._send({"type": "conversation.item.delete", "item_id": item_id})

    async def _submit_tool_results(
        self, results: tuple[RealtimeToolResult, ...], continue_response: bool
    ) -> None:
        messages: list[dict[str, Any]] = [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.output,
                },
            }
            for result in results
        ]
        if continue_response:
            messages.append({"type": "response.create"})
        await self._send(*messages)

    def _events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[RealtimeVoiceEvent]:
        from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

        try:
            async for raw in self._socket:
                event = self._decode_frame(raw)
                if event is not None:
                    yield event
        except ConnectionClosedOK as exc:
            yield SessionClosed(reason=_close_reason(exc))
            return
        except ConnectionClosed as exc:
            yield SessionFailure(
                code="connection_closed",
                message=f"provider closed the connection abnormally ({_close_reason(exc)})",
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced as the terminal event
            logger.warning("OpenAI realtime receive failed", exc_info=True)
            yield SessionFailure(
                code="transport", message=f"{type(exc).__name__}: {exc}"
            )
            return
        yield SessionClosed(reason="end of stream")

    @staticmethod
    def _decode_frame(raw: Any) -> RealtimeVoiceEvent | None:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            try:
                raw = bytes(raw).decode("utf-8")
            except UnicodeDecodeError:
                return SessionFailure(
                    code="protocol", message="provider sent a non-UTF-8 frame", terminal=False
                )
        try:
            wire_event = json.loads(raw)
        except (TypeError, ValueError):
            return SessionFailure(
                code="protocol", message="provider sent malformed JSON", terminal=False
            )
        if not isinstance(wire_event, dict):
            return SessionFailure(
                code="protocol", message="provider sent a non-object event", terminal=False
            )
        return decode_event(wire_event)

    async def _close(self) -> None:
        await self._socket.close()


def _close_reason(exc: Any) -> str:
    frame = getattr(exc, "rcvd", None) or getattr(exc, "sent", None)
    code = getattr(frame, "code", None)
    reason = getattr(frame, "reason", "") or ""
    if code is None:
        return "no close frame"
    return f"code {code}" + (f": {reason}" if reason else "")


# -- provider ----------------------------------------------------------------


class OpenAIRealtimeProvider(RealtimeVoiceProvider):
    capabilities = CAPABILITIES

    def __init__(self, *, connector: WebSocketConnector | None = None) -> None:
        self._connector = connector or _default_connector

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "OpenAI Realtime"

    def is_available(self) -> bool:
        try:
            return bool(_resolve_api_key())
        except Exception:  # noqa: BLE001 — availability probes never raise
            logger.debug("OpenAI realtime availability probe failed", exc_info=True)
            return False

    def list_models(self) -> tuple[Mapping[str, Any], ...]:
        return _MODELS

    def list_voices(self) -> tuple[Mapping[str, Any], ...]:
        return tuple({"id": voice, "display": voice.title()} for voice in _VOICES)

    def get_setup_schema(self) -> Mapping[str, Any]:
        return {
            "name": self.display_name,
            "badge": "paid",
            "tag": "gpt-realtime speech-to-speech",
            "env_vars": (
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key",
                    "url": "https://platform.openai.com/api-keys",
                },
            ),
        }

    async def open_session(self, setup: RealtimeVoiceSetup) -> RealtimeVoiceSession:
        for label, requested in (("input", setup.input_audio), ("output", setup.output_audio)):
            if requested is not None and requested != PCM16_24K:
                raise ValueError(
                    f"OpenAI Realtime {label} audio must be {_describe(PCM16_24K)}, "
                    f"got {_describe(requested)}"
                )
        api_key = _resolve_api_key()
        if not api_key:
            raise RuntimeError(
                "OpenAI Realtime needs an API key: set OPENAI_API_KEY (or "
                "VOICE_TOOLS_OPENAI_KEY), or run `hermes auth add openai-api`."
            )
        model = setup.model or DEFAULT_MODEL
        voice = setup.voice or DEFAULT_VOICE
        headers = {"Authorization": f"Bearer {api_key}"}
        del api_key
        socket = await self._connector(f"{REALTIME_WS_URL}?model={model}", headers)
        session = OpenAIRealtimeSession(socket)
        try:
            await session._send(build_session_update(setup, voice=voice))
        except BaseException:
            await session.close()
            raise
        return session


def _describe(audio: RealtimeAudioFormat) -> str:
    return f"{audio.mime_type} {audio.sample_rate_hz} Hz x{audio.channels}"


__all__ = [
    "CAPABILITIES",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "PROVIDER_NAME",
    "REALTIME_WS_URL",
    "OpenAIRealtimeProvider",
    "OpenAIRealtimeSession",
    "build_session_update",
    "decode_event",
]
