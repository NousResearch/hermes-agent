"""OpenAI Realtime voice session for server-side Discord voice streaming."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, cast

import aiohttp

from gateway.discord_realtime_audio import discord_pcm_to_realtime_pcm

logger = logging.getLogger(__name__)


class _RealtimeWebSocket(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> Any: ...


WebSocketFactory = Callable[..., Awaitable[_RealtimeWebSocket]]


@dataclass
class RealtimeVoiceConfig:
    api_key: str
    model: str = "gpt-realtime-2"
    voice: str = "marin"
    instructions: str = ""
    safety_identifier: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tools_enabled: bool = False
    input_rate: int = 24_000
    output_rate: int = 24_000
    turn_detection: str = "semantic_vad"


class RealtimeVoiceSession:
    def __init__(
        self,
        *,
        config: RealtimeVoiceConfig,
        audio_sink: Callable[[bytes], None],
        on_transcript: Callable[[str, str], Awaitable[None]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
        logger: logging.Logger | None = None,
        websocket_factory: WebSocketFactory | None = None,
        on_barge_in: Callable[[], None] | None = None,
    ):
        self.config = config
        self._audio_sink = audio_sink
        self._on_transcript = on_transcript
        self._tool_executor = tool_executor
        self._logger = logger or logging.getLogger(__name__)
        self._websocket_factory = websocket_factory
        self._on_barge_in = on_barge_in
        self._ws: _RealtimeWebSocket | None = None
        self._client_session: aiohttp.ClientSession | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._ws is not None:
            return
        self._ws = await self._connect()
        await self._send_json(self._session_update_event())
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def stop(self) -> None:
        task = self._receiver_task
        self._receiver_task = None
        if task:
            task.cancel()
        ws = self._ws
        self._ws = None
        if ws:
            try:
                await ws.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup.
                self._logger.debug("Realtime voice WebSocket close failed: %s", exc)
        if task:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - defensive cleanup.
                self._logger.debug("Realtime voice receiver task ended with error: %s", exc)
        client_session = self._client_session
        self._client_session = None
        if client_session:
            await client_session.close()

    async def append_discord_pcm(self, user_id: int, pcm_48k_stereo: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime voice session is not started")
        pcm_24k_mono = discord_pcm_to_realtime_pcm(pcm_48k_stereo)
        if not pcm_24k_mono:
            return
        await self._send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm_24k_mono).decode("ascii"),
        })

    async def _connect(self) -> _RealtimeWebSocket:
        url = f"wss://api.openai.com/v1/realtime?model={self.config.model}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
        }
        if self.config.safety_identifier:
            headers["OpenAI-Safety-Identifier"] = self.config.safety_identifier
        if self._websocket_factory is not None:
            return await self._websocket_factory(url, headers=headers)
        self._client_session = aiohttp.ClientSession()
        return cast(_RealtimeWebSocket, await self._client_session.ws_connect(url, headers=headers))

    def _session_update_event(self) -> dict[str, Any]:
        tools = self.config.tools if self.config.tools_enabled else []
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.config.model,
                "output_modalities": ["audio"],
                "instructions": self.config.instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": self.config.input_rate},
                        "turn_detection": {
                            "type": self.config.turn_detection,
                            "interrupt_response": True,
                            "create_response": True,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": self.config.output_rate},
                        "voice": self.config.voice,
                    },
                },
                "tools": tools,
                "tool_choice": "auto",
            },
        }

    async def _receive_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for msg in ws:
                msg_type = getattr(msg, "type", None)
                if msg_type == aiohttp.WSMsgType.TEXT:
                    await self._handle_server_event(json.loads(msg.data))
                elif msg_type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.warning("Realtime voice receive loop failed: %s", exc)

    async def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type in {"response.output_audio.delta", "response.audio.delta"}:
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                self._audio_sink(base64.b64decode(delta))
            return

        if event_type == "input_audio_buffer.speech_started":
            self._clear_audio_sink()
            return

        if event_type in {
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
        }:
            if self._on_transcript is not None:
                text = event.get("delta") or event.get("transcript") or ""
                if isinstance(text, str) and text:
                    await self._on_transcript("assistant", text)
            return

        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                await self._handle_output_item_done(item)
            return

        if event_type == "error":
            error = event.get("error")
            self._logger.warning("OpenAI Realtime voice error: %s", error)

    async def _handle_output_item_done(self, item: dict[str, Any]) -> None:
        if not self.config.tools_enabled:
            return
        if item.get("type") != "function_call":
            return
        if self._tool_executor is None:
            self._logger.warning("Realtime tool call received but no executor is configured")
            return

        name = item.get("name")
        call_id = item.get("call_id")
        if not isinstance(name, str) or not isinstance(call_id, str):
            self._logger.warning("Realtime function call missing name or call_id")
            return

        raw_arguments = item.get("arguments") or "{}"
        arguments: dict[str, Any]
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed = {}
            arguments = parsed if isinstance(parsed, dict) else {}
        else:
            arguments = {}

        output = await self._tool_executor(name, arguments)
        await self._send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        })
        await self._send_json({"type": "response.create"})

    def _clear_audio_sink(self) -> None:
        if self._on_barge_in is not None:
            self._on_barge_in()
            return
        clear = getattr(self._audio_sink, "clear", None)
        if callable(clear):
            clear()

    async def _send_json(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise RuntimeError("Realtime voice session is not started")
        async with self._send_lock:
            await ws.send_json(payload)
