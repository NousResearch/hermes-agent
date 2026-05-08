"""OpenAI Realtime voice session for server-side Discord voice streaming."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, cast

import aiohttp

from gateway.discord_realtime_audio import discord_pcm_to_realtime_pcm

logger = logging.getLogger(__name__)


class _RealtimeWebSocket(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> Any: ...


WebSocketFactory = Callable[..., Awaitable[_RealtimeWebSocket]]


def normalize_realtime_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Hermes/OpenAI Chat tool schemas to Realtime session tools.

    Hermes' model tool registry returns Chat Completions style schemas:
    ``{"type": "function", "function": {"name": ..., "parameters": ...}}``.
    The Realtime API expects the flattened shape in ``session.tools``:
    ``{"type": "function", "name": ..., "parameters": ...}``.
    Already-flattened tool schemas pass through unchanged.
    """
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue

        function = tool.get("function")
        if tool.get("type") == "function" and isinstance(function, dict):
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            realtime_tool: dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
            for key in ("description", "strict"):
                if key in function:
                    realtime_tool[key] = function[key]
            normalized.append(realtime_tool)
            continue

        name = tool.get("name")
        if tool.get("type") == "function" and isinstance(name, str) and name:
            realtime_tool = dict(tool)
            realtime_tool.setdefault("parameters", {"type": "object", "properties": {}})
            normalized.append(realtime_tool)

    return normalized


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
    audit_log_path: str | None = None
    audit_include_text: bool = True


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
        self._audit("session.start", {
            "model": self.config.model,
            "voice": self.config.voice,
            "turn_detection": self.config.turn_detection,
            "tools_enabled": self.config.tools_enabled,
            "instructions_chars": len(self.config.instructions or ""),
        })
        self._ws = await self._connect()
        session_update = self._session_update_event()
        self._audit("client.session_update", self._summarize_payload(session_update))
        await self._send_json(session_update)
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
        self._audit("client.input_audio_buffer.append", {
            "user_id": str(user_id),
            "discord_pcm_bytes": len(pcm_48k_stereo),
            "realtime_pcm_bytes": len(pcm_24k_mono),
        })
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
        tools = (
            normalize_realtime_tool_schemas(self.config.tools)
            if self.config.tools_enabled
            else []
        )
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
        self._audit("server.event", self._summarize_payload(event))
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

        if event_type == "response.function_call_arguments.done":
            item = {
                "type": "function_call",
                "name": event.get("name"),
                "call_id": event.get("call_id"),
                "arguments": event.get("arguments") or "{}",
            }
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

        try:
            output = await self._tool_executor(name, arguments)
        except Exception as exc:
            self._logger.exception("Realtime tool call %s failed", name)
            output = json.dumps({"error": f"Error executing {name}: {exc}"}, ensure_ascii=False)
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

    def _summarize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a safe, compact audit summary for a Realtime event.

        Raw base64 audio and API keys are never logged. Text is included only
        when ``audit_include_text`` is enabled because transcripts can be
        private.
        """
        event_type = str(payload.get("type", ""))
        summary: dict[str, Any] = {"type": event_type}
        if "event_id" in payload:
            summary["event_id"] = payload.get("event_id")

        delta = payload.get("delta")
        if isinstance(delta, str):
            if "audio" in event_type:
                summary["audio_delta_b64_chars"] = len(delta)
                try:
                    summary["audio_delta_bytes"] = len(base64.b64decode(delta))
                except Exception:
                    pass
            elif self.config.audit_include_text:
                summary["delta"] = delta
                summary["delta_chars"] = len(delta)
            else:
                summary["delta_chars"] = len(delta)

        transcript = payload.get("transcript")
        if isinstance(transcript, str):
            if self.config.audit_include_text:
                summary["transcript"] = transcript
            summary["transcript_chars"] = len(transcript)

        audio = payload.get("audio")
        if isinstance(audio, str):
            summary["audio_b64_chars"] = len(audio)
            try:
                summary["audio_bytes"] = len(base64.b64decode(audio))
            except Exception:
                pass

        session = payload.get("session")
        if isinstance(session, dict):
            summary["session"] = {
                "model": session.get("model"),
                "output_modalities": session.get("output_modalities"),
                "instructions_chars": len(str(session.get("instructions") or "")),
                "tools_count": len(session.get("tools") or []),
                "tool_choice": session.get("tool_choice"),
            }

        item = payload.get("item")
        if isinstance(item, dict):
            summary["item"] = {
                "type": item.get("type"),
                "name": item.get("name"),
                "call_id": item.get("call_id"),
            }
            if self.config.audit_include_text:
                text = item.get("text") or item.get("transcript")
                if isinstance(text, str):
                    summary["item"]["text"] = text
                    summary["item"]["text_chars"] = len(text)

        if event_type == "error":
            summary["error"] = payload.get("error")
        return summary

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        path = self.config.audit_log_path
        if not path:
            return
        try:
            audit_path = Path(path).expanduser()
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **payload,
            }
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            self._logger.debug("Realtime voice audit logging failed: %s", exc)
