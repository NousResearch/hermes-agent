"""
ESP device bridge adapter.

This adapter exposes a small WebSocket server so dedicated hardware devices
can talk to Hermes through the normal gateway/session pipeline.

V1 scope:
- text input
- final audio payload upload
- text responses
- status updates
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from aiohttp import web, WSMsgType
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    WSMsgType = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)


def check_esp_requirements() -> bool:
    """Check if ESP bridge dependencies are available."""
    return AIOHTTP_AVAILABLE


class ESPAdapter(BasePlatformAdapter):
    """WebSocket bridge for ESP-based hardware terminals."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.ESP)
        extra = config.extra or {}
        self._bind_host = str(extra.get("bind_host") or "0.0.0.0")
        self._bind_port = int(extra.get("bind_port") or 8765)
        self._shared_token = str(extra.get("shared_token") or "").strip()
        self._allowed_devices = {
            d.strip() for d in str(extra.get("allowed_devices") or "").split(",") if d.strip()
        }

        self._app: Optional[web.Application] = None if AIOHTTP_AVAILABLE else None
        self._runner: Optional[web.AppRunner] = None if AIOHTTP_AVAILABLE else None
        self._site: Optional[web.TCPSite] = None if AIOHTTP_AVAILABLE else None
        self._clients: Dict[str, web.WebSocketResponse] = {}

    async def connect(self) -> bool:
        """Start the local WebSocket bridge."""
        if not AIOHTTP_AVAILABLE:
            logger.error("[%s] aiohttp not installed", self.name)
            return False

        try:
            self._app = web.Application()
            self._app.add_routes([
                web.get("/ws", self._handle_ws),
                web.get("/health", self._handle_health),
            ])
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._bind_host, self._bind_port)
            await self._site.start()
            self._running = True
            logger.info("[%s] Listening on ws://%s:%s/ws", self.name, self._bind_host, self._bind_port)
            return True
        except Exception as exc:
            logger.error("[%s] Failed to start ESP bridge: %s", self.name, exc, exc_info=True)
            return False

    async def disconnect(self) -> None:
        """Stop the local bridge server."""
        for device_id, ws in list(self._clients.items()):
            try:
                await ws.close()
            except Exception:
                logger.debug("[%s] Failed to close socket for %s", self.name, device_id)
        self._clients.clear()

        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception as exc:
                logger.warning("[%s] Runner cleanup failed: %s", self.name, exc)

        self._site = None
        self._runner = None
        self._app = None
        self._running = False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text response back to a connected device."""
        ws = self._clients.get(str(chat_id))
        if not ws:
            return SendResult(success=False, error=f"Device {chat_id} is not connected")

        payload = {
            "type": "agent_text",
            "text": content,
        }
        if reply_to:
            payload["reply_to"] = reply_to
        if metadata:
            payload["metadata"] = metadata

        try:
            await ws.send_json(payload)
            return SendResult(success=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Expose a simple thinking status to the device."""
        await self._send_status(str(chat_id), "thinking", metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return device-oriented chat metadata."""
        return {
            "name": f"ESP Device {chat_id}",
            "type": "dm",
            "chat_id": str(chat_id),
        }

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "platform": self.platform.value,
            "connected_devices": sorted(self._clients.keys()),
        })

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)

        device_id: Optional[str] = None
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue

                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "error": "Invalid JSON payload"})
                    continue

                msg_type = str(payload.get("type") or "").strip()
                if msg_type == "hello":
                    device_id = await self._register_device(ws, payload)
                    if not device_id:
                        break
                    continue

                if not device_id:
                    await ws.send_json({"type": "error", "error": "hello required before other messages"})
                    continue

                if msg_type == "user_text":
                    text = str(payload.get("text") or "").strip()
                    if not text:
                        await ws.send_json({"type": "error", "error": "text is required"})
                        continue
                    await self._send_status(device_id, "thinking")
                    await self.handle_message(self._build_text_event(device_id, text, payload))
                    continue

                if msg_type == "audio_end":
                    await self._handle_audio_end(device_id, ws, payload)
                    continue

                if msg_type == "reset_session":
                    await self.handle_message(self._build_text_event(device_id, "/new", payload))
                    continue

                await ws.send_json({"type": "error", "error": f"Unsupported message type: {msg_type}"})
        finally:
            if device_id and self._clients.get(device_id) is ws:
                self._clients.pop(device_id, None)
                logger.info("[%s] Device disconnected: %s", self.name, device_id)

        return ws

    async def _register_device(self, ws: web.WebSocketResponse, payload: Dict[str, Any]) -> Optional[str]:
        device_id = str(payload.get("device_id") or "").strip()
        token = str(payload.get("token") or "").strip()

        if not device_id:
            await ws.send_json({"type": "error", "error": "device_id is required"})
            await ws.close()
            return None

        if self._allowed_devices and device_id not in self._allowed_devices:
            await ws.send_json({"type": "error", "error": "Device is not allowed"})
            await ws.close()
            return None

        if self._shared_token and token != self._shared_token:
            await ws.send_json({"type": "error", "error": "Invalid device token"})
            await ws.close()
            return None

        old_ws = self._clients.get(device_id)
        if old_ws and old_ws is not ws:
            try:
                await old_ws.close()
            except Exception:
                logger.debug("[%s] Failed to close previous socket for %s", self.name, device_id)

        self._clients[device_id] = ws
        await ws.send_json({
            "type": "hello_ack",
            "device_id": device_id,
            "platform": self.platform.value,
        })
        logger.info("[%s] Device connected: %s", self.name, device_id)
        return device_id

    async def _handle_audio_end(
        self,
        device_id: str,
        ws: web.WebSocketResponse,
        payload: Dict[str, Any],
    ) -> None:
        await self._send_status(device_id, "transcribing")
        transcript = await self._transcribe_audio_payload(payload)
        if not transcript:
            await ws.send_json({
                "type": "error",
                "error": "Transcription failed or returned empty text",
            })
            await self._send_status(device_id, "error")
            return

        await self._send_status(device_id, "thinking")
        await self.handle_message(self._build_text_event(device_id, transcript, payload))

    def _build_text_event(self, device_id: str, text: str, raw_message: Dict[str, Any]) -> MessageEvent:
        source = self.build_source(
            chat_id=device_id,
            chat_name=f"ESP Device {device_id}",
            chat_type="dm",
            user_id=device_id,
            user_name=f"ESP Device {device_id}",
        )
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=raw_message,
            message_id=str(raw_message.get("request_id") or ""),
        )

    async def _send_status(
        self,
        device_id: str,
        state: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ws = self._clients.get(str(device_id))
        if not ws:
            return
        payload = {
            "type": "status",
            "state": state,
        }
        if metadata:
            payload["metadata"] = metadata
        try:
            await ws.send_json(payload)
        except Exception:
            logger.debug("[%s] Failed to send status '%s' to %s", self.name, state, device_id)

    async def _transcribe_audio_payload(self, payload: Dict[str, Any]) -> str:
        """
        Convert the final uploaded audio into text.

        Uses Hermes' built-in STT stack so the ESP adapter inherits whatever
        the user configured in ``stt:`` (local faster-whisper, Groq, OpenAI,
        etc.).
        """
        passthrough_text = str(payload.get("text") or "").strip()
        if passthrough_text:
            return passthrough_text

        audio_b64 = payload.get("audio_base64")
        if not audio_b64:
            return ""

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            logger.warning("[%s] Invalid base64 audio payload", self.name)
            return ""

        suffix = ".wav"
        mime_type = str(payload.get("mime_type") or "").lower()
        if "ogg" in mime_type:
            suffix = ".ogg"
        elif "mp3" in mime_type:
            suffix = ".mp3"

        temp_dir = Path("/tmp/hermes_esp_audio")
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"esp_{id(self)}_{asyncio.get_running_loop().time():.0f}{suffix}"
        audio_path.write_bytes(audio_bytes)

        try:
            return await self._transcribe_via_hermes_stt(audio_path)
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("[%s] Failed to remove temp audio file %s", self.name, audio_path)

    async def _transcribe_via_hermes_stt(self, audio_path: Path) -> str:
        """Use Hermes' built-in transcription stack configured in config.yaml."""
        from tools.transcription_tools import transcribe_audio, get_stt_model_from_config

        configured_model = get_stt_model_from_config()
        result = await asyncio.to_thread(transcribe_audio, str(audio_path), model=configured_model)
        if not result.get("success"):
            logger.warning(
                "[%s] Hermes STT transcription failed for %s: %s",
                self.name,
                audio_path.name,
                result.get("error", "unknown error"),
            )
            return ""

        transcript = str(result.get("transcript") or "").strip()
        if transcript:
            logger.info("[%s] Transcribed %s (%d chars)", self.name, audio_path.name, len(transcript))
        return transcript
