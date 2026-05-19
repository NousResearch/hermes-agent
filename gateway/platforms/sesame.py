"""Native Sesame messaging platform adapter.

Connects Hermes Gateway to Sesame via the Sesame realtime WebSocket
(``/v1/connect``) for inbound events and the Sesame REST API for outbound
messages/files.  Configuration is intentionally env/config driven so any
Hermes agent with a Sesame API key can enable the platform without Bailey-
specific code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from hermes_constants import get_hermes_home
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

try:  # optional until a user enables the platform
    import aiohttp
except Exception:  # pragma: no cover - dependency absence tested via check fn
    aiohttp = None  # type: ignore[assignment]

try:  # optional until a user enables the platform
    import websockets
except Exception:  # pragma: no cover - dependency absence tested via check fn
    websockets = None  # type: ignore[assignment]


MAX_MESSAGE_LENGTH = 10_000
_DEFAULT_API_URL = "https://api.sesame.space"
_DEFAULT_WS_URL = "wss://ws.sesame.space"
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_DEFAULT_HEARTBEAT_INTERVAL = 30.0


class SesameAPIError(Exception):
    """Raised when a Sesame REST API call returns an error response."""

    def __init__(self, status: int, message: str, body: dict[str, Any]):
        self.status = status
        self.message = message
        self.body = body
        super().__init__(f"Sesame API {status}: {message}")


class SesameAuthError(Exception):
    """Raised when Sesame WebSocket authentication fails."""


def _extra(config: PlatformConfig) -> dict[str, Any]:
    return config.extra if isinstance(config.extra, dict) else {}


def _cfg_value(config: PlatformConfig, key: str, env: str, default: Optional[str] = None) -> Any:
    value = _extra(config).get(key)
    if value not in (None, ""):
        return value
    value = os.getenv(env)
    if value not in (None, ""):
        return str(value)
    return default


def _sesame_api_key(config: PlatformConfig) -> Optional[str]:
    return config.api_key or config.token or _cfg_value(config, "api_key", "SESAME_API_KEY")


def _allowed_users(config: PlatformConfig) -> set[str]:
    raw = _cfg_value(config, "allowed_users", "SESAME_ALLOWED_USERS", "") or ""
    if isinstance(raw, (list, tuple, set)):  # defensive; YAML bridge usually stringifies
        return {str(v).strip() for v in raw if str(v).strip()}
    return {uid.strip() for uid in str(raw).split(",") if uid.strip()}


def check_sesame_requirements() -> bool:
    """Return True when optional runtime dependencies are installed."""
    if aiohttp is None:
        logger.error("Sesame: aiohttp not installed. Run: pip install 'hermes-agent[messaging]'")
        return False
    if websockets is None:
        logger.error("Sesame: websockets not installed. Run: pip install websockets")
        return False
    return True


class SesameClient:
    """Small async Sesame REST + WebSocket client used by the adapter."""

    def __init__(self, api_url: str, ws_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self.api_key = api_key
        self.principal_id: Optional[str] = None
        self.handle: Optional[str] = None
        self.display_name: Optional[str] = None
        self.workspace_id: Optional[str] = None
        self._session: Any = None
        self._ws: Any = None
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False
        self._heartbeat_interval = _DEFAULT_HEARTBEAT_INTERVAL
        self._on_message: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None
        self._on_connected: Optional[Callable[[], Awaitable[None]]] = None

    def _get_session(self):
        if aiohttp is None:  # pragma: no cover - guarded by check_sesame_requirements
            raise RuntimeError("aiohttp is not installed")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        session = self._get_session()
        async with session.request(method, f"{self.api_url}{path}", json=json_body, params=params) as resp:
            try:
                body = await resp.json()
            except Exception:
                body = {"error": await resp.text()}
            if resp.status >= 400:
                raise SesameAPIError(resp.status, body.get("error") or body.get("message") or resp.reason, body)
            return body

    async def fetch_identity(self) -> dict[str, Any]:
        result = await self._request("GET", "/api/v1/auth/me")
        data = result.get("data", result)
        self.principal_id = data["id"]
        self.handle = data.get("handle")
        self.display_name = data.get("displayName")
        self.workspace_id = data.get("workspaceId")
        return data

    async def send_message(
        self,
        channel_id: str,
        content: str,
        *,
        kind: str = "text",
        intent: str = "chat",
        thread_root_id: Optional[str] = None,
        attachment_ids: Optional[list[str]] = None,
        client_generated_id: Optional[str] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"content": content, "kind": kind, "intent": intent}
        if attachment_ids and kind == "text":
            body["kind"] = "attachment"
        if thread_root_id:
            body["threadRootId"] = thread_root_id
        if attachment_ids:
            body["attachmentIds"] = attachment_ids
        body["clientGeneratedId"] = client_generated_id or f"hermes-{uuid.uuid4().hex}"
        result = await self._request("POST", f"/api/v1/channels/{channel_id}/messages", json_body=body)
        return result.get("data", result)

    async def edit_message(self, channel_id: str, message_id: str, content: str, *, streaming: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"content": content}
        if streaming:
            body["streaming"] = True
        result = await self._request("PATCH", f"/api/v1/channels/{channel_id}/messages/{message_id}", json_body=body)
        return result.get("data", result)

    async def get_channel_info(self, channel_id: str) -> dict[str, Any]:
        result = await self._request("GET", f"/api/v1/channels/{channel_id}")
        return result.get("data", result)

    async def upload_file(self, file_path: str, file_name: str, content_type: str, size: int, *, channel_id: Optional[str] = None) -> str:
        meta = await self._request(
            "POST",
            "/api/v1/drive/files/upload-url",
            json_body={"fileName": file_name, "contentType": content_type, "size": size},
        )
        data = meta.get("data", meta)
        upload_url = data["uploadUrl"]
        file_id = data["fileId"]
        s3_key = data["s3Key"]
        session = self._get_session()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        async with session.put(upload_url, data=file_bytes, headers={"Content-Type": content_type}) as resp:
            if resp.status >= 400:
                raise SesameAPIError(resp.status, "File upload failed", {})
        register_body: dict[str, Any] = {
            "fileId": file_id,
            "s3Key": s3_key,
            "fileName": file_name,
            "contentType": content_type,
            "size": size,
        }
        if channel_id:
            register_body["channelId"] = channel_id
        await self._request("POST", "/api/v1/drive/files", json_body=register_body)
        return file_id

    def on_message(self, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._on_message = callback

    def on_connected(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._on_connected = callback

    async def connect_ws(self) -> None:
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())

    async def disconnect(self) -> None:
        self._running = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send_ws(self, frame: dict[str, Any]) -> None:
        if self._ws:
            await self._ws.send(json.dumps(frame))

    async def send_typing(self, channel_id: str) -> None:
        await self.send_ws({"type": "typing", "channelId": channel_id})

    async def request_replay(self, cursors: dict[str, int]) -> None:
        await self.send_ws({"type": "replay", "cursors": cursors})

    async def send_runtime_meta(self, runtime: str, version: str) -> None:
        await self.send_ws({"type": "meta", "runtime": runtime, "version": version})

    async def _ws_loop(self) -> None:
        backoff = _INITIAL_BACKOFF
        while self._running:
            try:
                await self._connect_and_listen()
                backoff = _INITIAL_BACKOFF
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("Sesame WebSocket disconnected: %s — reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _connect_and_listen(self) -> None:
        if websockets is None:  # pragma: no cover - guarded by check_sesame_requirements
            raise RuntimeError("websockets is not installed")
        ws_uri = f"{self.ws_url}/v1/connect"
        async with websockets.connect(ws_uri, ping_interval=None, max_size=10 * 1024 * 1024, close_timeout=5) as ws:
            self._ws = ws
            await ws.send(json.dumps({"type": "auth", "apiKey": self.api_key}))
            auth_response = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if auth_response.get("type") != "authenticated":
                raise SesameAuthError(auth_response.get("error", "Authentication failed"))
            self._heartbeat_interval = auth_response.get("heartbeatIntervalMs", 30000) / 1000.0
            await self.send_runtime_meta("hermes", "1.0.0")
            if self._on_connected:
                await self._on_connected()
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
            try:
                await self._listen(ws)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _heartbeat_loop(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await ws.send(json.dumps({"type": "ping"}))
            except Exception:
                break

    async def _listen(self, ws: Any) -> None:
        async for raw in ws:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Sesame WS: invalid JSON frame: %s", str(raw)[:200])
                continue
            if frame.get("type") in {"pong", "authenticated", "delivery.ack"}:
                continue
            if self._on_message:
                try:
                    await self._on_message(frame)
                except Exception:
                    logger.exception("Error handling Sesame WS frame type=%s", frame.get("type"))


class SesameAdapter(BasePlatformAdapter):
    """Sesame messaging platform adapter for Hermes Gateway."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SESAME)
        api_key = _sesame_api_key(config)
        if not api_key:
            raise ValueError("Sesame API key is required. Set SESAME_API_KEY or config platforms.sesame.api_key/token.")
        self.api_key = api_key
        self.api_url = str(_cfg_value(config, "api_url", "SESAME_API_URL", _DEFAULT_API_URL) or _DEFAULT_API_URL)
        self.ws_url = str(_cfg_value(config, "ws_url", "SESAME_WS_URL", _DEFAULT_WS_URL) or _DEFAULT_WS_URL)
        self.allowed_users = _allowed_users(config)
        self.client = SesameClient(self.api_url, self.ws_url, self.api_key)
        self._cursor_path = get_hermes_home() / "gateway" / "sesame_cursors.json"

    async def connect(self) -> bool:
        if not check_sesame_requirements():
            self._set_fatal_error("missing_deps", "Missing aiohttp or websockets", retryable=False)
            await self._notify_fatal_error()
            return False
        try:
            await self.client.fetch_identity()
            self.client.on_message(self._on_ws_event)
            self.client.on_connected(self._request_replay_on_connect)
            await self.client.connect_ws()
            self._mark_connected()
            logger.info("Sesame adapter connected as %s (%s)", self.client.handle, self.client.principal_id)
            return True
        except Exception as exc:
            logger.error("Sesame connection failed: %s", exc)
            self._set_fatal_error("connect_failed", str(exc), retryable=True)
            await self._notify_fatal_error()
            return False

    async def disconnect(self) -> None:
        self._mark_disconnected()
        await self.client.disconnect()

    def _cursor_bucket(self) -> str:
        return self.client.principal_id or "default"

    def _load_cursors(self) -> dict[str, int]:
        try:
            data = json.loads(self._cursor_path.read_text())
        except Exception:
            return {}
        raw = data.get(self._cursor_bucket(), {}) if isinstance(data, dict) else {}
        if not isinstance(raw, dict):
            return {}
        cursors: dict[str, int] = {}
        for channel_id, seq in raw.items():
            try:
                cursors[str(channel_id)] = int(seq)
            except (TypeError, ValueError):
                continue
        return cursors

    def _save_cursor(self, channel_id: str, seq: Any) -> None:
        try:
            seq_int = int(seq)
        except (TypeError, ValueError):
            return
        if not channel_id or seq_int < 0:
            return
        try:
            data = json.loads(self._cursor_path.read_text())
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        bucket = data.setdefault(self._cursor_bucket(), {})
        if not isinstance(bucket, dict):
            bucket = {}
            data[self._cursor_bucket()] = bucket
        try:
            existing = int(bucket.get(channel_id, -1))
        except (TypeError, ValueError):
            existing = -1
        if existing >= seq_int:
            return
        bucket[channel_id] = seq_int
        self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cursor_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True))
        tmp.replace(self._cursor_path)

    async def _request_replay_on_connect(self) -> None:
        cursors = self._load_cursors()
        if cursors:
            await self.client.request_replay(cursors)

    async def _on_ws_event(self, frame: dict[str, Any]) -> None:
        if frame.get("type") == "message":
            await self._handle_incoming_message(frame.get("message") or {})

    async def _handle_incoming_message(self, msg: dict[str, Any]) -> None:
        sender_id = str(msg.get("senderId") or "")
        channel_id = str(msg.get("channelId") or "")
        content = msg.get("plaintext") or msg.get("content") or ""
        kind = msg.get("kind", "text")
        message_id = str(msg.get("id") or "") or None
        metadata = msg.get("metadata") or {}
        thread_root_id = msg.get("threadRootId")

        if channel_id and msg.get("seq") is not None:
            self._save_cursor(channel_id, msg.get("seq"))

        if sender_id and sender_id == self.client.principal_id:
            return
        if kind != "text" or not str(content).strip() or not channel_id:
            return
        if self.allowed_users and sender_id not in self.allowed_users:
            logger.debug("Sesame: ignoring message from unauthorized principal %s", sender_id)
            return

        chat_type = "channel"
        channel_name = channel_id
        try:
            channel_info = await self.client.get_channel_info(channel_id)
            channel_name = channel_info.get("name") or channel_id
            ch_kind = channel_info.get("kind", "channel")
            if ch_kind == "dm":
                chat_type = "dm"
            elif ch_kind == "group":
                chat_type = "group"
        except Exception:
            pass

        source = SessionSource(
            platform=Platform.SESAME,
            chat_id=f"sesame:{channel_id}",
            chat_name=channel_name,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=metadata.get("senderDisplayName") or metadata.get("senderHandle") or sender_id,
            thread_id=str(thread_root_id) if thread_root_id else None,
            message_id=message_id,
            is_bot=metadata.get("senderKind") == "agent",
        )

        media_urls: list[str] = []
        media_types: list[str] = []
        for att in metadata.get("attachments") or []:
            download_url = att.get("downloadUrl")
            content_type = att.get("contentType", "")
            if not download_url:
                continue
            if content_type.startswith("image/"):
                try:
                    from gateway.platforms.base import cache_image_from_url
                    ext = mimetypes.guess_extension(content_type) or ".jpg"
                    media_urls.append(await cache_image_from_url(download_url, ext))
                    media_types.append("photo")
                except Exception:
                    logger.warning("Failed to cache Sesame image attachment: %s", att.get("fileName"))
            elif content_type.startswith("audio/"):
                try:
                    from gateway.platforms.base import cache_audio_from_url
                    ext = mimetypes.guess_extension(content_type) or ".ogg"
                    media_urls.append(await cache_audio_from_url(download_url, ext))
                    media_types.append("audio")
                except Exception:
                    logger.warning("Failed to cache Sesame audio attachment: %s", att.get("fileName"))

        msg_type = MessageType.TEXT
        if media_types[:1] == ["photo"]:
            msg_type = MessageType.PHOTO
        elif media_types[:1] == ["audio"]:
            msg_type = MessageType.AUDIO

        event = MessageEvent(
            text=str(content),
            message_type=msg_type,
            source=source,
            raw_message=msg,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=str(thread_root_id) if thread_root_id else None,
        )
        await self.handle_message(event)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        channel_id = self._resolve_channel_id(chat_id)
        thread_root_id = (metadata or {}).get("thread_id") or reply_to
        try:
            result = await self.client.send_message(channel_id, content, thread_root_id=thread_root_id)
            return SendResult(success=True, message_id=result.get("id"), raw_response=result)
        except SesameAPIError as exc:
            return SendResult(success=False, error=str(exc), retryable=exc.status >= 500 or exc.status == 429)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        channel_id = self._resolve_channel_id(chat_id)
        try:
            result = await self.client.edit_message(channel_id, message_id, content, streaming=not finalize)
            return SendResult(success=True, message_id=message_id, raw_response=result)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_typing(self, chat_id: str, metadata: Optional[dict[str, Any]] = None) -> None:
        try:
            await self.client.send_typing(self._resolve_channel_id(chat_id))
        except Exception:
            pass

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        try:
            channel_id = self._resolve_channel_id(chat_id)
            file_id = await self._upload_from_path_or_url(image_url, channel_id)
            result = await self.client.send_message(
                channel_id,
                caption or "Image attachment",
                thread_root_id=(metadata or {}).get("thread_id") or reply_to,
                attachment_ids=[file_id],
            )
            return SendResult(success=True, message_id=result.get("id"), raw_response=result)
        except Exception as exc:
            logger.warning("Sesame send_image failed: %s", exc)
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id, text, reply_to, metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        try:
            channel_id = self._resolve_channel_id(chat_id)
            path = Path(file_path)
            name = file_name or path.name
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            file_id = await self.client.upload_file(str(path), name, content_type, path.stat().st_size, channel_id=channel_id)
            result = await self.client.send_message(
                channel_id,
                caption or f"Attachment: {name}",
                thread_root_id=(metadata or {}).get("thread_id") or reply_to,
                attachment_ids=[file_id],
            )
            return SendResult(success=True, message_id=result.get("id"), raw_response=result)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        channel_id = self._resolve_channel_id(chat_id)
        try:
            info = await self.client.get_channel_info(channel_id)
            ch_kind = info.get("kind", "channel")
            return {
                "name": info.get("name") or channel_id,
                "type": ch_kind if ch_kind in {"dm", "group"} else "channel",
                "chat_id": chat_id,
            }
        except Exception:
            return {"name": chat_id, "type": "channel", "chat_id": chat_id}

    def _resolve_channel_id(self, chat_id: str) -> str:
        return chat_id[7:] if chat_id.startswith("sesame:") else chat_id

    async def _upload_from_path_or_url(self, path_or_url: str, channel_id: str) -> str:
        p = Path(path_or_url)
        if p.exists():
            name = p.name
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            return await self.client.upload_file(str(p), name, content_type, p.stat().st_size, channel_id=channel_id)
        if aiohttp is None:  # pragma: no cover
            raise RuntimeError("aiohttp is not installed")
        async with aiohttp.ClientSession() as sess:
            async with sess.get(path_or_url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                content_type = resp.content_type or "application/octet-stream"
        ext = mimetypes.guess_extension(content_type) or ""
        tmp = Path(tempfile.mkdtemp()) / f"upload_{uuid.uuid4().hex[:8]}{ext}"
        tmp.write_bytes(data)
        try:
            return await self.client.upload_file(str(tmp), tmp.name, content_type, len(data), channel_id=channel_id)
        finally:
            tmp.unlink(missing_ok=True)
