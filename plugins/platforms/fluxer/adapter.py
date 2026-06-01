"""Fluxer platform plugin for Hermes Agent.

Text-first adapter:
- REST `POST /channels/:id/messages` for outbound messages.
- Fluxer Gateway websocket `MESSAGE_CREATE` events for inbound messages.

Fluxer self-hosting is still moving, so this adapter intentionally keeps the
surface conservative and easy to test. Media/rich embeds can layer on once the
API settles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_IMAGE_DOCUMENT_TYPES,
    cache_audio_from_url,
    cache_document_from_bytes,
    cache_image_from_url,
    safe_url_for_log,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
_DEFAULT_BASE_URL = "https://api.fluxer.app/v1"
_GATEWAY_VERSION = 1
_VOICE_MESSAGE_FLAG = 1 << 13


def _strip_slash(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _api_base(base_url: str) -> str:
    """Normalize a user-provided Fluxer URL to the REST API base.

    The official hosted API is already scoped as ``https://api.fluxer.app/v1``.
    Self-hosted Fluxer may expose either an already-scoped API URL or a plain
    web origin; preserve scoped URLs and append ``/api`` only for a plain origin.
    """
    base = _strip_slash(base_url)
    if not base:
        return ""
    if base.endswith("/api") or base.endswith("/api/v1") or base.endswith("/v1") or "/api/" in base:
        return base
    if base == "https://api.fluxer.app":
        return f"{base}/v1"
    return f"{base}/api"


def _build_identify_payload(bot_token: str) -> Dict[str, Any]:
    return {
        "op": 2,
        "d": {
            "token": bot_token,
            "properties": {
                "os": "linux",
                "browser": "hermes",
                "device": "hermes",
            },
        },
    }


def _headers(bot_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
        "User-Agent": "Hermes-Fluxer/0.1",
    }


def _auth_headers(bot_token: str) -> Dict[str, str]:
    """Headers for requests where httpx must set Content-Type itself."""
    return {
        "Authorization": f"Bot {bot_token}",
        "User-Agent": "Hermes-Fluxer/0.1",
    }


def _event_seq(payload: Dict[str, Any]) -> Optional[int]:
    seq = payload.get("s")
    try:
        return int(seq) if seq is not None else None
    except (TypeError, ValueError):
        return None


def _author_name(author: Dict[str, Any]) -> Optional[str]:
    return (
        author.get("global_name")
        or author.get("display_name")
        or author.get("username")
        or author.get("name")
    )


def _chat_type(raw: Any) -> str:
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"dm", "direct", "private"}:
            return "dm"
        if lowered in {"group_dm", "group", "group-dm"}:
            return "group"
        if lowered in {"thread"}:
            return "thread"
        return "channel"
    # Discord-like channel types in several codebases: 1 = DM, 3 = group DM.
    if raw == 1:
        return "dm"
    if raw == 3:
        return "group"
    return "channel"


def _attachment_url(att: Dict[str, Any]) -> str:
    return str(att.get("url") or att.get("proxy_url") or "").strip()


def _attachment_filename(att: Dict[str, Any]) -> str:
    return str(att.get("filename") or att.get("title") or att.get("name") or "attachment").strip()


def _extension_for_attachment(att: Dict[str, Any], content_type: str, default: str = ".bin") -> str:
    filename = _attachment_filename(att)
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    subtype = (content_type or "").split("/", 1)[-1].split(";", 1)[0].lower()
    aliases = {"jpeg": ".jpg", "plain": ".txt", "mpeg": ".mp3", "quicktime": ".mov"}
    if subtype:
        return aliases.get(subtype, f".{subtype}")
    return default


def _message_type_for_media(media_types: List[str]) -> MessageType:
    if not media_types:
        return MessageType.TEXT
    if any(m.startswith("image/") for m in media_types):
        return MessageType.PHOTO
    if any(m.startswith("audio/") for m in media_types):
        return MessageType.AUDIO
    if any(m.startswith("video/") for m in media_types):
        return MessageType.VIDEO
    return MessageType.DOCUMENT


def _is_voice_message(data: Dict[str, Any]) -> bool:
    try:
        flags = int(data.get("flags") or 0)
    except (TypeError, ValueError):
        flags = 0
    if flags & _VOICE_MESSAGE_FLAG:
        return True
    if str(data.get("message_type") or data.get("type") or "").lower() in {"voice", "voice_message"}:
        return True
    attachments = data.get("attachments") or []
    if isinstance(attachments, list):
        for att in attachments:
            if not isinstance(att, dict):
                continue
            if bool(att.get("is_voice_message") or att.get("voice") or att.get("voice_message")):
                return True
    return False


def _audio_duration_seconds(path: Path) -> int:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        duration = float((result.stdout or "").strip())
        if duration > 0:
            return max(1, int(round(duration)))
    except Exception:
        pass
    return 1


class FluxerAdapter(BasePlatformAdapter):
    """Fluxer adapter using bot REST + Gateway websocket APIs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("fluxer"))
        extra = getattr(config, "extra", {}) or {}
        self.base_url = _strip_slash(
            os.getenv("FLUXER_BASE_URL") or extra.get("base_url") or _DEFAULT_BASE_URL
        )
        self.api_base_url = _api_base(self.base_url)
        self.bot_token = (
            os.getenv("FLUXER_BOT_TOKEN") or extra.get("bot_token") or ""
        ).strip()
        self.gateway_url = _strip_slash(
            os.getenv("FLUXER_GATEWAY_URL") or extra.get("gateway_url") or ""
        )
        self.bot_user_id: Optional[str] = str(extra.get("bot_user_id")) if extra.get("bot_user_id") else None
        self._ws = None
        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_seq: Optional[int] = None
        self._seen_message_ids: set[str] = set()

    async def connect(self) -> bool:
        if not self.base_url or not self.bot_token:
            self._set_fatal_error(
                "missing_config",
                "Fluxer requires FLUXER_BOT_TOKEN; FLUXER_BASE_URL is optional and defaults to https://api.fluxer.app/v1",
                retryable=False,
            )
            return False
        try:
            if not self.gateway_url:
                info = await self._request("GET", "/gateway/bot")
                self.gateway_url = _strip_slash(str(info.get("url") or ""))
            if not self.gateway_url:
                raise RuntimeError("Fluxer gateway URL missing from /gateway/bot")

            import websockets

            sep = "&" if "?" in self.gateway_url else "?"
            ws_url = f"{self.gateway_url}{sep}v={_GATEWAY_VERSION}&encoding=json"
            self._ws = await websockets.connect(ws_url, open_timeout=15, close_timeout=5, max_size=None)
            self._listener_task = asyncio.create_task(self._listen_loop(), name="fluxer-listen")
            self._mark_connected()
            return True
        except Exception as exc:
            logger.warning("Fluxer connect failed: %s", exc)
            self._set_fatal_error("connect_failed", f"Fluxer connect failed: {exc}", retryable=True)
            return False

    async def disconnect(self) -> None:
        self._running = False
        for task in (self._heartbeat_task, self._listener_task):
            if task and not task.done():
                task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._heartbeat_task = None
        self._listener_task = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        payload: Dict[str, Any] = {"content": content}
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        if metadata:
            thread_id = metadata.get("thread_id")
            if thread_id and "message_reference" not in payload:
                # Fluxer thread semantics are still stabilizing; keep this as
                # metadata only when callers explicitly provide it.
                payload["message_reference"] = {"message_id": str(thread_id)}

        try:
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)
            message_ids: List[str] = []
            responses: List[Dict[str, Any]] = []

            for index, chunk in enumerate(chunks):
                chunk_payload = dict(payload)
                chunk_payload["content"] = chunk
                if index > 0:
                    # Reply/reference metadata only belongs on the first split
                    # chunk; applying the same reference to every continuation
                    # creates noisy threads and can make partial retries nastier.
                    chunk_payload.pop("message_reference", None)

                data = await self._request(
                    "POST",
                    f"/channels/{chat_id}/messages",
                    json=chunk_payload,
                )
                responses.append(data)
                if data.get("id"):
                    message_ids.append(str(data["id"]))

            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids, "responses": responses},
            )
        except Exception as exc:
            logger.warning("Fluxer send failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            cached = await cache_image_from_url(image_url, Path(image_url.split("?", 1)[0]).suffix or ".jpg")
            return await self.send_image_file(chat_id, cached, caption=caption, reply_to=reply_to, metadata=metadata)
        except Exception as exc:
            logger.warning("Fluxer image URL upload failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            video_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            audio_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            flags=_VOICE_MESSAGE_FLAG,
            is_voice=True,
            duration=kwargs.get("duration"),
            waveform=kwargs.get("waveform"),
            title=kwargs.get("title"),
        )

    async def _send_file_message(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        flags: int = 0,
        is_voice: bool = False,
        duration: Optional[int] = None,
        waveform: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SendResult:
        resolved = self.validate_media_delivery_path(file_path)
        if not resolved:
            return SendResult(success=False, error=f"Unsafe or missing file path: {file_path}", retryable=False)

        path = Path(resolved)
        filename = file_name or path.name
        payload: Dict[str, Any] = {"nonce": str(int(time.time() * 1000))}
        if caption and not is_voice:
            payload["content"] = caption
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        if metadata:
            thread_id = metadata.get("thread_id")
            if thread_id and "message_reference" not in payload:
                payload["message_reference"] = {"message_id": str(thread_id)}
        if flags:
            payload["flags"] = flags

        attachment: Dict[str, Any] = {"id": 0, "filename": filename, "title": title or filename}
        if is_voice:
            # Fluxer's schema requires duration + waveform for VOICE_MESSAGE uploads.
            attachment["duration"] = int(duration) if duration is not None else _audio_duration_seconds(path)
            attachment["waveform"] = str(waveform or "AAAA")
        payload["attachments"] = [attachment]

        try:
            data = await self._multipart_request(
                "POST",
                f"/channels/{chat_id}/messages",
                payload=payload,
                files=[("files[0]", path, filename)],
            )
            return SendResult(success=True, message_id=str(data.get("id")) if data.get("id") else None, raw_response=data)
        except Exception as exc:
            logger.warning("Fluxer file upload failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            data = await self._request("GET", f"/channels/{chat_id}")
            return {
                "id": str(data.get("id") or chat_id),
                "name": data.get("name") or str(chat_id),
                "type": _chat_type(data.get("type")),
                "raw": data,
            }
        except Exception:
            return {"id": str(chat_id), "name": str(chat_id), "type": "channel"}

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for Fluxer adapter") from exc

        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.request(method, url, headers=_headers(self.bot_token), **kwargs)
            if response.status_code >= 400:
                logger.warning(
                    "Fluxer REST %s %s failed: status=%s body=%s",
                    method,
                    path,
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    async def _multipart_request(
        self,
        method: str,
        path: str,
        *,
        payload: Dict[str, Any],
        files: List[tuple[str, Path, str]],
    ) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for Fluxer adapter") from exc

        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        multipart_files = []
        handles = []
        try:
            for field_name, file_path, filename in files:
                handle = file_path.open("rb")
                handles.append(handle)
                content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                multipart_files.append((field_name, (filename, handle, content_type)))
            data = {"payload_json": json.dumps(payload, separators=(",", ":"))}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.request(
                    method,
                    url,
                    headers=_auth_headers(self.bot_token),
                    data=data,
                    files=multipart_files,
                )
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
        finally:
            for handle in handles:
                try:
                    handle.close()
                except Exception:
                    pass

    async def _download_attachment_bytes(self, url: str) -> bytes:
        try:
            import httpx
            from tools.url_safety import is_safe_url
        except ImportError as exc:
            raise RuntimeError("httpx and url safety helpers are required for Fluxer attachments") from exc

        if not is_safe_url(url):
            raise ValueError(f"Blocked unsafe Fluxer attachment URL: {safe_url_for_log(url)}")
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "Authorization": f"Bot {self.bot_token}",
                    "User-Agent": "Hermes-Fluxer/0.1",
                    "Accept": "*/*",
                },
            )
            response.raise_for_status()
            return response.content

    async def _cache_attachment(self, att: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        url = _attachment_url(att)
        if not url:
            return None, None
        content_type = str(att.get("content_type") or att.get("contentType") or "application/octet-stream").split(";", 1)[0].lower()
        filename = _attachment_filename(att)
        ext = _extension_for_attachment(att, content_type)

        try:
            if content_type.startswith("image/") or ext in SUPPORTED_IMAGE_DOCUMENT_TYPES:
                image_type = content_type if content_type.startswith("image/") else SUPPORTED_IMAGE_DOCUMENT_TYPES.get(ext, "image/jpeg")
                image_ext = ext if ext in SUPPORTED_IMAGE_DOCUMENT_TYPES else _extension_for_attachment(att, image_type, ".jpg")
                return await cache_image_from_url(url, image_ext), image_type
            if content_type.startswith("audio/"):
                return await cache_audio_from_url(url, ext if ext != ".bin" else ".ogg"), content_type

            data = await self._download_attachment_bytes(url)
            return cache_document_from_bytes(data, filename), content_type or "application/octet-stream"
        except Exception as exc:
            logger.warning("Fluxer failed to cache attachment %s: %s", filename, exc)
            return url, content_type or "application/octet-stream"

    async def _extract_attachments(self, data: Dict[str, Any]) -> tuple[List[str], List[str]]:
        media_urls: List[str] = []
        media_types: List[str] = []
        attachments = data.get("attachments") or []
        if not isinstance(attachments, list):
            return media_urls, media_types
        for att in attachments:
            if not isinstance(att, dict):
                continue
            cached, mtype = await self._cache_attachment(att)
            if cached:
                media_urls.append(cached)
                media_types.append(mtype or "application/octet-stream")
        return media_urls, media_types

    async def _listen_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                await self._handle_gateway_dispatch(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._running:
                logger.warning("Fluxer listener stopped: %s", exc)
                self._set_fatal_error("listener_stopped", f"Fluxer listener stopped: {exc}", retryable=True)

    async def _heartbeat_loop(self, interval_ms: int) -> None:
        try:
            while self._running and self._ws is not None:
                await asyncio.sleep(max(interval_ms, 1000) / 1000)
                await self._ws.send(json.dumps({"op": 1, "d": self._last_seq}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Fluxer heartbeat stopped: %s", exc)

    async def _handle_gateway_dispatch(self, payload: Dict[str, Any]) -> None:
        op = payload.get("op")
        self._last_seq = _event_seq(payload) or self._last_seq

        if op == 10:  # HELLO
            interval = int(((payload.get("d") or {}).get("heartbeat_interval") or 41250))
            if self._ws is not None:
                await self._ws.send(json.dumps(_build_identify_payload(self.bot_token)))
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval), name="fluxer-heartbeat")
            return
        if op != 0:  # not DISPATCH
            return

        event_name = payload.get("t")
        data = payload.get("d") or {}
        if event_name == "READY":
            user = data.get("user") or data.get("bot") or {}
            if user.get("id"):
                self.bot_user_id = str(user["id"])
            return
        if event_name != "MESSAGE_CREATE":
            return
        await self._handle_message_create(data, payload)

    async def _handle_message_create(self, data: Dict[str, Any], raw_payload: Dict[str, Any]) -> None:
        msg_id = str(data.get("id") or "")
        if msg_id:
            if msg_id in self._seen_message_ids:
                return
            self._seen_message_ids.add(msg_id)
            if len(self._seen_message_ids) > 2000:
                self._seen_message_ids = set(list(self._seen_message_ids)[-1000:])

        author = data.get("author") or data.get("user") or {}
        author_id = str(author.get("id") or data.get("author_id") or "")
        if author.get("bot") or (self.bot_user_id and author_id == str(self.bot_user_id)):
            return

        text = data.get("content") or ""
        media_urls, media_types = await self._extract_attachments(data)
        if not text and not media_urls:
            return

        channel_id = str(data.get("channel_id") or data.get("channel", {}).get("id") or "")
        if not channel_id:
            return
        source = self.build_source(
            chat_id=channel_id,
            chat_name=(data.get("channel") or {}).get("name"),
            chat_type=_chat_type(data.get("channel_type") or (data.get("channel") or {}).get("type")),
            user_id=author_id or None,
            user_name=_author_name(author),
            guild_id=data.get("guild_id"),
            message_id=msg_id or None,
        )

        timestamp = datetime.now(tz=timezone.utc)
        ts_raw = data.get("timestamp") or data.get("created_at")
        if isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                pass

        event = MessageEvent(
            text=text,
            message_type=MessageType.VOICE if _is_voice_message(data) else _message_type_for_media(media_types),
            source=source,
            raw_message=raw_payload,
            message_id=msg_id or None,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
        )
        await self.handle_message(event)


def check_requirements() -> bool:
    if not os.getenv("FLUXER_BOT_TOKEN"):
        return False
    try:
        import httpx  # noqa: F401
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("FLUXER_BOT_TOKEN") or extra.get("bot_token", "")
    return bool(str(token).strip())


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> dict | None:
    base_url = os.getenv("FLUXER_BASE_URL", "").strip()
    token = os.getenv("FLUXER_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {"bot_token": token}
    if base_url:
        seed["base_url"] = base_url
    gateway_url = os.getenv("FLUXER_GATEWAY_URL", "").strip()
    if gateway_url:
        seed["gateway_url"] = gateway_url
    home = os.getenv("FLUXER_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("FLUXER_HOME_CHANNEL_NAME", "").strip() or home,
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    adapter = FluxerAdapter(pconfig)
    metadata = {"thread_id": thread_id} if thread_id else None
    try:
        last: Optional[SendResult] = None
        if message:
            last = await adapter.send(chat_id, message, metadata=metadata)
            if not last.success:
                return {"error": last.error or "Fluxer send failed"}
        for media_item in media_files or []:
            if isinstance(media_item, (tuple, list)):
                media_path = str(media_item[0])
                is_voice_directive = bool(media_item[1]) if len(media_item) > 1 else False
            else:
                media_path = str(media_item)
                is_voice_directive = False
            ext = Path(media_path).suffix.lower()
            if not force_document and ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                last = await adapter.send_image_file(chat_id, media_path, metadata=metadata)
            elif not force_document and ext in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
                last = await adapter.send_video(chat_id, media_path, metadata=metadata)
            elif not force_document and (is_voice_directive or ext in {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac", ".aac"}):
                last = await adapter.send_voice(chat_id, media_path, metadata=metadata)
            else:
                last = await adapter.send_document(chat_id, media_path, metadata=metadata)
            if not last.success:
                return {"error": last.error or "Fluxer media send failed"}
        if last and last.success:
            return {"success": True, "platform": "fluxer", "chat_id": chat_id, "message_id": last.message_id}
        return {"error": "Fluxer send failed: empty message and no media"}
    except Exception as exc:
        return {"error": str(exc)}


def interactive_setup() -> None:
    print("Fluxer platform setup")
    print("Set FLUXER_BOT_TOKEN in ~/.hermes/.env, then restart the gateway.")
    print("Optional: set FLUXER_BASE_URL for self-hosted Fluxer; official hosted defaults to https://api.fluxer.app/v1.")


def register(ctx) -> None:
    ctx.register_platform(
        name="fluxer",
        label="Fluxer",
        adapter_factory=lambda cfg: FluxerAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["FLUXER_BOT_TOKEN"],
        install_hint="pip install httpx websockets   # Fluxer adapter dependencies",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="FLUXER_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="FLUXER_ALLOWED_USERS",
        allow_all_env="FLUXER_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="⚡",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Fluxer, a Discord-like open-source chat "
            "platform. Fluxer supports rich Markdown, channels, DMs, files, "
            "and voice/video. Prefer normal Markdown for structured replies."
        ),
    )
