"""Blooio platform adapter.

Webhook-based Hermes gateway integration for Blooio messaging.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_API_BASE = "https://backend.blooio.com/v2/api"
_DEFAULT_WEBHOOK_PORT = 8081
_DEFAULT_BIND_HOST = "0.0.0.0"
_MEDIA_TTL_SECONDS = 3600
_WEBHOOK_STATE_PREFIX = "instance-"
_PHONE_OR_EMAIL_RE = re.compile(r"^(\+\d[\d\- ]{6,}|[^@\s]+@[^@\s]+\.[^@\s]+)$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def check_blooio_requirements() -> bool:
    """Return True when Blooio adapter runtime requirements are present."""
    return bool(
        AIOHTTP_AVAILABLE
        and os.getenv("BLOOIO_API_KEY")
        and os.getenv("BLOOIO_PUBLIC_BASE_URL")
    )


def _sanitize_filename(name: str) -> str:
    cleaned = Path(name or "attachment").name.replace("\x00", "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", cleaned)
    return cleaned or "attachment"


def _guess_extension(filename: str, mime_type: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext:
        return ext
    guessed = mimetypes.guess_extension(mime_type or "")
    return guessed or ""


def _media_kind(filename: str, mime_type: str) -> str:
    mime_type = (mime_type or "").lower()
    ext = Path(filename).suffix.lower()
    if mime_type.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic"}:
        return "image"
    if mime_type.startswith("audio/") or ext in {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".aac"}:
        return "audio"
    if mime_type.startswith("video/") or ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}:
        return "video"
    return "document"


class BlooioAdapter(BasePlatformAdapter):
    """Webhook-based Blooio adapter."""

    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.BLOOIO)
        extra = config.extra or {}
        self._hermes_home = get_hermes_home()
        self._state_root = self._hermes_home / "blooio"
        self._media_root = self._state_root / "media"
        self._api_key = (config.api_key or os.getenv("BLOOIO_API_KEY", "")).strip()
        self._public_base_url = str(
            extra.get("public_base_url") or os.getenv("BLOOIO_PUBLIC_BASE_URL", "")
        ).strip().rstrip("/")
        self._bind_host = str(
            extra.get("bind_host") or os.getenv("BLOOIO_BIND_HOST", _DEFAULT_BIND_HOST)
        ).strip() or _DEFAULT_BIND_HOST
        self._webhook_port = int(extra.get("webhook_port") or os.getenv("BLOOIO_WEBHOOK_PORT", str(_DEFAULT_WEBHOOK_PORT)))
        self._from_number = str(extra.get("from_number") or os.getenv("BLOOIO_FROM_NUMBER", "")).strip() or None
        self._instance_id = str(
            extra.get("instance_id") or os.getenv("BLOOIO_INSTANCE_ID", "")
        ).strip() or self._derive_instance_id()
        self._webhook_path = f"/webhooks/blooio/{self._instance_id}"
        self._media_path_prefix = f"/blooio/media/{self._instance_id}"
        self._webhook_url = f"{self._public_base_url}{self._webhook_path}" if self._public_base_url else ""
        self._state_file = self._state_root / f"{_WEBHOOK_STATE_PREFIX}{self._instance_id}.json"
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.BaseSite] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._management_lock_identity = f"{self._api_key}:{self._webhook_url}"
        self._status_by_message_id: Dict[str, Dict[str, Any]] = {}
        self._webhook_state = self._load_state()
        self._webhook_secret = str(self._webhook_state.get("signing_secret", "")).strip()
        self._media_secret = str(self._webhook_state.get("media_secret", "")).strip()
        if not self._media_secret:
            self._media_secret = secrets.token_hex(16)
            self._persist_state({"media_secret": self._media_secret})

    def _derive_instance_id(self) -> str:
        home = str(self._hermes_home.resolve())
        return hashlib.sha256(home.encode("utf-8")).hexdigest()[:12]

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_file.exists():
            return {}
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            logger.debug("[%s] Failed to load Blooio state file %s", self.name, self._state_file)
        return {}

    def _persist_state(self, updates: Dict[str, Any]) -> None:
        payload = dict(self._webhook_state)
        payload.update(updates)
        payload["instance_id"] = self._instance_id
        payload["webhook_url"] = self._webhook_url
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._webhook_state = payload
        self._webhook_secret = str(payload.get("signing_secret", "")).strip()
        self._media_secret = str(payload.get("media_secret", self._media_secret)).strip()

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.error("[%s] aiohttp not installed", self.name)
            return False
        if not self._api_key:
            logger.error("[%s] BLOOIO_API_KEY not set", self.name)
            return False
        if not self._public_base_url:
            logger.error("[%s] BLOOIO_PUBLIC_BASE_URL not set", self.name)
            return False

        from gateway.status import acquire_scoped_lock

        acquired, existing = acquire_scoped_lock(
            "blooio-webhook",
            self._management_lock_identity,
            metadata={"platform": self.platform.value, "webhook_url": self._webhook_url},
        )
        if not acquired:
            owner_pid = existing.get("pid") if isinstance(existing, dict) else None
            message = (
                "Another local Hermes gateway is already managing this Blooio webhook"
                + (f" (PID {owner_pid})." if owner_pid else ".")
            )
            self._set_fatal_error("blooio_webhook_lock", message, retryable=False)
            logger.error("[%s] %s", self.name, message)
            return False

        self._state_root.mkdir(parents=True, exist_ok=True)
        self._media_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_expired_media()

        try:
            self._http_session = aiohttp.ClientSession(headers=self._auth_headers())
            await self._validate_api_key()
            await self._ensure_webhook()
            await self._start_server()
            self._mark_connected()
            logger.info("[%s] Connected using webhook %s", self.name, self._webhook_url)
            return True
        except Exception as exc:
            await self._teardown_session()
            try:
                from gateway.status import release_scoped_lock
                release_scoped_lock("blooio-webhook", self._management_lock_identity)
            except Exception:
                pass
            message = f"Blooio startup failed: {exc}"
            self._set_fatal_error("blooio_connect_error", message, retryable=True)
            logger.error("[%s] %s", self.name, message, exc_info=True)
            return False

    async def disconnect(self) -> None:
        await self.cancel_background_tasks()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        await self._teardown_session()
        try:
            from gateway.status import release_scoped_lock
            release_scoped_lock("blooio-webhook", self._management_lock_identity)
        except Exception:
            pass
        self._mark_disconnected()
        logger.info("[%s] Disconnected", self.name)

    async def _teardown_session(self) -> None:
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def _validate_api_key(self) -> None:
        payload = await self._api_request("GET", "/me")
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid Blooio auth response")

    async def _start_server(self) -> None:
        assert web is not None
        app = web.Application()
        app.router.add_post(self._webhook_path, self._handle_webhook)
        app.router.add_get(f"{self._media_path_prefix}/{{token}}/{{filename}}", self._serve_media)
        app.router.add_get("/health", self._handle_health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._bind_host, self._webhook_port)
        await self._site.start()

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "platform": self.platform.value, "instance_id": self._instance_id})

    async def _api_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        allow_status: Optional[set[int]] = None,
    ) -> Any:
        if not self._http_session:
            raise RuntimeError("Blooio session not initialized")
        url = f"{_API_BASE}{path}"
        allow_status = allow_status or {200}
        async with self._http_session.request(method, url, json=json_body) as resp:
            text = await resp.text()
            if resp.status not in allow_status:
                raise RuntimeError(f"Blooio {method} {path} failed: {resp.status} {text[:300]}")
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}

    def _coerce_webhook_list(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("webhooks", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            if {"webhook_id", "webhook_url"} <= set(payload.keys()):
                return [payload]
        return []

    async def _ensure_webhook(self) -> None:
        all_webhooks = self._coerce_webhook_list(await self._api_request("GET", "/webhooks"))
        if len(all_webhooks) > 1:
            logger.warning(
                "[%s] Multiple Blooio webhooks exist for this API key. Overlapping allowlists across Hermes instances may cause duplicate responses.",
                self.name,
            )
        existing = next((item for item in all_webhooks if item.get("webhook_url") == self._webhook_url), None)
        if existing:
            webhook_id = str(existing.get("webhook_id", "")).strip()
            if not webhook_id:
                raise RuntimeError("Matched Blooio webhook missing webhook_id")
            webhook_type = str(existing.get("webhook_type", "")).strip().lower()
            if webhook_type and webhook_type != "all":
                await self._api_request(
                    "PATCH",
                    f"/webhooks/{quote(webhook_id, safe='')}",
                    json_body={"webhook_type": "all"},
                    allow_status={200},
                )
            self._persist_state({"webhook_id": webhook_id})
            if not self._webhook_secret:
                rotate = await self._api_request(
                    "POST",
                    f"/webhooks/{quote(webhook_id, safe='')}/secret/rotate",
                    allow_status={200, 201},
                )
                secret = str((rotate or {}).get("signing_secret", "")).strip()
                if not secret:
                    raise RuntimeError("Blooio webhook secret rotation did not return signing_secret")
                self._persist_state({"signing_secret": secret})
            return

        created = await self._api_request(
            "POST",
            "/webhooks",
            json_body={"webhook_url": self._webhook_url, "webhook_type": "all", "valid_until": -1},
            allow_status={200, 201, 409},
        )
        created = created if isinstance(created, dict) else {}
        webhook_id = str(created.get("webhook_id", "")).strip()
        secret = str(created.get("signing_secret", "")).strip()
        if not webhook_id:
            refreshed = self._coerce_webhook_list(await self._api_request("GET", "/webhooks"))
            existing = next((item for item in refreshed if item.get("webhook_url") == self._webhook_url), None)
            webhook_id = str((existing or {}).get("webhook_id", "")).strip()
        if not webhook_id:
            raise RuntimeError("Failed to determine Blooio webhook_id for this instance")
        updates = {"webhook_id": webhook_id}
        if secret:
            updates["signing_secret"] = secret
        self._persist_state(updates)
        if not self._webhook_secret:
            rotate = await self._api_request(
                "POST",
                f"/webhooks/{quote(webhook_id, safe='')}/secret/rotate",
                allow_status={200, 201},
            )
            secret = str((rotate or {}).get("signing_secret", "")).strip()
            if not secret:
                raise RuntimeError("Blooio webhook secret rotation did not return signing_secret")
            self._persist_state({"signing_secret": secret})

    def _verify_signature(self, raw_body: bytes, signature_header: str) -> None:
        if not self._webhook_secret:
            raise ValueError("Missing Blooio signing secret")
        parts = {}
        for chunk in signature_header.split(","):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            parts[key.strip()] = value.strip()
        timestamp = parts.get("t")
        provided = parts.get("v1")
        if not timestamp or not provided:
            raise ValueError("Invalid signature format")
        age = int(time.time()) - int(timestamp)
        if age > 300:
            raise ValueError("Webhook timestamp too old")
        signed_payload = f"{timestamp}.{raw_body.decode('utf-8')}"
        expected = hmac.new(
            self._webhook_secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(provided, expected):
            raise ValueError("Signature mismatch")

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        raw_body = await request.read()
        signature = request.headers.get("X-Blooio-Signature", "")
        if not signature:
            return web.Response(text="Missing signature", status=401)
        try:
            self._verify_signature(raw_body, signature)
        except Exception as exc:
            return web.Response(text=str(exc), status=401)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            return web.Response(text="Invalid JSON", status=400)

        event_payload = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload
        if not isinstance(event_payload, dict):
            return web.Response(text="Invalid payload", status=400)

        event_name = str(event_payload.get("event", "")).strip()
        if event_name != "message.received":
            self._record_control_event(event_name, event_payload)
            return web.Response(text="ok", status=200)

        message_event = await self._build_message_event(event_payload)
        if message_event is None:
            return web.Response(text="ignored", status=200)
        asyncio.create_task(self.handle_message(message_event))
        return web.Response(text="ok", status=200)

    def _record_control_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        message_id = str(payload.get("message_id", "")).strip()
        if message_id:
            self._status_by_message_id[message_id] = {
                "event": event_name,
                "payload": payload,
                "updated_at": time.time(),
            }
        if event_name == "message.failed":
            logger.warning("[%s] Blooio message failed: %s", self.name, payload)
        elif event_name:
            logger.debug("[%s] Blooio control event %s", self.name, event_name)

    async def _build_message_event(self, payload: Dict[str, Any]) -> Optional[MessageEvent]:
        sender = str(payload.get("sender", "")).strip()
        if not sender:
            return None
        internal_id = str(payload.get("internal_id", "")).strip()
        if internal_id and sender == internal_id:
            logger.debug("[%s] Ignoring self-sent Blooio inbound payload", self.name)
            return None
        if self._from_number and sender == self._from_number:
            logger.debug("[%s] Ignoring inbound echo from configured from_number", self.name)
            return None

        is_group = bool(payload.get("is_group"))
        chat_id = str(payload.get("group_id") if is_group else sender).strip()
        if not chat_id:
            return None
        chat_name = str(payload.get("group_name", "")).strip() or None
        chat_type = "group" if is_group else "dm"
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name or sender,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender,
        )

        attachments = payload.get("attachments") or []
        media_urls: List[str] = []
        media_types: List[str] = []
        message_type = MessageType.TEXT
        for attachment in attachments:
            cached = await self._cache_attachment(attachment)
            if not cached:
                continue
            media_urls.append(cached["path"])
            media_types.append(cached["media_type"])
            kind = cached["kind"]
            if kind == "image":
                message_type = MessageType.PHOTO
            elif kind == "audio" and message_type == MessageType.TEXT:
                message_type = MessageType.AUDIO
            elif kind == "video" and message_type == MessageType.TEXT:
                message_type = MessageType.VIDEO
            elif kind == "document" and message_type == MessageType.TEXT:
                message_type = MessageType.DOCUMENT

        text = str(payload.get("text", "") or "").strip()
        if not text and message_type == MessageType.VIDEO:
            text = "[The user sent a video attachment.]"

        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=payload,
            message_id=str(payload.get("message_id", "")).strip() or None,
            media_urls=media_urls,
            media_types=media_types,
        )

    async def _cache_attachment(self, attachment: Any) -> Optional[Dict[str, str]]:
        if isinstance(attachment, str):
            url = attachment
            name = Path(urlparse(url).path).name or "attachment"
        elif isinstance(attachment, dict):
            url = str(attachment.get("url", "")).strip()
            name = str(attachment.get("name") or Path(urlparse(url).path).name or "attachment").strip()
        else:
            return None
        if not url:
            return None
        filename = _sanitize_filename(name or "attachment")
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"Accept": "*/*"}) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Failed downloading Blooio attachment: {resp.status}")
                data = await resp.read()
                mime_type = (resp.headers.get("Content-Type", "").split(";")[0]).strip().lower()

        kind = _media_kind(filename, mime_type)
        ext = _guess_extension(filename, mime_type)
        if kind == "image":
            path = cache_image_from_bytes(data, ext=ext or ".jpg")
        elif kind == "audio":
            path = cache_audio_from_bytes(data, ext=ext or ".ogg")
        else:
            final_name = filename
            if ext and not final_name.endswith(ext):
                final_name = f"{final_name}{ext}"
            path = cache_document_from_bytes(data, final_name)
        final_mime = mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return {"path": path, "media_type": final_mime, "kind": kind}

    def format_message(self, content: str) -> str:
        content = _MARKDOWN_LINK_RE.sub(r"\1: \2", content)
        content = re.sub(r"\*\*(.+?)\*\*", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"__(.+?)__", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"_(.+?)_", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", content)
        content = content.replace("```", "")
        content = re.sub(r"`(.+?)`", r"\1", content)
        content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del reply_to  # Blooio currently exposes conversation-level replies only.
        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, max_length=self.MAX_MESSAGE_LENGTH)
        last_message_id = None
        for chunk in chunks:
            body: Dict[str, Any] = {"text": chunk}
            if self._from_number:
                body["from_number"] = self._from_number
            result = await self._send_message_request(chat_id, body)
            if not result.success:
                return result
            last_message_id = result.message_id
        return SendResult(success=True, message_id=last_message_id)

    async def _send_message_request(self, chat_id: str, body: Dict[str, Any]) -> SendResult:
        safe_chat_id = quote(str(chat_id), safe="")
        try:
            payload = await self._api_request(
                "POST",
                f"/chats/{safe_chat_id}/messages",
                json_body=body,
                allow_status={200, 202},
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))
        message_id = None
        if isinstance(payload, dict):
            message_id = str(payload.get("message_id", "")).strip() or None
            if message_id:
                self._status_by_message_id[message_id] = {
                    "event": "message.sent",
                    "payload": payload,
                    "updated_at": time.time(),
                }
        return SendResult(success=True, message_id=message_id, raw_response=payload)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        del metadata
        safe_chat_id = quote(str(chat_id), safe="")
        try:
            await self._api_request("POST", f"/chats/{safe_chat_id}/typing", allow_status={200})
        except Exception as exc:
            logger.debug("[%s] Failed to start Blooio typing indicator: %s", self.name, exc)

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        del metadata
        safe_chat_id = quote(str(chat_id), safe="")
        try:
            await self._api_request("DELETE", f"/chats/{safe_chat_id}/typing", allow_status={200})
        except Exception as exc:
            logger.debug("[%s] Failed to stop Blooio typing indicator: %s", self.name, exc)

    async def mark_read(self, chat_id: str, metadata=None) -> None:
        del metadata
        safe_chat_id = quote(str(chat_id), safe="")
        try:
            await self._api_request("POST", f"/chats/{safe_chat_id}/read", allow_status={200})
        except Exception as exc:
            logger.debug("[%s] Failed to mark Blooio chat as read: %s", self.name, exc)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del reply_to, metadata
        body: Dict[str, Any] = {"attachments": [image_url]}
        if caption:
            body["text"] = self.format_message(caption)
        if self._from_number:
            body["from_number"] = self._from_number
        return await self._send_message_request(chat_id, body)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del reply_to, kwargs
        staged = self._stage_file(file_path, file_name=file_name)
        body: Dict[str, Any] = {"attachments": [staged]}
        if caption:
            body["text"] = self.format_message(caption)
        if self._from_number:
            body["from_number"] = self._from_number
        return await self._send_message_request(chat_id, body)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_document(chat_id, image_path, caption=caption, reply_to=reply_to, **kwargs)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_document(chat_id, audio_path, caption=caption, reply_to=reply_to, **kwargs)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_document(chat_id, video_path, caption=caption, reply_to=reply_to, **kwargs)

    def _stage_file(self, file_path: str, *, file_name: Optional[str] = None) -> Dict[str, str]:
        source = Path(file_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        staged_dir = self._media_root / self._instance_id
        staged_dir.mkdir(parents=True, exist_ok=True)
        stored_key = secrets.token_hex(16)
        staged_path = staged_dir / stored_key
        shutil.copy2(source, staged_path)
        filename = _sanitize_filename(file_name or source.name)
        expires_at = int(time.time()) + _MEDIA_TTL_SECONDS
        token = self._build_media_token(stored_key, expires_at)
        url = f"{self._public_base_url}{self._media_path_prefix}/{token}/{quote(filename, safe='')}"
        return {"url": url, "name": filename}

    def _build_media_token(self, stored_key: str, expires_at: int) -> str:
        payload = f"{stored_key}:{expires_at}"
        signature = hmac.new(
            self._media_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]
        return f"{expires_at}.{stored_key}.{signature}"

    def _parse_media_token(self, token: str) -> tuple[str, int]:
        expires_raw, stored_key, provided = token.split(".", 2)
        expires_at = int(expires_raw)
        expected = hmac.new(
            self._media_secret.encode("utf-8"),
            f"{stored_key}:{expires_at}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]
        if expires_at < int(time.time()):
            raise PermissionError("Media URL expired")
        if not hmac.compare_digest(provided, expected):
            raise PermissionError("Invalid media token")
        return stored_key, expires_at

    async def _serve_media(self, request: web.Request) -> web.StreamResponse:
        token = request.match_info["token"]
        filename = _sanitize_filename(request.match_info["filename"])
        try:
            stored_key, _expires = self._parse_media_token(token)
        except Exception:
            raise web.HTTPForbidden()
        staged_path = self._media_root / self._instance_id / stored_key
        if not staged_path.exists():
            raise web.HTTPNotFound()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        headers = {"Content-Disposition": f'inline; filename="{filename}"'}
        response = web.FileResponse(path=staged_path, headers=headers)
        response.content_type = content_type
        return response

    def _cleanup_expired_media(self) -> None:
        cutoff = time.time() - _MEDIA_TTL_SECONDS
        base = self._media_root / self._instance_id
        if not base.exists():
            return
        for path in base.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = "dm" if _PHONE_OR_EMAIL_RE.match(str(chat_id).strip()) else "group"
        return {"name": str(chat_id), "type": chat_type, "chat_id": str(chat_id)}
