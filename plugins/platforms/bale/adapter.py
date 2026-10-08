"""Bale Messenger platform plugin.

Bale exposes a Telegram-compatible Bot API at https://tapi.bale.ai.  This
adapter implements Hermes' platform contract directly so Bale keeps its own
platform identity, authorization, sessions, configuration, and delivery hooks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from agent.secret_scope import UnscopedSecretError, get_secret
from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_media_bytes,
    classify_send_error,
    get_inbound_media_max_bytes,
    validate_inbound_media_size,
)

logger = logging.getLogger(__name__)

API_BASE = "https://tapi.bale.ai/bot"
FILE_BASE = "https://tapi.bale.ai/file/bot"
MAX_MESSAGE_LENGTH = 4096


class BaleAPIError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


def _secret(name: str, default: str = "") -> str:
    try:
        value = get_secret(name, default)
    except UnscopedSecretError:
        value = os.getenv(name, default)
    return str(value or "").strip()


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _token(config: Any) -> str:
    extra = getattr(config, "extra", {}) or {}
    return (
        str(getattr(config, "token", "") or "").strip()
        or str(extra.get("token", "") or "").strip()
        or _secret("BALE_BOT_TOKEN")
    )


def _home_chat(config: Any) -> str:
    home = getattr(config, "home_channel", None)
    if home is not None and getattr(home, "chat_id", None):
        return str(home.chat_id)
    extra = getattr(config, "extra", {}) or {}
    return os.getenv("BALE_HOME_CHANNEL", "").strip() or str(extra.get("home_channel", "") or "").strip()


def _send_error(exc: BaseException) -> SendResult:
    kind = classify_send_error(exc, str(exc))
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(exc, BaleAPIError):
        if exc.code == 429:
            kind = "rate_limited"
        elif exc.code is not None and exc.code >= 500:
            kind = "transient"
    return SendResult(
        success=False,
        error=str(exc),
        retryable=kind in {"transient", "rate_limited"},
        retry_after=retry_after,
        error_kind=kind,
    )


class BaleAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    splits_long_messages = True

    def __init__(self, config: Any, **_: Any) -> None:
        super().__init__(config=config, platform=Platform("bale"))
        extra = getattr(config, "extra", {}) or {}
        self.token = _token(config)
        self.api_base = (os.getenv("BALE_API_BASE_URL") or extra.get("base_url") or API_BASE).rstrip("/")
        self.file_base = (os.getenv("BALE_FILE_BASE_URL") or extra.get("base_file_url") or FILE_BASE).rstrip("/")
        try:
            timeout = int(os.getenv("BALE_POLL_TIMEOUT") or extra.get("poll_timeout", 25))
        except (TypeError, ValueError):
            timeout = 25
        self.poll_timeout = max(1, min(timeout, 50))
        env_mention = os.getenv("BALE_REQUIRE_MENTION")
        self.require_mention = _truthy(env_mention if env_mention is not None else extra.get("require_mention"), True)
        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None
        self._offset = 0
        self._bot_id = ""
        self._bot_username = ""

    def _url(self, method: str) -> str:
        return f"{self.api_base}{self.token}/{method}"

    def _client_factory(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=self.poll_timeout + 10, write=30, pool=10),
            follow_redirects=True,
        )

    async def _api(self, method: str, payload: dict | None = None, *, files: dict | None = None) -> dict:
        if not self.token:
            raise BaleAPIError("BALE_BOT_TOKEN is not configured")
        client = self._client
        own = client is None
        if client is None:
            client = self._client_factory()
        try:
            if files:
                data = {k: str(v) for k, v in (payload or {}).items() if v is not None}
                response = await client.post(self._url(method), data=data, files=files)
            else:
                response = await client.post(self._url(method), json=payload or {})
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError as exc:
                raise BaleAPIError(f"Bale {method} returned invalid JSON") from exc
            if not isinstance(body, dict):
                raise BaleAPIError(f"Bale {method} returned an invalid response")
            if body.get("ok", True) is False:
                params = body.get("parameters") or {}
                retry_after = params.get("retry_after") if isinstance(params, dict) else None
                try:
                    retry_after = float(retry_after) if retry_after is not None else None
                except (TypeError, ValueError):
                    retry_after = None
                try:
                    code = int(body.get("error_code")) if body.get("error_code") is not None else None
                except (TypeError, ValueError):
                    code = None
                raise BaleAPIError(str(body.get("description") or f"Bale {method} failed"), code, retry_after)
            return body
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise BaleAPIError(f"Bale {method} HTTP {code}", code) from exc
        except httpx.RequestError as exc:
            raise BaleAPIError(f"Bale {method} network error: {type(exc).__name__}", 503) from exc
        finally:
            if own:
                await client.aclose()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.token:
            self._set_fatal_error("config_missing", "BALE_BOT_TOKEN must be configured", retryable=False)
            return False
        if self._client is not None:
            await self.disconnect()
        identity = hashlib.sha256(self.token.encode()).hexdigest()
        if not self._acquire_platform_lock("bale_bot", identity, "Bale bot token"):
            return False
        self._client = self._client_factory()
        try:
            body = await self._api("getMe")
            me = body.get("result") or {}
            if not isinstance(me, dict) or me.get("id") is None:
                raise BaleAPIError("Bale getMe returned no bot identity")
            self._bot_id = str(me["id"])
            self._bot_username = str(me.get("username") or "").lstrip("@")
        except Exception as exc:
            await self._client.aclose()
            self._client = None
            self._release_platform_lock()
            permanent = isinstance(exc, BaleAPIError) and exc.code in {401, 403}
            self._set_fatal_error("auth_failed" if permanent else "connect_failed", str(exc), retryable=not permanent)
            return False
        self._mark_connected()
        self._wire_plugin_handlers(None)
        self._poll_task = asyncio.create_task(self._poll_loop(), name="bale-poll")
        logger.info("[Bale] Connected as %s", f"@{self._bot_username}" if self._bot_username else self._bot_id)
        return True

    async def disconnect(self) -> None:
        self._running = False
        current = asyncio.current_task()
        task, self._poll_task = self._poll_task, None
        if task is not None and task is not current and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._release_platform_lock()
        self._mark_disconnected()

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                body = await self._api("getUpdates", {
                    "offset": self._offset,
                    "timeout": self.poll_timeout,
                    "allowed_updates": ["message", "edited_message", "channel_post", "edited_channel_post"],
                })
                for update in body.get("result") or []:
                    if not isinstance(update, dict):
                        continue
                    if isinstance(update.get("update_id"), int):
                        self._offset = max(self._offset, update["update_id"] + 1)
                    try:
                        await self._dispatch_update(update)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("[Bale] Failed to dispatch update")
            except asyncio.CancelledError:
                raise
            except BaleAPIError as exc:
                if exc.code in {401, 403}:
                    self._set_fatal_error("auth_failed", str(exc), retryable=False)
                    await self._notify_fatal_error()
                    return
                logger.warning("[Bale] Poll failed: %s", exc)
                await asyncio.sleep(exc.retry_after or 2)
            except Exception as exc:
                logger.warning("[Bale] Poll failed: %s", exc)
                await asyncio.sleep(2)

    @staticmethod
    def _chat_type(raw: str) -> str:
        return "dm" if raw in {"private", "dm"} else ("channel" if raw == "channel" else "group")

    def _is_reply_to_bot(self, message: dict) -> bool:
        reply = message.get("reply_to_message") or {}
        sender = reply.get("from") or {}
        return bool(self._bot_id and str(sender.get("id") or "") == self._bot_id)

    def _mention_gate(self, text: str, message: dict, chat_type: str) -> tuple[bool, str]:
        if chat_type == "dm" or not self.require_mention or text.lstrip().startswith("/") or self._is_reply_to_bot(message):
            return True, text
        if not self._bot_username:
            return False, text
        pattern = re.compile(rf"(?<!\w)@{re.escape(self._bot_username)}\b", re.IGNORECASE)
        if not pattern.search(text):
            return False, text
        return True, pattern.sub("", text).strip()

    async def _download(self, file_path: str, media_type: str) -> bytes:
        if self._client is None:
            raise BaleAPIError("Bale client is not connected")
        limit = get_inbound_media_max_bytes()
        url = f"{self.file_base}{self.token}/{file_path.lstrip('/')}"
        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()
                raw_size = response.headers.get("content-length")
                if raw_size:
                    try:
                        validate_inbound_media_size(int(raw_size), media_type=media_type, max_bytes=limit)
                    except ValueError:
                        raise
                    except Exception:
                        pass
                chunks, total = [], 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    validate_inbound_media_size(total, media_type=media_type, max_bytes=limit)
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.HTTPStatusError as exc:
            raise BaleAPIError(f"Bale file download HTTP {exc.response.status_code}", exc.response.status_code) from exc
        except httpx.RequestError as exc:
            raise BaleAPIError(f"Bale file download network error: {type(exc).__name__}", 503) from exc

    async def _cache(self, file_id: str, filename: str, mime: str, kind: str):
        info = (await self._api("getFile", {"file_id": file_id})).get("result") or {}
        if not isinstance(info, dict) or not info.get("file_path"):
            return None
        data = await self._download(str(info["file_path"]), kind)
        return cache_media_bytes(data, filename=filename, mime_type=mime, default_kind=kind)

    async def _dispatch_update(self, update: dict) -> None:
        message = next((update.get(k) for k in ("message", "edited_message", "channel_post", "edited_channel_post") if isinstance(update.get(k), dict)), None)
        if message is None:
            return
        sender = message.get("from") or {}
        if sender.get("is_bot"):
            return
        chat = message.get("chat") or {}
        if chat.get("id") is None:
            return
        chat_id = str(chat["id"])
        chat_type = self._chat_type(str(chat.get("type") or "private"))
        text = str(message.get("text") or message.get("caption") or "")
        allowed, text = self._mention_gate(text, message, chat_type)
        if not allowed:
            return
        media_paths: list[str] = []
        media_types: list[str] = []
        message_type = MessageType.COMMAND if text.lstrip().startswith("/") else MessageType.TEXT

        media = None
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            photo = photos[-1]
            if isinstance(photo, dict) and photo.get("file_id"):
                media = await self._cache(str(photo["file_id"]), "photo.jpg", "image/jpeg", "image")
                message_type = MessageType.PHOTO
        elif isinstance(message.get("voice"), dict) and message["voice"].get("file_id"):
            item = message["voice"]
            media = await self._cache(str(item["file_id"]), "voice.ogg", str(item.get("mime_type") or "audio/ogg"), "audio")
            message_type = MessageType.VOICE
        elif isinstance(message.get("audio"), dict) and message["audio"].get("file_id"):
            item = message["audio"]
            media = await self._cache(str(item["file_id"]), str(item.get("file_name") or "audio.mp3"), str(item.get("mime_type") or "audio/mpeg"), "audio")
            message_type = MessageType.AUDIO
        elif isinstance(message.get("video"), dict) and message["video"].get("file_id"):
            item = message["video"]
            media = await self._cache(str(item["file_id"]), str(item.get("file_name") or "video.mp4"), str(item.get("mime_type") or "video/mp4"), "video")
            message_type = MessageType.VIDEO
        elif isinstance(message.get("document"), dict) and message["document"].get("file_id"):
            item = message["document"]
            media = await self._cache(str(item["file_id"]), str(item.get("file_name") or "document.bin"), str(item.get("mime_type") or "application/octet-stream"), "document")
            message_type = MessageType.DOCUMENT
        elif isinstance(message.get("location"), dict):
            loc = message["location"]
            text = text or f"Location: {loc.get('latitude')}, {loc.get('longitude')}"
            message_type = MessageType.LOCATION

        if media is not None:
            media_paths.append(media.path)
            media_types.append(media.media_type)
            if not text:
                text = media.context_note()

        first = str(sender.get("first_name") or "").strip()
        last = str(sender.get("last_name") or "").strip()
        user_name = " ".join(p for p in (first, last) if p) or str(sender.get("username") or "") or None
        source = self.build_source(
            chat_id=chat_id,
            chat_name=str(chat.get("title") or chat.get("username") or "") or None,
            chat_type=chat_type,
            user_id=str(sender.get("id")) if sender.get("id") is not None else None,
            user_name=user_name,
            thread_id=str(message.get("message_thread_id")) if message.get("message_thread_id") is not None else None,
            message_id=str(message.get("message_id")) if message.get("message_id") is not None else None,
        )
        reply = message.get("reply_to_message") or {}
        reply_sender = reply.get("from") or {}
        event = MessageEvent(
            text=text,
            message_type=message_type,
            user_id=source.user_id,
            user_name=user_name,
            source=source,
            raw_message=message,
            message_id=str(message.get("message_id")) if message.get("message_id") is not None else None,
            platform_update_id=update.get("update_id") if isinstance(update.get("update_id"), int) else None,
            media_urls=media_paths,
            media_types=media_types,
            reply_to_message_id=str(reply.get("message_id")) if reply.get("message_id") is not None else None,
            reply_to_text=str(reply.get("text") or reply.get("caption") or "") or None,
            reply_to_author_id=str(reply_sender.get("id")) if reply_sender.get("id") is not None else None,
            reply_to_author_name=str(reply_sender.get("username") or reply_sender.get("first_name") or "") or None,
            reply_to_is_own_message=bool(self._bot_id and str(reply_sender.get("id") or "") == self._bot_id),
            timestamp=datetime.fromtimestamp(message["date"]) if isinstance(message.get("date"), (int, float)) else datetime.now(),
        )
        await self.handle_message(event)

    @staticmethod
    def _route(reply_to: str | None, metadata: dict | None) -> dict:
        out: dict[str, Any] = {}
        if reply_to:
            out["reply_to_message_id"] = reply_to
        if metadata and metadata.get("thread_id") is not None:
            out["message_thread_id"] = str(metadata["thread_id"])
        return out

    async def send(self, chat_id: str, content: str, reply_to: str | None = None, metadata: dict | None = None) -> SendResult:
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        ids: list[str] = []
        try:
            for index, chunk in enumerate(chunks):
                body = await self._api("sendMessage", {"chat_id": str(chat_id), "text": chunk, **self._route(reply_to if index == 0 else None, metadata)})
                result = body.get("result") or {}
                if isinstance(result, dict) and result.get("message_id") is not None:
                    ids.append(str(result["message_id"]))
            return SendResult(success=True, message_id=ids[-1] if ids else None, continuation_message_ids=tuple(ids[:-1]))
        except Exception as exc:
            return _send_error(exc)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        try:
            body = await self._api("editMessageText", {"chat_id": str(chat_id), "message_id": str(message_id), "text": content[:self.MAX_MESSAGE_LENGTH]})
            result = body.get("result") or {}
            returned = str(result.get("message_id")) if isinstance(result, dict) and result.get("message_id") is not None else str(message_id)
            return SendResult(success=True, message_id=returned)
        except Exception as exc:
            return _send_error(exc)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        try:
            await self._api("deleteMessage", {"chat_id": str(chat_id), "message_id": str(message_id)})
            return True
        except Exception:
            return False

    async def send_typing(self, chat_id: str, metadata: dict | None = None) -> None:
        try:
            await self._api("sendChatAction", {"chat_id": str(chat_id), "action": "typing", **self._route(None, metadata)})
        except Exception:
            logger.debug("[Bale] sendChatAction failed", exc_info=True)

    async def _send_file(self, method: str, field: str, chat_id: str, file_path: str, *, caption: str | None = None, file_name: str | None = None, reply_to: str | None = None, metadata: dict | None = None) -> SendResult:
        path = Path(file_path)
        if not path.is_file():
            return SendResult(success=False, error=f"File not found: {path.name}", error_kind="not_found")
        mime = mimetypes.guess_type(file_name or path.name)[0] or "application/octet-stream"
        payload = {"chat_id": str(chat_id), **self._route(reply_to, metadata)}
        if caption:
            payload["caption"] = caption[:1024]
        try:
            with path.open("rb") as handle:
                body = await self._api(method, payload, files={field: (file_name or path.name, handle, mime)})
            result = body.get("result") or {}
            mid = str(result.get("message_id")) if isinstance(result, dict) and result.get("message_id") is not None else None
            return SendResult(success=True, message_id=mid)
        except Exception as exc:
            return _send_error(exc)

    async def send_image(self, chat_id: str, image_url: str, caption: str | None = None, reply_to: str | None = None, metadata: dict | None = None) -> SendResult:
        try:
            body = await self._api("sendPhoto", {"chat_id": str(chat_id), "photo": image_url, "caption": (caption or "")[:1024], **self._route(reply_to, metadata)})
            result = body.get("result") or {}
            mid = str(result.get("message_id")) if isinstance(result, dict) and result.get("message_id") is not None else None
            return SendResult(success=True, message_id=mid)
        except Exception as exc:
            return _send_error(exc)

    async def send_image_file(self, chat_id: str, image_path: str, caption: str | None = None, reply_to: str | None = None, metadata: dict | None = None, **kwargs: Any) -> SendResult:
        return await self._send_file("sendPhoto", "photo", chat_id, image_path, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_voice(self, chat_id: str, audio_path: str, caption: str | None = None, reply_to: str | None = None, metadata: dict | None = None, **kwargs: Any) -> SendResult:
        ext = Path(audio_path).suffix.lower()
        if ext in {".ogg", ".opus"}:
            return await self._send_file("sendVoice", "voice", chat_id, audio_path, caption=caption, reply_to=reply_to, metadata=metadata)
        if ext in {".mp3", ".m4a"}:
            return await self._send_file("sendAudio", "audio", chat_id, audio_path, caption=caption, reply_to=reply_to, metadata=metadata)
        return await self.send_document(chat_id, audio_path, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_video(self, chat_id: str, video_path: str, caption: str | None = None, reply_to: str | None = None, metadata: dict | None = None, **kwargs: Any) -> SendResult:
        return await self._send_file("sendVideo", "video", chat_id, video_path, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_document(self, chat_id: str, file_path: str, caption: str | None = None, file_name: str | None = None, reply_to: str | None = None, metadata: dict | None = None, **kwargs: Any) -> SendResult:
        return await self._send_file("sendDocument", "document", chat_id, file_path, caption=caption, file_name=file_name, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            chat = (await self._api("getChat", {"chat_id": str(chat_id)})).get("result") or {}
            if not isinstance(chat, dict):
                chat = {}
            name = str(chat.get("title") or chat.get("username") or chat.get("first_name") or chat_id)
            return {"name": name, "type": self._chat_type(str(chat.get("type") or "private")), "id": str(chat.get("id") or chat_id)}
        except Exception as exc:
            return {"name": str(chat_id), "type": "dm", "id": str(chat_id), "error": str(exc)}


def check_requirements() -> bool:
    """Passive probe. httpx is already a Hermes core dependency."""
    return True


def validate_config(config: Any) -> bool:
    return bool(_token(config))


def is_connected(config: Any) -> bool:
    return validate_config(config)


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _apply_yaml_config(yaml_cfg: dict, bale_cfg: dict) -> Optional[dict]:
    if not isinstance(bale_cfg, dict):
        return None
    extra = dict(bale_cfg.get("extra") or {})
    for key in ("base_url", "base_file_url", "require_mention", "poll_timeout", "home_channel"):
        if key in bale_cfg:
            extra.setdefault(key, bale_cfg[key])
    token = bale_cfg.get("bot_token") or bale_cfg.get("token")
    if token:
        extra.setdefault("token", str(token).strip())
        os.environ.setdefault("BALE_BOT_TOKEN", str(token).strip())
    mapping = {
        "allowed_users": "BALE_ALLOWED_USERS",
        "allow_all_users": "BALE_ALLOW_ALL_USERS",
        "require_mention": "BALE_REQUIRE_MENTION",
        "home_channel": "BALE_HOME_CHANNEL",
        "base_url": "BALE_API_BASE_URL",
        "base_file_url": "BALE_FILE_BASE_URL",
        "poll_timeout": "BALE_POLL_TIMEOUT",
    }
    for key, env_name in mapping.items():
        if key in bale_cfg and not os.getenv(env_name):
            value = _env_value(bale_cfg[key])
            if value:
                os.environ[env_name] = value
    return extra


def _env_enablement() -> Optional[dict]:
    token = _secret("BALE_BOT_TOKEN")
    if not token:
        return None
    seed: dict[str, Any] = {"token": token}
    if os.getenv("BALE_API_BASE_URL"):
        seed["base_url"] = os.environ["BALE_API_BASE_URL"]
    if os.getenv("BALE_FILE_BASE_URL"):
        seed["base_file_url"] = os.environ["BALE_FILE_BASE_URL"]
    if os.getenv("BALE_REQUIRE_MENTION") is not None:
        seed["require_mention"] = _truthy(os.getenv("BALE_REQUIRE_MENTION"))
    if os.getenv("BALE_POLL_TIMEOUT"):
        try:
            seed["poll_timeout"] = int(os.environ["BALE_POLL_TIMEOUT"])
        except ValueError:
            pass
    if os.getenv("BALE_HOME_CHANNEL"):
        seed["home_channel"] = {"chat_id": os.environ["BALE_HOME_CHANNEL"], "name": "Home"}
    return seed


async def _standalone_send(pconfig: Any, chat_id: str, message: str, *, thread_id: str | None = None, media_files: list | None = None, force_document: bool = False) -> dict:
    target = str(chat_id or "").strip() or _home_chat(pconfig)
    if not target:
        return {"error": "Bale standalone send: no target chat (set BALE_HOME_CHANNEL)"}
    if not _token(pconfig):
        return {"error": "Bale standalone send: BALE_BOT_TOKEN is not configured"}
    adapter = BaleAdapter(pconfig)
    adapter._client = adapter._client_factory()
    metadata = {"thread_id": str(thread_id)} if thread_id is not None else None
    last_id = None
    try:
        if message:
            result = await adapter.send(target, message, metadata=metadata)
            if not result.success:
                return {"error": result.error or "Bale text delivery failed"}
            last_id = result.message_id
        for item in media_files or []:
            if isinstance(item, (tuple, list)):
                path, is_voice = str(item[0]), bool(item[1]) if len(item) > 1 else False
            else:
                path, is_voice = str(item), False
            ext = Path(path).suffix.lower()
            if force_document:
                result = await adapter.send_document(target, path, metadata=metadata)
            elif is_voice or ext in {".ogg", ".opus", ".mp3", ".m4a"}:
                result = await adapter.send_voice(target, path, metadata=metadata)
            elif ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                result = await adapter.send_image_file(target, path, metadata=metadata)
            elif ext in {".mp4", ".mov", ".webm", ".mkv"}:
                result = await adapter.send_video(target, path, metadata=metadata)
            else:
                result = await adapter.send_document(target, path, metadata=metadata)
            if not result.success:
                return {"error": result.error or f"Bale media delivery failed: {Path(path).name}"}
            last_id = result.message_id or last_id
        return {"success": True, "message_id": last_id, "platform": "bale"}
    finally:
        await adapter._client.aclose()
        adapter._client = None


def register(ctx) -> None:
    ctx.register_platform(
        name="bale",
        label="Bale",
        adapter_factory=lambda cfg: BaleAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["BALE_BOT_TOKEN"],
        apply_yaml_config_fn=_apply_yaml_config,
        env_enablement_fn=_env_enablement,
        allowed_users_env="BALE_ALLOWED_USERS",
        allow_all_env="BALE_ALLOW_ALL_USERS",
        cron_deliver_env_var="BALE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🔵",
        platform_hint="Bale chat ID",
    )
