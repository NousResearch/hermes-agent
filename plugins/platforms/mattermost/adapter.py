"""Mattermost gateway adapter — REST API v4 + WebSocket via aiohttp (no Mattermost SDK).

Environment variables:
    MATTERMOST_URL              Server URL (e.g. https://mm.example.com)
    MATTERMOST_TOKEN            Bot token or personal-access token
    MATTERMOST_ALLOWED_USERS    Comma-separated user IDs
    MATTERMOST_HOME_CHANNEL     Channel ID for cron/notification delivery
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from urllib.request import url2pathname
from typing import Any, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import gateway_trust_env, BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret, profile_scoped as _profile_scoped_config_load

logger = logging.getLogger(__name__)

_Metadata = Optional[Dict[str, Any]]

# Server default is 16383, but 4000 is the practical limit for readable messages.
MAX_POST_LENGTH = 4000


def _resolve_max_post_length(extra: Optional[Dict[str, Any]] = None) -> int:
    """Resolve the profile's YAML-only outbound limit; never read process env."""
    raw = (extra or {}).get("max_post_length")
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return MAX_POST_LENGTH
    try:
        limit = int(raw)
    except ValueError:
        return MAX_POST_LENGTH
    return min(limit, 16383) if limit >= 500 else MAX_POST_LENGTH

# Channel type codes returned by the Mattermost API ("P" private → treat as group).
_CHANNEL_TYPE_MAP = {"D": "dm", "G": "group", "P": "group", "O": "channel"}

_MATTERMOST_DISABLE_MENTIONS_PROPS = {"disable_mentions": True}

_RECONNECT_BASE_DELAY, _RECONNECT_MAX_DELAY, _RECONNECT_JITTER = 2.0, 60.0, 0.2  # exponential backoff

_POST_WITH_FILE_ERROR = "Failed to post with file"
_POST_DELIVERY_UNCERTAIN = "Mattermost content delivery uncertain"
_POST_DELIVERY_PARTIAL = "Mattermost content partially delivered; replay suppressed"
_MEDIA_MSG_TYPES = (("image/", MessageType.PHOTO), ("audio/", MessageType.VOICE))  # first match wins
_INBOUND_CACHE_EXT = {"image/": ".png", "audio/": ".ogg"}  # mime prefix → default extension for cached media


def _with_mentions_disabled(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a post payload that prevents Mattermost from firing mentions."""
    props, disable = payload.get("props"), _MATTERMOST_DISABLE_MENTIONS_PROPS
    payload["props"] = {**props, **disable} if isinstance(props, dict) else dict(disable)
    return payload


def _channel_id_set(raw: Any) -> set:
    """Parse a list or comma-separated string of channel IDs into a stripped set."""
    items = raw if isinstance(raw, list) else str(raw).split(",")
    return {str(c).strip() for c in items if str(c).strip()}


def _csv(value: Any) -> str:
    return ",".join(str(v) for v in value) if isinstance(value, list) else str(value)


def _post_result(data: Dict[str, Any], error: str) -> SendResult:
    data = data or {}
    ids = data.get("message_ids") or ([data["id"]] if data.get("id") else [])
    success = bool(data.get("id"))
    if data.get("_delivery_uncertain"):
        error = _POST_DELIVERY_UNCERTAIN
    return SendResult(success=success, message_id=ids[-1] if ids else None,
                      continuation_message_ids=tuple(ids[:-1]), error=None if success else error)


def _post_was_rejected(data: Dict[str, Any]) -> bool:
    """Only a response-bound rejection with no acknowledged or uncertain content permits recovery."""
    return (data.get("_post_rejection_status") in {400, 403, 404, 413, 422}
            and not data.get("id") and not data.get("message_ids") and not data.get("_delivery_uncertain"))


def _visible_file_post_name(name: Any, max_len: int = 160) -> str:
    """Compact filename label that cannot create an accidental mention."""
    label = re.sub(r"\s+", " ", str(name or "")).strip().replace("@", "@\u200b")
    return label if len(label) <= max_len else f"{label[:max_len - 1]}…"


def _file_post_message(caption: Optional[str], filenames: List[str]) -> str:
    """Preserve explicit captions; only fallback filenames are normalized."""
    if caption:
        return caption
    # Five generated labels must fit even the smallest supported post budget (500).
    # A fixed floor keeps file delivery independent of configurable-limit support.
    name_budget = min(160, 501 // max(1, len(filenames)) - 3)
    labels = [_visible_file_post_name(name, name_budget) for name in filenames]
    return "\n".join(f"📎 {label}" for label in labels if label) or "📎 Attachment"


def _media_delivery_receipt(chat_id, total, delivered, message_ids, error=None):
    """Direct-send success requires all requested work and an acknowledged post."""
    if delivered != total and not error:
        error = "Not all requested Mattermost media were delivered"
    if not message_ids and not error:
        error = "No deliverable Mattermost media"
    result = {"success": not error, "platform": "mattermost", "chat_id": chat_id,
              "message_id": message_ids[-1] if message_ids else None,
              "message_ids": list(message_ids), "total_media": total, "delivered_media": delivered,
              "failed_media": total - delivered,
              "media_delivered": bool(delivered), "partial_failure": bool(error and message_ids)}
    if error:
        result["error"] = error
    return result


def _media_path(descriptor):
    if isinstance(descriptor, dict):
        return descriptor.get("path")
    if isinstance(descriptor, (tuple, list)):
        return descriptor[0] if descriptor else None
    return descriptor


def _url_filename(url: str, fallback: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0] or fallback


def _url_and_token(config) -> Tuple[str, str]:
    """(server URL, token): ``config`` first, MATTERMOST_URL / MATTERMOST_TOKEN env fallback."""
    extra = getattr(config, "extra", {}) or {}
    return (extra.get("url") or _get_scoped_secret("MATTERMOST_URL", ""),
            getattr(config, "token", None) or _get_scoped_secret("MATTERMOST_TOKEN", ""))


def check_mattermost_requirements() -> bool:
    """Return True if the Mattermost adapter runtime dependency is available."""
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Mattermost: aiohttp not installed")
        return False


def validate_mattermost_config(config: PlatformConfig) -> bool:
    """Return True when Mattermost has enough config to connect."""
    url, token = _url_and_token(config)
    if not token.strip():
        logger.debug("Mattermost: MATTERMOST_TOKEN not set")
        return False
    if not url.strip():
        logger.warning("Mattermost: MATTERMOST_URL not set")
        return False
    return True


class MattermostAdapter(BasePlatformAdapter):
    """Gateway adapter for Mattermost (self-hosted or cloud)."""

    splits_long_messages = True  # send() chunks at the profile-local limit

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATTERMOST)
        self.MAX_MESSAGE_LENGTH = _resolve_max_post_length(config.extra)
        self._base_url, self._token = _url_and_token(config)
        self._base_url = self._base_url.rstrip("/")
        self._bot_user_id = self._bot_username = ""
        self._session: Any = None  # aiohttp.ClientSession
        self._ws: Any = None  # aiohttp.ClientWebSocketResponse
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._closing = False
        # Reply mode: "thread" to nest replies, "off" for flat messages.
        self._reply_mode: str = (
            config.extra.get("reply_mode", "") or _get_scoped_secret("MATTERMOST_REPLY_MODE", "off")).lower()
        self._last_post_status: Optional[int] = None  # POST-only, read by the broken-thread-root fallback
        self._last_post_error: str = ""
        self._dedup = MessageDeduplicator()

    # --- HTTP helpers ---

    def _headers(self) -> Dict[str, str]:
        return {**self._auth_header(), "Content-Type": "application/json"}

    def _auth_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def _api(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """{method} /api/v4/{path}; POST also records _last_post_status/_last_post_error."""
        import aiohttp
        if ".." in path:
            logger.error("MM API path traversal blocked: %s", path)
            return {}
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        is_post = method == "POST"
        if is_post:
            self._last_post_status, self._last_post_error = None, ""
        kwargs: Dict[str, Any] = {"headers": self._headers()}
        if payload is not None:
            kwargs["json"] = payload
        if method != "PUT":  # PUT relies on the session default timeout
            kwargs["timeout"] = aiohttp.ClientTimeout(total=30)
        # Another POST may finish while this request's response body is pending.
        response_status: Optional[int] = None
        try:
            async with getattr(self._session, method.lower())(url, **kwargs) as resp:
                response_status = resp.status
                if is_post:
                    self._last_post_status = resp.status
                if resp.status >= 400:
                    body = await resp.text()
                    if is_post:
                        self._last_post_error = body or ""
                    logger.error("MM API %s %s → %s: %s", method, path, resp.status, body[:200])
                    if is_post and path == "posts" and resp.status >= 500:
                        # A server/proxy failure does not establish that the POST was rejected.
                        return {"_delivery_uncertain": True}
                    return {"_post_rejection_status": resp.status} if is_post and path == "posts" else {}
                try:
                    data = await resp.json()
                except ValueError:
                    if is_post and path == "posts":
                        return {"_delivery_uncertain": True}
                    raise
                if is_post and path == "posts":
                    # Only a usable post ID acknowledges delivery; malformed success
                    # bodies cannot authorize replay of a non-idempotent request.
                    if not isinstance(data, dict) or not isinstance(data.get("id"), str) or not data["id"].strip():
                        return {"_delivery_uncertain": True}
                    # Internal evidence comes only from the HTTP status, never response JSON.
                    data.pop("_post_rejection_status", None)
                return data
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if is_post:
                self._last_post_error = str(exc)
            logger.error("MM API %s %s network error: %s", method, path, exc)
            if isinstance(exc, aiohttp.ClientConnectorError):
                return {}
            if is_post and path == "posts":
                if response_status is None or response_status < 400 or response_status >= 500:
                    return {"_delivery_uncertain": True}
                return {}
            if isinstance(exc, asyncio.TimeoutError):
                raise
            return {}

    async def _api_get(self, path: str) -> Dict[str, Any]:
        return await self._api("GET", path)

    async def _api_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._api("POST", path, payload)

    @staticmethod
    def _is_timeout_error(error: Optional[str]) -> bool:
        # The gateway's timeout guard also prevents replay after a lost POST acknowledgement.
        return error in {_POST_DELIVERY_UNCERTAIN, _POST_DELIVERY_PARTIAL} or BasePlatformAdapter._is_timeout_error(error)

    def _last_post_failure_is_broken_thread_root(self) -> bool:
        """Return True only for clear invalid/missing Mattermost thread roots."""
        body = (self._last_post_error or "").lower()
        if self._last_post_status not in {400, 404} or not body:
            return False
        return (any(marker in body for marker in ("root_id", "rootid", "root id", "thread", "post"))
                and any(marker in body for marker in ("invalid", "not found", "does not exist", "missing")))

    async def _post_preserving_thread(
        self, chat_id: str, payload: Dict[str, Any], metadata: _Metadata) -> Dict[str, Any]:
        """Post once, optionally falling back flat for final notify content."""
        data = await self._api_post("posts", payload)
        if ((data and not _post_was_rejected(data)) or "root_id" not in payload
                or not (isinstance(metadata, dict) and metadata.get("notify"))
                or not self._last_post_failure_is_broken_thread_root()):
            return data
        flat_payload = {k: v for k, v in payload.items() if k != "root_id"}
        warning = "⚠️ Mattermost thread delivery failed; posting final reply in channel.\n\n"
        message = str(flat_payload.get("message") or "")
        logger.warning("Mattermost: falling back to flat channel delivery for notify-worthy post in %s", chat_id)
        if len(warning) + len(message) > self.MAX_MESSAGE_LENGTH:
            # The answer/caption already owns its budget. Keep it intact and attach files only once.
            notice = _with_mentions_disabled({"channel_id": chat_id, "message": warning.rstrip()})
            notice_data = await self._api_post("posts", notice)
            if not notice_data.get("id"):
                return notice_data
            ids = [str(notice_data["id"])]
            try:
                data = await self._api_post("posts", flat_payload) or {}
                return {**data, "message_ids": ids + ([str(data["id"])] if data.get("id") else [])}
            except asyncio.CancelledError:
                raise
            except Exception:
                # A lost content response must not erase the acknowledged warning or cause replay.
                logger.warning("Mattermost: content post failed after acknowledged fallback notice")
                return {"message_ids": ids, "_delivery_uncertain": True}
        else:
            flat_payload["message"] = warning + message
        return await self._api_post("posts", flat_payload)

    async def _post_message(self, chat_id: str, message: str, reply_to: Optional[str], metadata: _Metadata,
                            file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build a mentions-disabled post payload (+ optional root_id) and post it."""
        base: Dict[str, Any] = {"channel_id": chat_id, "message": message}
        if file_ids is not None:
            base["file_ids"] = file_ids
        payload = _with_mentions_disabled(base)
        if self._reply_mode == "thread" or (metadata or {}).get("mattermost_explicit_thread"):
            # root_id from reply_to, else metadata["thread_id"]/["root_id"], resolved to the true thread root.
            candidate = reply_to or (
                isinstance(metadata, dict) and (metadata.get("thread_id") or metadata.get("root_id")))
            if candidate:
                payload["root_id"] = await self._resolve_root_id(str(candidate))
        # Slice captions without formatting, trimming, or synthetic chunk markers.
        # Attach files to the final slice only; earlier slices retain their receipts.
        chunks = [message[i:i + self.MAX_MESSAGE_LENGTH] for i in range(0, len(message), self.MAX_MESSAGE_LENGTH)] or [""]
        ids = []
        data: Dict[str, Any] = {}
        for index, chunk in enumerate(chunks):
            chunk_payload = {**payload, "message": chunk}
            if index < len(chunks) - 1:
                chunk_payload.pop("file_ids", None)
            try:
                data = await self._post_preserving_thread(chat_id, chunk_payload, metadata) or {}
            except asyncio.CancelledError:
                raise
            except Exception:
                if not ids:
                    raise
                # Replaying the whole caption would repeat acknowledged slices.
                return {"message_ids": ids, "_delivery_uncertain": True}
            ids.extend(data.get("message_ids") or ([str(data["id"])] if data.get("id") else []))
            if not data.get("id"):
                break
        return {**data, "message_ids": ids}

    async def _post_with_file(self, chat_id: str, file_id: str, caption: Optional[str], reply_to: Optional[str],
                              metadata: _Metadata) -> SendResult:
        return _post_result(await self._post_message(chat_id, caption or "", reply_to, metadata, [file_id]),
                            _POST_WITH_FILE_ERROR)

    async def _upload_file(self, channel_id: str, file_data: bytes, filename: str,
                           content_type: str = "application/octet-stream") -> Optional[str]:
        """Upload a file and return its file ID, or None on failure."""
        import aiohttp
        form = aiohttp.FormData()
        form.add_field("channel_id", channel_id)
        form.add_field("files", file_data, filename=filename, content_type=content_type)
        async with self._session.post(f"{self._base_url}/api/v4/files", headers=self._auth_header(), data=form,
                                      timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.error("MM file upload → %s: %s", resp.status, body[:200])
                return None
            infos = (await resp.json()).get("file_infos", [])
            return infos[0]["id"] if infos else None

    # --- Required overrides ---

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Mattermost and start the WebSocket listener."""
        import aiohttp
        if not self._base_url or not self._token:
            logger.error("Mattermost: URL or token not configured")
            return False
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=gateway_trust_env())
        self._closing = False
        me = await self._api_get("users/me")
        if not me or "id" not in me:
            logger.error("Mattermost: failed to authenticate — check MATTERMOST_TOKEN and MATTERMOST_URL")
            await self._session.close()
            return False
        self._bot_user_id, self._bot_username = me["id"], me.get("username", "")
        logger.info(
            "Mattermost: authenticated as @%s (%s) on %s", self._bot_username, self._bot_user_id, self._base_url)
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        self._closing = True
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._ws_task
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Mattermost: disconnected")

    async def _resolve_root_id(self, post_id: str) -> str:
        """Resolve a post_id to its thread root_id (a reply's own ID causes "Invalid RootId parameter")."""
        if not post_id:
            return post_id
        data = await self._api_get(f"posts/{post_id}")
        return data["root_id"] if data and data.get("root_id") else post_id

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        """Send a message (or multiple chunks) to a channel; reply_to / metadata["thread_id"] is the root post."""
        if not content:
            return SendResult(success=True)
        ids = []
        data: Dict[str, Any] = {}
        for chunk in self.truncate_message(self.format_message(content), self.MAX_MESSAGE_LENGTH):
            try:
                data = await self._post_message(chat_id, chunk, reply_to, metadata) or {}
            except asyncio.CancelledError:
                raise
            except Exception:
                if not ids:
                    raise
                data = {"_delivery_uncertain": True}
            ids.extend(data.get("message_ids") or ([str(data["id"])] if data.get("id") else []))
            if not data.get("id"):
                break
        # The base's no-replay guard also applies to acknowledged partial text,
        # even if its last POST was definitively rejected rather than uncertain.
        result = _post_result({**data, "message_ids": ids},
                              _POST_DELIVERY_PARTIAL if ids else "Failed to create post")
        result.raw_response = {**data, "message_ids": ids}
        return result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        data = await self._api_get(f"channels/{chat_id}")
        if not data:
            return {"name": chat_id, "type": "channel"}
        return {"name": data.get("display_name") or data.get("name") or chat_id,
                "type": _CHANNEL_TYPE_MAP.get(data.get("type", "O"), "channel")}

    # --- Optional overrides ---

    async def send_typing(self, chat_id: str, metadata: _Metadata = None) -> None:
        await self._api_post(f"users/{self._bot_user_id}/typing", {"channel_id": chat_id})

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
                           metadata: _Metadata = None) -> SendResult:
        # Slice the source before formatting: offsets must stay in the caller's
        # domain, including whitespace, image markdown and literal chunk numbers.
        chunks = ([content] if len(self.format_message(content)) <= self.MAX_MESSAGE_LENGTH else
                  [content[i:i + self.MAX_MESSAGE_LENGTH] for i in range(0, len(content), self.MAX_MESSAGE_LENGTH)])
        payload = _with_mentions_disabled({"message": self.format_message(chunks[0])})
        data: Dict[str, Any] = await self._api("PUT", f"posts/{message_id}/patch", payload) or {}
        if not data.get("id"):
            return _post_result(data, "Failed to edit post")
        ids = [str(data["id"])]
        consumed = len(chunks[0])
        for chunk in chunks[1:]:
            try:
                data = await self._post_message(chat_id, self.format_message(chunk), message_id, metadata) or {}
            except asyncio.CancelledError:
                raise
            except Exception:
                data = {"_delivery_uncertain": self._last_post_status is None or self._last_post_status < 400}
            ids.extend(data.get("message_ids") or ([str(data["id"])] if data.get("id") else []))
            if not data.get("id"):
                result = _post_result({**data, "message_ids": ids}, "Failed to create edit continuation")
                result.raw_response = {**data, "message_ids": ids, "partial_overflow": True,
                                       "last_message_id": ids[-1], "delivered_prefix": content[:consumed]}
                return result
            consumed += len(chunk)
        return _post_result({**data, "message_ids": ids}, "Failed to edit post")

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_url_as_file(chat_id, image_url, caption, reply_to, "image", metadata)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_local_file(chat_id, image_path, caption, reply_to, metadata=metadata)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_local_file(chat_id, file_path, caption, reply_to, file_name, metadata)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_local_file(chat_id, audio_path, caption, reply_to, metadata=metadata)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_local_file(chat_id, video_path, caption, reply_to, metadata=metadata)

    def format_message(self, content: str) -> str:
        """Mattermost renders standard Markdown; reduce ![alt](url) to the bare URL (inline preview)."""
        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)

    # --- File helpers ---

    async def _send_url_as_file(self, chat_id: str, url: str, caption: Optional[str], reply_to: Optional[str],
                                kind: str = "file", metadata: _Metadata = None) -> SendResult:
        """Download a URL and upload it as a file attachment (text fallback with the URL on failure)."""
        from tools.url_safety import is_safe_url

        async def fallback() -> SendResult:
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        if not is_safe_url(url):
            logger.warning("Mattermost: blocked unsafe URL (SSRF protection)")
            return await fallback()
        import aiohttp
        for attempt in range(3):  # retry 5xx/429 and network errors twice with linear backoff
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if (resp.status >= 500 or resp.status == 429) and attempt < 2:
                        logger.debug("Mattermost download retry %d/2 for %s (status %d)",
                                     attempt + 1, url[:80], resp.status)
                    elif resp.status >= 400:
                        return await fallback()
                    else:
                        file_data, ct = await resp.read(), resp.content_type or "application/octet-stream"
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == 2:
                    logger.warning("Mattermost: failed to download %s after %d attempts: %s", url, attempt + 1, exc)
                    return await fallback()
            await asyncio.sleep(1.5 * (attempt + 1))
        filename = _url_filename(url, f"{kind}.png")
        file_id = await self._upload_file(chat_id, file_data, filename, ct)
        return await self._post_with_file(
            chat_id, file_id, _file_post_message(caption, [filename]), reply_to, metadata) if file_id else await fallback()

    async def _send_local_file(
        self, chat_id: str, file_path: str, caption: Optional[str], reply_to: Optional[str],
        file_name: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        """Upload a local file and attach it to a post."""
        p = Path(file_path)
        if not p.exists():
            logger.warning("Mattermost: local file not found, skipping: %s", file_path)
            return SendResult(success=False, error="Local file not found")
        fname = file_name or p.name
        file_id = await self._upload_file(chat_id, p.read_bytes(), fname,
                                          mimetypes.guess_type(fname)[0] or "application/octet-stream")
        if not file_id:
            return SendResult(success=False, error="File upload failed")
        return await self._post_with_file(chat_id, file_id, _file_post_message(caption, [fname]), reply_to, metadata)

    async def _load_batch_image(self, image_url: str, index: int) -> Optional[Tuple[bytes, str, str]]:
        """Read a file:// or remote image for a batch post → (data, filename, content_type), or None to skip."""
        import aiohttp
        if not image_url:
            return None
        if "://" not in image_url or image_url.startswith("file://"):
            local_path = url2pathname(image_url[5:]) if image_url.startswith("file://") else image_url
            p = Path(local_path)
            if not p.exists():
                logger.warning("Mattermost: skipping missing image %s", local_path)
                return None
            return p.read_bytes(), p.name, mimetypes.guess_type(p.name)[0] or "image/png"
        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            logger.warning("Mattermost: blocked unsafe image URL in batch")
            return None
        try:
            async with self._session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status >= 400:
                    logger.warning("Mattermost: failed to download image (HTTP %d): %s", resp.status, image_url[:80])
                    return None
                file_data, ct = await resp.read(), resp.content_type or "image/png"
        except Exception as dl_err:
            logger.warning("Mattermost: download failed for %s: %s", image_url[:80], dl_err)
            return None
        return file_data, _url_filename(image_url, f"image_{index}.png"), ct

    async def send_multiple_images(self, chat_id: str, images: List[Tuple[str, str]],
                                   metadata: _Metadata = None, human_delay: float = 0.0,
                                   *, caption: Optional[str] = None) -> SendResult:
        """Post up to five files per batch, retaining receipts without replaying uncertain POSTs."""
        message_ids: List[str] = []
        delivered = 0
        error = None
        caption_pending = caption
        pending_captions: List[str] = []
        try:
            for offset in range(0, len(images), 5):
                if human_delay > 0 and offset:
                    await asyncio.sleep(human_delay)
                file_ids, names = [], []
                for image_url, alt_text in images[offset:offset + 5]:
                    if alt_text:
                        pending_captions.append(alt_text)
                    loaded = await self._load_batch_image(image_url, len(file_ids))
                    if loaded is not None and (fid := await self._upload_file(chat_id, *loaded)):
                        file_ids.append(fid)
                        names.append(loaded[1])
                if not file_ids:
                    continue
                explicit_caption = caption_pending or "\n".join(pending_captions)
                body = _file_post_message(explicit_caption, names)
                data = await self._post_message(chat_id, body, None, metadata, file_ids)
                if data:
                    message_ids.extend(data.get("message_ids") or ([str(data["id"])] if data.get("id") else []))
                if not data or not data.get("id"):
                    if _post_was_rejected(data):
                        # Reuse the uploaded files through the native bounded post path.
                        # Do not download/upload again or replay an acknowledged caption.
                        for file_id, name in zip(file_ids, names):
                            if human_delay > 0:
                                await asyncio.sleep(human_delay)
                            single = await self._post_message(
                                chat_id, _file_post_message(explicit_caption, [name]), None, metadata, [file_id])
                            message_ids.extend(single.get("message_ids") or
                                               ([str(single["id"])] if single.get("id") else []))
                            if not single.get("id"):
                                error = "Mattermost single image post was not acknowledged"
                                if explicit_caption and not message_ids and _post_was_rejected(single):
                                    # Both attachment attempts were rejected: preserve the prose once.
                                    text = await self._post_message(chat_id, explicit_caption, None, metadata)
                                    message_ids.extend(text.get("message_ids") or
                                                       ([str(text["id"])] if text.get("id") else []))
                                break
                            delivered += 1
                            explicit_caption = caption_pending = None
                            pending_captions.clear()
                        if not error:
                            continue
                    else:
                        error = "Mattermost image post was not acknowledged"
                    break
                delivered += len(file_ids)
                caption_pending = None
                pending_captions.clear()
            caption_pending = caption_pending or "\n".join(pending_captions)
            if caption_pending and not error:
                # Text still belongs to the delivery when every image was missing.
                data = await self._post_message(chat_id, caption_pending, None, metadata)
                if data:
                    message_ids.extend(data.get("message_ids") or ([str(data["id"])] if data.get("id") else []))
                if not data or not data.get("id"):
                    error = "Mattermost caption post was not acknowledged"
        except asyncio.CancelledError:
            raise
        except Exception:
            error = "Mattermost image batch delivery failed"
        receipt = _media_delivery_receipt(chat_id, len(images), delivered, message_ids, error)
        # Gateway's batch contract means ANY image delivered; direct callers use counts.
        return SendResult(success=bool(delivered), message_id=receipt["message_id"],
                          continuation_message_ids=tuple(message_ids[:-1]),
                          error=receipt.get("error"), raw_response=receipt)

    # --- WebSocket ---

    async def _ws_loop(self) -> None:
        """Connect to the WebSocket and listen for events, reconnecting on failure."""
        import aiohttp
        import random
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await self._ws_connect_and_listen()
                delay = _RECONNECT_BASE_DELAY  # clean disconnect — reset backoff
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Permanent auth failure: escalate via the fatal-error hook (a bare return leaves is_connected()
                # healthy with a dead listener). Type-based: substring "401" matching misclassified transient errors.
                if isinstance(exc, aiohttp.WSServerHandshakeError) and exc.status in {401, 403}:
                    logger.error("Mattermost WS auth failed (HTTP %d) — stopping reconnect", exc.status)
                    # Escalate through the fatal-error hook instead of a bare return: the old silent exit
                    # left _running True, so is_connected() kept reporting healthy while the listener was
                    # dead and the gateway was never told (OOF-156 class). Type-based only — the substring
                    # fallback that used to sit below this branch misclassified transient errors whose
                    # message merely contained "401" (#80489).
                    self._set_fatal_error(
                        "mattermost_auth_error",
                        f"Mattermost WebSocket authentication rejected (HTTP {exc.status}). The bot token is "
                        "invalid, revoked, or lacks permission — check MATTERMOST_TOKEN and the bot account in "
                        "the System Console.", retryable=False)
                    await self._notify_fatal_error()
                    return
                logger.warning("Mattermost WS error: %s — reconnecting in %.0fs", exc, delay)
            if self._closing:
                return
            await asyncio.sleep(delay + delay * _RECONNECT_JITTER * random.random())
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _ws_connect_and_listen(self) -> None:
        """Single WebSocket session: connect, authenticate, process events."""
        ws_url = re.sub(r"^http", "ws", self._base_url) + "/api/v4/websocket"  # https→wss, http→ws
        logger.info("Mattermost: connecting to %s", ws_url)
        self._ws = await self._session.ws_connect(ws_url, heartbeat=30.0)
        await self._ws.send_json({"seq": 1, "action": "authentication_challenge", "data": {"token": self._token}})
        logger.info("Mattermost: WebSocket connected and authenticated")

        async for raw_msg in self._ws:
            if self._closing:
                return
            kind = raw_msg.type
            if kind in {kind.TEXT, kind.BINARY}:
                try:
                    event = json.loads(raw_msg.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._handle_ws_event(event)
            elif kind in {kind.ERROR, kind.CLOSE, kind.CLOSING, kind.CLOSED}:
                logger.info("Mattermost: WebSocket closed (%s)", kind)
                break

    def _extra_or_env(self, key: str, env: str, default: str = "") -> Any:
        """config.yaml ``mattermost.<key>`` (PlatformConfig.extra) first, env var fallback."""
        raw = self.config.extra.get(key) if self.config.extra else None
        return _get_scoped_secret(env, default) if raw is None else raw

    def _apply_channel_gating(self, channel_id: str, message_text: str) -> Optional[str]:
        """Mention-gate a non-DM post; return the cleaned text, or None to ignore it. allowed_channels is a
        whitelist checked first (@mentions elsewhere are ignored); require_mention (default true) is
        bypassed in free_response_channels."""
        allowed_channels = _channel_id_set(self._extra_or_env("allowed_channels", "MATTERMOST_ALLOWED_CHANNELS"))
        if allowed_channels and channel_id not in allowed_channels:
            logger.debug("Mattermost: ignoring message in non-allowed channel: %s", channel_id)
            return None
        require_mention = str(self._extra_or_env("require_mention", "MATTERMOST_REQUIRE_MENTION", "true")
                              ).lower() not in {"false", "0", "no"}
        free_channels = _channel_id_set(
            self._extra_or_env("free_response_channels", "MATTERMOST_FREE_RESPONSE_CHANNELS"))
        mention_patterns = [f"@{self._bot_username}", f"@{self._bot_user_id}"]
        has_mention = any(pattern.lower() in message_text.lower() for pattern in mention_patterns)
        if require_mention and channel_id not in free_channels and not has_mention:
            logger.debug("Mattermost: skipping non-DM message without @mention (channel=%s)", channel_id)
            return None
        if has_mention:  # strip the @mention so the agent sees clean input
            for pattern in mention_patterns:
                message_text = re.sub(re.escape(pattern), "", message_text, flags=re.IGNORECASE).strip()
        return message_text

    async def _download_attachments(self, file_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Download attachments now (URLs need auth headers downstream tools lack) → (paths, mime types)."""
        import aiohttp
        from gateway.platforms.base import (
            cache_audio_from_bytes_async,
            cache_document_from_bytes_async,
            cache_image_from_bytes_async,
        )
        media_urls, media_types = [], []
        cache_fns = {"image/": cache_image_from_bytes_async, "audio/": cache_audio_from_bytes_async}
        for fid in file_ids:
            try:
                file_info = await self._api_get(f"files/{fid}/info")
                fname = file_info.get("name", f"file_{fid}")
                mime = file_info.get("mime_type", "application/octet-stream")
                async with self._session.get(
                    f"{self._base_url}/api/v4/files/{fid}", headers=self._auth_header(),
                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status >= 400:
                        logger.warning("Mattermost: failed to download file %s: HTTP %s", fid, resp.status)
                        continue
                    file_data = await resp.read()
                    prefix = next((p for p in cache_fns if mime.startswith(p)), None)
                    if prefix:
                        media_urls.append(
                            await cache_fns[prefix](file_data, Path(fname).suffix or _INBOUND_CACHE_EXT[prefix]))
                    else:
                        media_urls.append(await cache_document_from_bytes_async(file_data, fname))
                    media_types.append(mime)
            except Exception as exc:
                logger.warning("Mattermost: error downloading file %s: %s", fid, exc)
        return media_urls, media_types

    async def _handle_ws_event(self, event: Dict[str, Any]) -> None:
        if event.get("event") != "posted":
            return
        data = event.get("data", {})
        try:
            post = json.loads(data.get("post") or "")
        except (json.JSONDecodeError, TypeError):
            return
        # Ignore own messages, system posts and redeliveries.
        sender_id, post_id = post.get("user_id", ""), post.get("id", "")
        if sender_id == self._bot_user_id or post.get("type") or self._dedup.is_duplicate(post_id):
            return
        channel_id, is_dm = post.get("channel_id", ""), data.get("channel_type", "O") == "D"
        message_text = post.get("message", "")
        if not is_dm:  # DMs need no gating; channels are mention-gated.
            message_text = self._apply_channel_gating(channel_id, message_text)
            if message_text is None:
                return
        # Thread support: replies use root_id; in thread mode a top-level channel post is itself a valid root.
        thread_id = post.get("root_id") or None
        if not thread_id and self._reply_mode == "thread" and not is_dm and post_id:
            thread_id = post_id
        if message_text[:1].isspace() and message_text.lstrip().startswith("/"):
            message_text = message_text.lstrip()
        media_urls, media_types = await self._download_attachments(post.get("file_ids") or [])
        if message_text.startswith("/"):
            msg_type = MessageType.COMMAND
        elif media_types:
            msg_type = next((mt for prefix, mt in _MEDIA_MSG_TYPES if any(m.startswith(prefix) for m in media_types)),
                            MessageType.DOCUMENT)
        else:
            msg_type = MessageType.TEXT
        source = self.build_source(
            chat_id=channel_id, chat_type=_CHANNEL_TYPE_MAP.get(data.get("channel_type", "O"), "channel"),
            user_id=sender_id, user_name=data.get("sender_name", "").lstrip("@") or sender_id,
            thread_id=thread_id, message_id=post_id)
        from gateway.platforms.base import resolve_channel_prompt
        await self.handle_message(MessageEvent(
            text=message_text, message_type=msg_type, source=source, raw_message=post, message_id=post_id,
            media_urls=media_urls or None, media_types=media_types or None,
            channel_prompt=resolve_channel_prompt(self.config.extra, channel_id, None)))


# --- Plugin standalone-send (out-of-process cron delivery via Mattermost REST) ---

async def _standalone_send(pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
                           media_files: Optional[list] = None, force_document: bool = False) -> Dict[str, Any]:
    """Native live delivery, else standalone REST; acknowledge each batch once, never replay."""
    from tools.send_message_tool import _dispatch_on_gateway_loop, _live_adapter
    runner, adapter = _live_adapter(Platform.MATTERMOST)
    if adapter is not None:
        metadata = {"thread_id": thread_id, "mattermost_explicit_thread": True} if thread_id else None
        if media_files:
            # Native paths stay native; URI conversion corrupts Windows drive/UNC paths.
            images = [(os.fspath(path) if isinstance(path := _media_path(item), (str, os.PathLike)) else "", "")
                      for item in media_files]
            make_coro = lambda: adapter.send_multiple_images(chat_id, images, metadata=metadata, caption=message)
        else:
            make_coro = lambda: adapter.send(chat_id=chat_id, content=message, metadata=metadata)
        try:
            result = await _dispatch_on_gateway_loop(runner, make_coro, "Mattermost native send failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            return {"error": "Mattermost native send failed"}
        if isinstance(result, dict):
            return result
        if media_files:
            return dict(result.raw_response)
        ids = [*result.continuation_message_ids, *([result.message_id] if result.message_id else [])]
        return _media_delivery_receipt(chat_id, 0, 0, ids, result.error)
    media_files = media_files or []
    message_ids: List[str] = []
    delivered = 0

    def receipt(error=None):
        return _media_delivery_receipt(chat_id, len(media_files), delivered, message_ids, error)

    try:
        import aiohttp
    except ImportError:
        return receipt("aiohttp not installed. Run: pip install aiohttp")
    base_url, token = _url_and_token(pconfig)
    base_url, token = base_url.rstrip("/"), token.strip()
    if not base_url or not token:
        return receipt("Mattermost standalone send: MATTERMOST_URL and MATTERMOST_TOKEN must both be set")
    upload_headers = {"Authorization": f"Bearer {token}"}
    headers = {**upload_headers, "Content-Type": "application/json"}
    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url(platform_env_var="MATTERMOST_PROXY"))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), **_sess_kw) as session:
            caption_pending = message
            batches = [media_files[i:i + 5] for i in range(0, len(media_files), 5)] or [[]]
            for index, batch in enumerate(batches):
                file_ids, names = [], []
                for descriptor in batch:
                    file_path = _media_path(descriptor)
                    if not isinstance(file_path, (str, os.PathLike)) or not file_path:
                        continue
                    filename = os.path.basename(file_path)
                    try:
                        with open(file_path, "rb") as fh:
                            file_data = fh.read()
                    except OSError:
                        # Includes disappearance between path validation and opening the file.
                        continue
                    form = aiohttp.FormData()
                    form.add_field("channel_id", chat_id)
                    form.add_field("files", file_data, filename=filename)
                    async with session.post(f"{base_url}/api/v4/files", data=form, headers=upload_headers,
                                            **_req_kw) as upload_resp:
                        if upload_resp.status not in {200, 201}:
                            return receipt(f"Mattermost file upload failed ({upload_resp.status})")
                        infos = (await upload_resp.json()).get("file_infos", [])
                        if not infos or not infos[0].get("id"):
                            return receipt("Mattermost file upload returned no file ID")
                        file_ids.append(infos[0]["id"])
                        names.append(filename)
                # Keep the caption for the first actual post, even if the first batch vanished.
                if not file_ids and (index < len(batches) - 1 or not caption_pending):
                    continue
                payload = _with_mentions_disabled({
                    "channel_id": chat_id,
                    "message": _file_post_message(caption_pending, names) if file_ids else caption_pending,
                })
                if thread_id:
                    payload["root_id"] = thread_id
                if file_ids:
                    payload["file_ids"] = file_ids
                limit = _resolve_max_post_length(getattr(pconfig, "extra", None))
                body = payload["message"]
                chunks = [body[i:i + limit] for i in range(0, len(body), limit)] or [""]
                for chunk_index, chunk in enumerate(chunks):
                    chunk_payload = {**payload, "message": chunk}
                    if chunk_index < len(chunks) - 1:
                        chunk_payload.pop("file_ids", None)
                    status = None
                    try:
                        async with session.post(f"{base_url}/api/v4/posts", headers=headers,
                                                json=chunk_payload, **_req_kw) as resp:
                            status = resp.status
                            if status not in {200, 201}:
                                return receipt(f"Mattermost API error ({status})")
                            data = await resp.json()
                    except asyncio.CancelledError:
                        raise
                    except aiohttp.ClientConnectorError:
                        return receipt("Mattermost connection failed before sending")
                    except Exception:
                        if status is None or status < 400:
                            return {**receipt(_POST_DELIVERY_UNCERTAIN), "_delivery_uncertain": True}
                        return receipt(f"Mattermost API error ({status})")
                    if not isinstance(data, dict) or not data.get("id"):
                        return receipt("Mattermost post was not acknowledged")
                    message_ids.append(str(data["id"]))
                    delivered += len(chunk_payload.get("file_ids", []))
                caption_pending = ""
        return receipt()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return receipt(f"Mattermost send failed: {exc}")


# --- Interactive setup wizard ---

def interactive_setup() -> None:
    """Guide the user through Mattermost bot setup (URL + token, allowlist, home channel)."""
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.cli_output import prompt, prompt_yes_no, print_header, print_info, print_success

    def info(*lines: str) -> None:
        for line in lines:
            print_info(line)

    print_header("Mattermost")
    if get_env_value("MATTERMOST_TOKEN"):
        print_info("Mattermost: already configured")
        if not prompt_yes_no("Reconfigure Mattermost?", False):
            return
    info("Works with any self-hosted Mattermost instance.",
         "   1. In Mattermost: Integrations → Bot Accounts → Add Bot Account", "   2. Copy the bot token")
    print()
    mm_url = prompt("Mattermost server URL (e.g. https://mm.example.com)")
    if mm_url:
        save_env_value("MATTERMOST_URL", mm_url.rstrip("/"))
    token = prompt("Bot token", password=True)
    if not token:
        return
    save_env_value("MATTERMOST_TOKEN", token)
    print_success("Mattermost token saved")
    print()
    info("🔒 Security: Restrict who can use your bot", "   To find your user ID: click your avatar → Profile",
         "   or use the API: GET /api/v4/users/me")
    print()
    allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
    if allowed_users:
        save_env_value("MATTERMOST_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Mattermost allowlist configured")
    else:
        print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")
    print()
    info("📬 Home Channel: where Hermes delivers cron job results and notifications.",
         "   To get a channel ID: click channel name → View Info → copy the ID",
         "   You can also set this later by typing /set-home in a Mattermost channel.")
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)").strip()
    if home_channel:
        save_env_value("MATTERMOST_HOME_CHANNEL", home_channel)
    elif remove_env_value("MATTERMOST_HOME_CHANNEL"):
        print_info("Home channel cleared.")
    print_info("   Open config in your editor:  hermes config edit")


# --- YAML → env config bridge (apply_yaml_config_fn) ---

_YAML_BRIDGE = (  # (yaml key, env var, yaml value → env string); allowed_channels is a whitelist
    ("require_mention", "MATTERMOST_REQUIRE_MENTION", lambda v: str(v).lower()),
    ("free_response_channels", "MATTERMOST_FREE_RESPONSE_CHANNELS", _csv),
    ("allowed_channels", "MATTERMOST_ALLOWED_CHANNELS", _csv))


def _apply_yaml_config(yaml_cfg: dict, mattermost_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``mattermost:`` keys into env vars + ``PlatformConfig.extra``.

    Env vars win over YAML (writes guarded by ``not os.getenv``). Under a multiplexed secondary
    profile the env write is skipped (it would leak into every profile via ``os.environ``); the
    values are returned so the caller seeds this profile's ``extra``, which read sites check first.

    Implements the ``apply_yaml_config_fn`` contract (#24836 / #25443). Mirrors the legacy
    ``mattermost_cfg`` block that used to live in ``gateway/config.py::load_gateway_config()`` before this
    migration.
    """
    skip_env_bridge = _profile_scoped_config_load()
    seeded: dict = {}
    if "max_post_length" in mattermost_cfg:
        seeded["max_post_length"] = _resolve_max_post_length(mattermost_cfg)
    for key, env, to_env in _YAML_BRIDGE:
        value = mattermost_cfg.get(key)
        if value is None and not (key == "require_mention" and key in mattermost_cfg):
            continue
        seeded[key] = value
        if not skip_env_bridge and not os.getenv(env):
            os.environ[env] = to_env(value)
    return seeded or None


def _is_connected(config) -> bool:
    """Connected when BOTH MATTERMOST_TOKEN and MATTERMOST_URL are set (``get_env_value`` looked up at
    call time so tests patching ``gateway_mod.get_env_value`` can suppress ambient env vars)."""
    import hermes_cli.gateway as gateway_mod
    return bool(
        (gateway_mod.get_env_value("MATTERMOST_TOKEN") or "").strip()
        and (gateway_mod.get_env_value("MATTERMOST_URL") or "").strip())


# --- Plugin registration entry point ---

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="mattermost", label="Mattermost", adapter_factory=MattermostAdapter,
        check_fn=check_mattermost_requirements, validate_config=validate_mattermost_config,
        is_connected=_is_connected, required_env=["MATTERMOST_URL", "MATTERMOST_TOKEN"],
        install_hint="pip install aiohttp", setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,  # YAML→env bridge (see _YAML_BRIDGE)
        allowed_users_env="MATTERMOST_ALLOWED_USERS", allow_all_env="MATTERMOST_ALLOW_ALL_USERS",
        cron_deliver_env_var="MATTERMOST_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,  # out-of-process cron; without it `deliver=mattermost` fails
        max_message_length=MAX_POST_LENGTH, emoji="💬", allow_update_command=True)
