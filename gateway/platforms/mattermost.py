"""Mattermost gateway adapter.

Connects to a self-hosted (or cloud) Mattermost instance via its REST API
(v4) and WebSocket for real-time events.  No external Mattermost library
required — uses aiohttp which is already a Hermes dependency.

Environment variables:
    MATTERMOST_URL              Server URL (e.g. https://mm.example.com)
    MATTERMOST_TOKEN            Bot token or personal-access token
    MATTERMOST_ALLOWED_USERS    Comma-separated user IDs
    MATTERMOST_HOME_CHANNEL     Channel ID for cron/notification delivery
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Mattermost post size limit (server default is 16383, but 4000 is the
# practical limit for readable messages — matching OpenClaw's choice).
MAX_POST_LENGTH = 4000

# Channel type codes returned by the Mattermost API.
_CHANNEL_TYPE_MAP = {
    "D": "dm",
    "G": "group",
    "P": "group",   # private channel → treat as group
    "O": "channel",
}

# Reconnect parameters (exponential backoff).
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2


def check_mattermost_requirements() -> bool:
    """Return True if the Mattermost adapter can be used."""
    token = os.getenv("MATTERMOST_TOKEN", "")
    url = os.getenv("MATTERMOST_URL", "")
    if not token:
        logger.debug("Mattermost: MATTERMOST_TOKEN not set")
        return False
    if not url:
        logger.warning("Mattermost: MATTERMOST_URL not set")
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Mattermost: aiohttp not installed")
        return False


class MattermostAdapter(BasePlatformAdapter):
    """Gateway adapter for Mattermost (self-hosted or cloud)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATTERMOST)

        self._base_url: str = (
            config.extra.get("url", "")
            or os.getenv("MATTERMOST_URL", "")
        ).rstrip("/")
        self._token: str = config.token or os.getenv("MATTERMOST_TOKEN", "")

        self._bot_user_id: str = ""
        self._bot_username: str = ""

        # aiohttp session + websocket handle
        self._session: Any = None  # aiohttp.ClientSession
        self._ws: Any = None       # aiohttp.ClientWebSocketResponse
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._closing = False

        # Reply mode: "thread" to nest replies, "off" for flat messages.
        # Mattermost is noisy without threads; make the safe/default behavior
        # stay in the triggering post's thread unless explicitly disabled.
        self._reply_mode: str = (
            config.extra.get("reply_mode", "")
            or os.getenv("MATTERMOST_REPLY_MODE", "thread")
        ).lower()
        no_thread_raw = config.extra.get("no_thread_channels") or []
        if isinstance(no_thread_raw, str):
            no_thread_values = no_thread_raw.split(",")
        else:
            no_thread_values = list(no_thread_raw)
        env_no_thread = os.getenv("MATTERMOST_NO_THREAD_CHANNELS", "")
        if env_no_thread:
            no_thread_values.extend(env_no_thread.split(","))
        self._no_thread_channels = {str(v).strip() for v in no_thread_values if str(v).strip()}

        auto_thread_raw = (
            config.extra.get("auto_thread_channels", "")
            or os.getenv("MATTERMOST_AUTO_THREAD_CHANNELS", "")
        )
        if isinstance(auto_thread_raw, (list, tuple, set)):
            self._auto_thread_channels = {
                str(channel).strip() for channel in auto_thread_raw if str(channel).strip()
            }
        else:
            self._auto_thread_channels = {
                channel.strip() for channel in str(auto_thread_raw).split(",") if channel.strip()
            }

        # Dedup cache (prevent reprocessing)
        self._dedup = MessageDeduplicator()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def _api_get(self, path: str) -> Dict[str, Any]:
        """GET /api/v4/{path}."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            async with self._session.get(url, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API GET %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("MM API GET %s network error: %s", path, exc)
            return {}

    async def _api_post(
        self, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST /api/v4/{path} with JSON body."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            async with self._session.post(
                url, headers=self._headers(), json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API POST %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("MM API POST %s network error: %s", path, exc)
            return {}

    async def _api_put(
        self, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """PUT /api/v4/{path} with JSON body."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            async with self._session.put(
                url, headers=self._headers(), json=payload
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API PUT %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("MM API PUT %s network error: %s", path, exc)
            return {}

    async def _upload_file(
        self, channel_id: str, file_data: bytes, filename: str, content_type: str = "application/octet-stream"
    ) -> Optional[str]:
        """Upload a file and return its file ID, or None on failure."""
        import aiohttp

        url = f"{self._base_url}/api/v4/files"
        form = aiohttp.FormData()
        form.add_field("channel_id", channel_id)
        form.add_field(
            "files",
            file_data,
            filename=filename,
            content_type=content_type,
        )
        headers = {"Authorization": f"Bearer {self._token}"}
        async with self._session.post(url, headers=headers, data=form, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.error("MM file upload → %s: %s", resp.status, body[:200])
                return None
            data = await resp.json()
            infos = data.get("file_infos", [])
            return infos[0]["id"] if infos else None

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to Mattermost and start the WebSocket listener."""
        import aiohttp

        if not self._base_url or not self._token:
            logger.error("Mattermost: URL or token not configured")
            return False

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._closing = False

        # Verify credentials and fetch bot identity.
        me = await self._api_get("users/me")
        if not me or "id" not in me:
            logger.error("Mattermost: failed to authenticate — check MATTERMOST_TOKEN and MATTERMOST_URL")
            await self._session.close()
            return False

        self._bot_user_id = me["id"]
        self._bot_username = me.get("username", "")
        logger.info(
            "Mattermost: authenticated as @%s (%s) on %s",
            self._bot_username,
            self._bot_user_id,
            self._base_url,
        )

        # Start WebSocket in background.
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Disconnect from Mattermost."""
        self._closing = True

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Mattermost: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message (or multiple chunks) to a channel."""
        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_POST_LENGTH)

        last_id = None
        for chunk in chunks:
            payload: Dict[str, Any] = {
                "channel_id": chat_id,
                "message": chunk,
            }
            root_id = self._root_id_from_send_context(chat_id, reply_to, metadata)
            if root_id:
                payload["root_id"] = root_id

            data = await self._api_post("posts", payload)
            if not data or "id" not in data:
                return SendResult(success=False, error="Failed to create post")
            last_id = data["id"]

        return SendResult(success=True, message_id=last_id)

    def _should_thread_reply(self, chat_id: str) -> bool:
        """Return whether replies in this channel should be nested in a Mattermost thread."""
        return self._reply_mode == "thread" and chat_id not in self._no_thread_channels

    def _root_id_from_send_context(
        self,
        chat_id: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Resolve the Mattermost thread root for an outbound post.

        Gateway progress, streaming, approval, and background notifications
        often carry the canonical thread root in metadata and leave reply_to
        empty. For Mattermost, root_id must be the root post id; child post ids
        are rejected with Invalid RootId.
        """
        if not self._should_thread_reply(chat_id):
            return None
        root_id = None
        if metadata:
            root_id = metadata.get("mattermost_root_id") or metadata.get("thread_id")
        root_id = root_id or reply_to
        return str(root_id) if root_id else None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return channel name and type."""
        data = await self._api_get(f"channels/{chat_id}")
        if not data:
            return {"name": chat_id, "type": "channel"}

        ch_type = _CHANNEL_TYPE_MAP.get(data.get("type", "O"), "channel")
        display_name = data.get("display_name") or data.get("name") or chat_id
        return {"name": display_name, "type": ch_type}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a typing indicator, scoped to a thread when Mattermost supports it."""
        payload = {"channel_id": chat_id}
        thread_id = None
        if metadata:
            thread_id = metadata.get("thread_id") or metadata.get("mattermost_root_id")
        if thread_id:
            payload["parent_id"] = str(thread_id)
        await self._api_post(
            f"users/{self._bot_user_id}/typing",
            payload,
        )

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False
    ) -> SendResult:
        """Edit an existing post."""
        formatted = self.format_message(content)
        data = await self._api_put(
            f"posts/{message_id}/patch",
            {"message": formatted},
        )
        if not data or "id" not in data:
            return SendResult(success=False, error="Failed to edit post")
        return SendResult(success=True, message_id=data["id"])

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download an image and upload it as a file attachment."""
        return await self._send_url_as_file(
            chat_id, image_url, caption, reply_to, metadata=metadata, kind="image"
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local image file."""
        return await self._send_local_file(
            chat_id, image_path, caption, reply_to, metadata=metadata
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file as a document."""
        return await self._send_local_file(
            chat_id, file_path, caption, reply_to, file_name, metadata=metadata
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload an audio file."""
        return await self._send_local_file(
            chat_id, audio_path, caption, reply_to, metadata=metadata
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a video file."""
        return await self._send_local_file(
            chat_id, video_path, caption, reply_to, metadata=metadata
        )

    def format_message(self, content: str) -> str:
        """Mattermost uses standard Markdown — mostly pass through.

        Strip image markdown into plain links (files are uploaded separately).
        """
        # Convert ![alt](url) to just the URL — Mattermost renders
        # image URLs as inline previews automatically.
        content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)
        return content

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    async def _send_url_as_file(
        self,
        chat_id: str,
        url: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        kind: str = "file",
    ) -> SendResult:
        """Download a URL and upload it as a file attachment."""
        from tools.url_safety import is_safe_url
        if not is_safe_url(url):
            logger.warning("Mattermost: blocked unsafe URL (SSRF protection)")
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        import aiohttp

        file_data = None
        ct = "application/octet-stream"
        fname = url.rsplit("/", 1)[-1].split("?")[0] or f"{kind}.png"

        for attempt in range(3):
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        if attempt < 2:
                            logger.debug("Mattermost download retry %d/2 for %s (status %d)",
                                         attempt + 1, url[:80], resp.status)
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    if resp.status >= 400:
                        return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)
                    file_data = await resp.read()
                    ct = resp.content_type or "application/octet-stream"
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                logger.warning("Mattermost: failed to download %s after %d attempts: %s", url, attempt + 1, exc)
                return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        if file_data is None:
            logger.warning("Mattermost: download returned no data for %s", url)
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        file_id = await self._upload_file(chat_id, file_data, fname, ct)
        if not file_id:
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": caption or "",
            "file_ids": [file_id],
        }
        root_id = self._root_id_from_send_context(chat_id, reply_to, metadata)
        if root_id:
            payload["root_id"] = root_id

        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            return SendResult(success=False, error="Failed to post with file")
        return SendResult(success=True, message_id=data["id"])

    async def _send_local_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        file_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file and attach it to a post."""
        import mimetypes

        p = Path(file_path)
        if not p.exists():
            return await self.send(
                chat_id, f"{caption or ''}\n(file not found: {file_path})", reply_to, metadata=metadata
            )

        fname = file_name or p.name
        ct = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        file_data = p.read_bytes()

        file_id = await self._upload_file(chat_id, file_data, fname, ct)
        if not file_id:
            return SendResult(success=False, error="File upload failed")

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": caption or "",
            "file_ids": [file_id],
        }
        root_id = self._root_id_from_send_context(chat_id, reply_to, metadata)
        if root_id:
            payload["root_id"] = root_id

        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            return SendResult(success=False, error="Failed to post with file")
        return SendResult(success=True, message_id=data["id"])

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images as a single Mattermost post with multiple attachments.

        Mattermost supports up to 5 ``file_ids`` per post. Each image is
        uploaded individually (Mattermost's file API is one-at-a-time),
        then a single post is created referencing all uploaded file_ids
        at once. Batches larger than 5 are chunked. Falls back to the
        base per-image loop on total failure.
        """
        if not images:
            return

        import mimetypes
        import aiohttp
        from urllib.parse import unquote as _unquote

        CHUNK = 5  # Mattermost post file_ids cap
        chunks = [images[i:i + CHUNK] for i in range(0, len(images), CHUNK)]

        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)

            file_ids: List[str] = []
            caption_parts: List[str] = []
            try:
                for image_url, alt_text in chunk:
                    if alt_text:
                        caption_parts.append(alt_text)

                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        p = Path(local_path)
                        if not p.exists():
                            logger.warning("Mattermost: skipping missing image %s", local_path)
                            continue
                        fname = p.name
                        ct = mimetypes.guess_type(fname)[0] or "image/png"
                        file_data = p.read_bytes()
                    else:
                        from tools.url_safety import is_safe_url
                        if not is_safe_url(image_url):
                            logger.warning("Mattermost: blocked unsafe image URL in batch")
                            continue
                        try:
                            async with self._session.get(
                                image_url, timeout=aiohttp.ClientTimeout(total=30)
                            ) as resp:
                                if resp.status >= 400:
                                    logger.warning(
                                        "Mattermost: failed to download image (HTTP %d): %s",
                                        resp.status, image_url[:80],
                                    )
                                    continue
                                file_data = await resp.read()
                                ct = resp.content_type or "image/png"
                        except Exception as dl_err:
                            logger.warning("Mattermost: download failed for %s: %s", image_url[:80], dl_err)
                            continue
                        fname = image_url.rsplit("/", 1)[-1].split("?")[0] or f"image_{len(file_ids)}.png"

                    fid = await self._upload_file(chat_id, file_data, fname, ct)
                    if fid:
                        file_ids.append(fid)

                if not file_ids:
                    continue

                payload: Dict[str, Any] = {
                    "channel_id": chat_id,
                    "message": "\n".join(caption_parts),
                    "file_ids": file_ids,
                }
                root_id = self._root_id_from_send_context(chat_id, metadata=metadata)
                if root_id:
                    payload["root_id"] = root_id
                logger.info(
                    "Mattermost: sending %d image(s) as single post (chunk %d/%d)",
                    len(file_ids), chunk_idx + 1, len(chunks),
                )
                data = await self._api_post("posts", payload)
                if not data or "id" not in data:
                    logger.warning("Mattermost: multi-image post failed, falling back")
                    await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
            except Exception as e:
                logger.warning(
                    "Mattermost: multi-image send failed (chunk %d/%d), falling back: %s",
                    chunk_idx + 1, len(chunks), e, exc_info=True,
                )
                await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Connect to the WebSocket and listen for events, reconnecting on failure."""
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await self._ws_connect_and_listen()
                # Clean disconnect — reset delay.
                delay = _RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Detect permanent auth/permission failures that will never
                # succeed on retry — stop reconnecting instead of looping forever.
                import aiohttp
                err_str = str(exc).lower()
                if isinstance(exc, aiohttp.WSServerHandshakeError) and exc.status in {401, 403}:
                    logger.error("Mattermost WS auth failed (HTTP %d) — stopping reconnect", exc.status)
                    return
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                    logger.error("Mattermost WS permanent error: %s — stopping reconnect", exc)
                    return
                logger.warning("Mattermost WS error: %s — reconnecting in %.0fs", exc, delay)

            if self._closing:
                return

            # Exponential backoff with jitter.
            import random
            jitter = delay * _RECONNECT_JITTER * random.random()
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _ws_connect_and_listen(self) -> None:
        """Single WebSocket session: connect, authenticate, process events."""
        # Build WS URL: https:// → wss://, http:// → ws://
        ws_url = re.sub(r"^http", "ws", self._base_url) + "/api/v4/websocket"
        logger.info("Mattermost: connecting to %s", ws_url)

        self._ws = await self._session.ws_connect(ws_url, heartbeat=30.0)

        # Authenticate via the WebSocket.
        auth_msg = {
            "seq": 1,
            "action": "authentication_challenge",
            "data": {"token": self._token},
        }
        await self._ws.send_json(auth_msg)
        logger.info("Mattermost: WebSocket connected and authenticated")

        async for raw_msg in self._ws:
            if self._closing:
                return

            if raw_msg.type in {
                raw_msg.type.TEXT,
                raw_msg.type.BINARY,
            }:
                try:
                    event = json.loads(raw_msg.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._handle_ws_event(event)
            elif raw_msg.type in {
                raw_msg.type.ERROR,
                raw_msg.type.CLOSE,
                raw_msg.type.CLOSING,
                raw_msg.type.CLOSED,
            }:
                logger.info("Mattermost: WebSocket closed (%s)", raw_msg.type)
                break

    async def _handle_ws_event(self, event: Dict[str, Any]) -> None:
        """Process a single WebSocket event."""
        event_type = event.get("event")
        if event_type != "posted":
            return

        data = event.get("data", {})
        raw_post_str = data.get("post")
        if not raw_post_str:
            return

        try:
            post = json.loads(raw_post_str)
        except (json.JSONDecodeError, TypeError):
            return

        # Ignore own messages.
        if post.get("user_id") == self._bot_user_id:
            return

        # Ignore system posts.
        if post.get("type"):
            return

        post_id = post.get("id", "")

        # Dedup.
        if self._dedup.is_duplicate(post_id):
            return

        # Build message event.
        channel_id = post.get("channel_id", "")
        channel_type_raw = data.get("channel_type", "O")
        chat_type = _CHANNEL_TYPE_MAP.get(channel_type_raw, "channel")

        # For DMs, user_id is sufficient.  For channels, check for @mention.
        message_text = post.get("message", "")
        sender_id = post.get("user_id", "")
        sender_name = data.get("sender_name", "").lstrip("@") or sender_id

        # Mention-gating for non-DM channels.
        # Config (config.yaml `mattermost.*` with env-var fallback):
        #   require_mention / MATTERMOST_REQUIRE_MENTION: Require @mention in channels (default: true)
        #   free_response_channels / MATTERMOST_FREE_RESPONSE_CHANNELS: Channel IDs where bot responds without mention
        #   allowed_channels / MATTERMOST_ALLOWED_CHANNELS: If set, bot ONLY responds in these channels (whitelist)
        if channel_type_raw != "D":
            # allowed_channels check (whitelist — must pass before other gating).
            # When set, messages from channels NOT in this list are silently
            # ignored, even if @mentioned.  DMs are already excluded above.
            allowed_raw = self.config.extra.get("allowed_channels") if self.config.extra else None
            if allowed_raw is None:
                allowed_raw = os.getenv("MATTERMOST_ALLOWED_CHANNELS", "")
            if isinstance(allowed_raw, list):
                allowed_channels = {str(c).strip() for c in allowed_raw if str(c).strip()}
            else:
                allowed_channels = {
                    c.strip() for c in str(allowed_raw).split(",") if c.strip()
                }
            if allowed_channels and channel_id not in allowed_channels:
                logger.debug(
                    "Mattermost: ignoring message in non-allowed channel: %s",
                    channel_id,
                )
                return

            # Moon delivery approvals are intentionally plain replies like
            # "go" inside coding threads. They must be recognized before the
            # generic @mention gate, but the helper still fail-closes on
            # configured channel, thread root, exact approval phrase, plan
            # readiness, and approved sender allowlist.
            if await self._maybe_enqueue_moon_delivery_goal(
                post=post,
                data=data,
                message_text=message_text,
                channel_id=channel_id,
                sender_id=sender_id,
                sender_name=sender_name,
            ):
                return

            require_mention = os.getenv(
                "MATTERMOST_REQUIRE_MENTION", "true"
            ).lower() not in {"false", "0", "no"}

            free_channels_raw = os.getenv("MATTERMOST_FREE_RESPONSE_CHANNELS", "")
            free_channels = {ch.strip() for ch in free_channels_raw.split(",") if ch.strip()}
            is_free_channel = channel_id in free_channels

            mention_patterns = [
                f"@{self._bot_username}",
                f"@{self._bot_user_id}",
            ]
            has_mention = any(
                pattern.lower() in message_text.lower()
                for pattern in mention_patterns
            )

            if require_mention and not is_free_channel and not has_mention:
                logger.debug(
                    "Mattermost: skipping non-DM message without @mention (channel=%s)",
                    channel_id,
                )
                return

            # Strip @mention from the message text so the agent sees clean input.
            if has_mention:
                for pattern in mention_patterns:
                    message_text = re.sub(
                        re.escape(pattern), "", message_text, flags=re.IGNORECASE
                    ).strip()
        # Thread support: if the post is in a thread, use root_id. For
        # configured coding/work channels, a top-level post is the thread root
        # so each task gets a siloed session and all bot replies stay nested.
        thread_id = post.get("root_id") or None
        if not thread_id and channel_id in self._auto_thread_channels and channel_type_raw != "D":
            thread_id = post_id or None

        # Determine message type.
        file_ids = post.get("file_ids") or []
        msg_type = MessageType.TEXT
        if message_text.startswith("/"):
            msg_type = MessageType.COMMAND

        # Download file attachments immediately (URLs require auth headers
        # that downstream tools won't have).
        media_urls: List[str] = []
        media_types: List[str] = []
        for fid in file_ids:
            try:
                file_info = await self._api_get(f"files/{fid}/info")
                fname = file_info.get("name", f"file_{fid}")
                ext = Path(fname).suffix or ""
                mime = file_info.get("mime_type", "application/octet-stream")

                import aiohttp
                dl_url = f"{self._base_url}/api/v4/files/{fid}"
                async with self._session.get(
                    dl_url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status < 400:
                        file_data = await resp.read()
                        from gateway.platforms.base import cache_image_from_bytes, cache_document_from_bytes
                        if mime.startswith("image/"):
                            local_path = cache_image_from_bytes(file_data, ext or ".png")
                            media_urls.append(local_path)
                            media_types.append(mime)
                        elif mime.startswith("audio/"):
                            from gateway.platforms.base import cache_audio_from_bytes
                            local_path = cache_audio_from_bytes(file_data, ext or ".ogg")
                            media_urls.append(local_path)
                            media_types.append(mime)
                        else:
                            local_path = cache_document_from_bytes(file_data, fname)
                            media_urls.append(local_path)
                            media_types.append(mime)
                    else:
                        logger.warning("Mattermost: failed to download file %s: HTTP %s", fid, resp.status)
            except Exception as exc:
                logger.warning("Mattermost: error downloading file %s: %s", fid, exc)

        # Set message type based on downloaded media types.
        if media_types and msg_type == MessageType.TEXT:
            if any(m.startswith("image/") for m in media_types):
                msg_type = MessageType.PHOTO
            elif any(m.startswith("audio/") for m in media_types):
                msg_type = MessageType.VOICE
            elif media_types:
                msg_type = MessageType.DOCUMENT

        source = self.build_source(
            chat_id=channel_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            thread_id=thread_id,
            message_id=post_id,
        )

        # Per-channel ephemeral prompt
        from gateway.platforms.base import resolve_channel_prompt
        _channel_prompt = resolve_channel_prompt(
            self.config.extra, channel_id, None,
        )

        msg_event = MessageEvent(
            text=message_text,
            message_type=msg_type,
            source=source,
            raw_message=post,
            message_id=post_id,
            media_urls=media_urls if media_urls else None,
            media_types=media_types if media_types else None,
            channel_prompt=_channel_prompt,
        )

        await self.handle_message(msg_event)

    async def _maybe_enqueue_moon_delivery_goal(
        self,
        *,
        post: Dict[str, Any],
        data: Dict[str, Any],
        message_text: str,
        channel_id: str,
        sender_id: str,
        sender_name: str,
    ) -> bool:
        """Turn approved Moon coding threads into durable Kanban work.

        The human-facing flow stays natural: a top-level post gets a plan in
        its Mattermost thread; when Mauri/Aaron replies with an approval phrase,
        the gateway creates exactly one Kanban task for that Mattermost thread
        and lets the embedded dispatcher spawn the worker.  This keeps feature
        delivery durable instead of relying on a long interactive agent run.
        """
        cfg = self._moon_delivery_goal_config()
        if not cfg.get("enabled"):
            return False
        channels = {str(ch).strip() for ch in (cfg.get("channels") or []) if str(ch).strip()}
        if channel_id not in channels:
            return False

        root_id = str(post.get("root_id") or "").strip()
        require_thread = bool(cfg.get("require_thread", True))
        if require_thread and not root_id:
            return False

        text = (message_text or "").strip().lower()
        if not text:
            return False
        approvals = {str(p).strip().lower() for p in (cfg.get("approval_phrases") or []) if str(p).strip()}
        if text not in approvals:
            return False

        allowed_sender_ids = {str(s).strip() for s in (cfg.get("approved_sender_ids") or []) if str(s).strip()}
        allowed_senders = {str(s).strip().lower().lstrip("@") for s in (cfg.get("approved_senders") or []) if str(s).strip()}
        sender_key = str(sender_name or "").strip().lower().lstrip("@")
        sender_id_key = str(sender_id or "").strip()
        sender_allowed = bool(allowed_sender_ids and sender_id_key in allowed_sender_ids)
        # Name fallback is only used while bootstrapping, when immutable IDs are
        # not configured yet. Once approved_sender_ids exists, IDs are the sole
        # authority. If neither allowlist exists, fail closed; approval phrases
        # must never become "anyone in channel can launch autonomous work".
        if not allowed_sender_ids and allowed_senders:
            sender_allowed = sender_key in allowed_senders
        if not sender_allowed:
            logger.info(
                "Mattermost: consuming denied Moon delivery approval from non-approved sender %s/%s in %s",
                sender_name,
                sender_id,
                channel_id,
            )
            await self.send(
                channel_id,
                "I saw an approval phrase, but that account is not allowed to launch Moon delivery tasks.",
                reply_to=root_id or post.get("id"),
            )
            return True

        task_root_id = root_id or post.get("id", "")
        thread_transcript = await asyncio.to_thread(self._fetch_thread_transcript_sync, task_root_id)
        if not self._thread_looks_ready_for_moon_delivery(thread_transcript):
            logger.info(
                "Mattermost: consuming approval phrase in %s/%s because thread does not look like an approved Moon delivery plan",
                channel_id,
                task_root_id,
            )
            await self.send(
                channel_id,
                "I saw the approval phrase, but I don't see a complete Moon delivery plan in this thread yet. Post/approve the plan with risk, verification, and release impact first.",
                reply_to=task_root_id,
            )
            return True

        task_id = await asyncio.to_thread(
            self._create_moon_delivery_task_sync,
            cfg,
            post,
            data,
            channel_id,
            task_root_id,
            sender_name,
            thread_transcript,
        )
        if not task_id:
            await self.send(
                channel_id,
                "I saw the approval, but couldn't create the Kanban task. Check gateway logs — the conveyor belt jammed before worker dispatch.",
                reply_to=root_id or post.get("id"),
            )
            return True

        board = str(cfg.get("board") or "moon")
        assignee = str(cfg.get("assignee") or cfg.get("default_assignee") or "mooncoder")
        await self.send(
            channel_id,
            f"Approved plan captured. Kanban task `{task_id}` is queued on `{board}` for `{assignee}`; the dispatcher will pick it up autonomously.",
            reply_to=root_id or post.get("id"),
        )
        return True

    def _moon_delivery_goal_config(self) -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            section = cfg.get("moon_delivery_goal", {}) if isinstance(cfg, dict) else {}
            return section if isinstance(section, dict) else {}
        except Exception as exc:
            logger.warning("Mattermost: failed to load moon_delivery_goal config: %s", exc)
            return {}

    def _create_moon_delivery_task_sync(
        self,
        cfg: Dict[str, Any],
        post: Dict[str, Any],
        data: Dict[str, Any],
        channel_id: str,
        root_id: str,
        sender_name: str,
        thread_transcript: str,
    ) -> Optional[str]:
        try:
            from hermes_cli import kanban_db as kb

            board = str(cfg.get("board") or "moon")
            assignee = str(cfg.get("assignee") or cfg.get("default_assignee") or "mooncoder")
            tenant = str(cfg.get("tenant") or "moon")
            thread_url = self._mattermost_thread_url(data, channel_id, root_id)
            body = self._moon_delivery_task_body(cfg, channel_id, root_id, thread_url, sender_name, thread_transcript)
            title = self._moon_delivery_task_title(body)
            metadata = {
                "source": "mattermost_moon_delivery_goal",
                "platform": "mattermost",
                "channel_id": channel_id,
                "thread_id": root_id,
                "thread_url": thread_url,
                "approved_by": sender_name,
                "repo": str(cfg.get("repo") or "/Users/mauri/app"),
                "release_train": [
                    "scope_and_plan_approved",
                    "branch_or_worktree",
                    "implement",
                    "self_review_diff",
                    "automated_tests",
                    "simulator_or_emulator_if_available",
                    "physical_device_when_required",
                    "ota_store_or_backend_gate",
                    "final_report",
                ],
            }
            with kb.connect(board=board) as conn:
                return kb.create_task(
                    conn,
                    title=title,
                    body=body,
                    assignee=assignee,
                    created_by=f"mattermost:{sender_name}",
                    workspace_kind=str(cfg.get("workspace") or "worktree"),
                    tenant=tenant,
                    priority=int(cfg.get("priority", 10) or 10),
                    idempotency_key=f"mattermost:{channel_id}:{root_id}:moon-delivery-goal",
                    max_runtime_seconds=int(cfg.get("max_runtime_seconds", 7200) or 7200),
                    skills=["expo-eas-mobile-release", "github-pr-workflow", "requesting-code-review"],
                    metadata=metadata,
                )
        except Exception as exc:
            logger.error("Mattermost: failed to create Moon delivery Kanban task: %s", exc, exc_info=True)
            return None

    def _mattermost_thread_url(self, data: Dict[str, Any], channel_id: str, root_id: str) -> str:
        team = data.get("team_name") or data.get("team_id") or ""
        if self._base_url and team and root_id:
            return f"{self._base_url}/{team}/pl/{root_id}"
        if self._base_url and root_id:
            return f"{self._base_url}/_redirect/pl/{root_id}"
        return f"mattermost:{channel_id}:{root_id}"

    def _moon_delivery_task_body(
        self,
        cfg: Dict[str, Any],
        channel_id: str,
        root_id: str,
        thread_url: str,
        sender_name: str,
        thread_transcript: str,
    ) -> str:
        thread_text = self._redact_mattermost_transcript(thread_transcript)
        template = str(cfg.get("goal_template") or "Implement the approved Moon task from this Mattermost thread.")
        repo = str(cfg.get("repo") or "/Users/mauri/app")
        return (
            f"{template}\n\n"
            f"Approved by: {sender_name}\n"
            f"Source: {thread_url}\n"
            f"Repo: {repo}\n\n"
            "Mandatory workflow:\n"
            "1. Work in an isolated branch/worktree; do not touch unrelated dirty files.\n"
            "2. Implement the approved plan from the transcript below.\n"
            "3. Run targeted tests plus ./scripts/moon-verify-branch.sh when available.\n"
            "4. Test simulator/emulator/physical-device paths when the environment supports them; if tooling is missing, report the blocker explicitly.\n"
            "5. Review every diff before commit.\n"
            "6. Push/merge only low/medium-risk app-only work when green. Ask before production Supabase/backend deploys, EAS credit-spending builds, store submission, Stripe/billing, destructive data work, or unclear high-risk rollback.\n\n"
            "Mattermost thread transcript:\n"
            f"{thread_text}"
        )

    def _fetch_thread_transcript_sync(self, root_id: str) -> str:
        # Called from an asyncio.to_thread context; use the REST API with urllib
        # instead of trying to await the adapter's aiohttp session from a worker thread.
        if not root_id or not self._base_url or not self._token:
            return "[thread transcript unavailable]"
        try:
            import urllib.request
            from urllib.parse import urlsplit

            base_host = urlsplit(self._base_url).netloc
            if not base_host or urlsplit(self._base_url).scheme not in {"http", "https"}:
                return "[thread transcript unavailable]"

            class _NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
                    return None

            req = urllib.request.Request(
                f"{self._base_url}/api/v4/posts/{root_id}/thread",
                headers={"Authorization": f"Bearer {self._token}"},
            )
            opener = urllib.request.build_opener(_NoRedirect)
            with opener.open(req, timeout=30) as resp:  # nosec B310: configured Mattermost base URL, redirects disabled
                final_host = urlsplit(resp.geturl()).netloc
                if final_host and final_host != base_host:
                    return "[thread transcript unavailable]"
                payload = json.loads(resp.read().decode("utf-8"))
            order = payload.get("order") or []
            posts = payload.get("posts") or {}
            lines: List[str] = []
            for pid in order:
                p = posts.get(pid) or {}
                msg = str(p.get("message") or "").strip()
                if not msg:
                    continue
                user = p.get("props", {}).get("from_webhook") or p.get("user_id") or "unknown"
                lines.append(f"[{user}] {msg}")
            return "\n\n".join(lines) if lines else "[thread transcript empty]"
        except Exception as exc:
            logger.warning("Mattermost: failed to fetch thread transcript for %s: %s", root_id, exc)
            return "[thread transcript unavailable]"

    def _thread_looks_ready_for_moon_delivery(self, transcript: str) -> bool:
        text = (transcript or "").lower()
        if not text or text.startswith("[thread transcript unavailable]"):
            return False
        # The channel prompts require the plan to cover these exact concepts.
        # Requiring them makes casual replies like "go" in a random thread fall
        # through to normal agent handling instead of launching a worker.
        has_plan_shape = all(marker in text for marker in ("risk", "verification", "release"))
        # Use token/word-aware matching so generic words like "release" do not
        # accidentally satisfy the EAS marker via the substring "eas".
        has_moon_context = bool(re.search(r"\b(?:moon|mooninv|mooncus|supabase|eas|ota)\b", text))
        return has_plan_shape and has_moon_context

    def _redact_mattermost_transcript(self, transcript: str) -> str:
        redacted = transcript or "[thread transcript unavailable]"
        patterns = (
            r'(?i)(api[_-]?key|secret|password|passwd|token|service[_-]?role[_-]?key)\s*[:=]\s*[^\s`\'\"]+',
            r'(?i)(bearer\s+)[a-z0-9._~+/=-]{12,}',
            r'(?i)(sk|rk|pk)_[a-z0-9_\-]{16,}',
            r'postgres(?:ql)?://[^\s`\'\"]+',
        )
        for pattern in patterns:
            redacted = re.sub(pattern, lambda m: m.group(1) + "[REDACTED]" if m.lastindex else "[REDACTED]", redacted)
        return redacted

    def _moon_delivery_task_title(self, body: str) -> str:
        in_transcript = False
        for line in body.splitlines():
            stripped = line.strip()
            if stripped == "Mattermost thread transcript:":
                in_transcript = True
                continue
            if not in_transcript or not stripped:
                continue
            if "]" in stripped:
                stripped = stripped.split("]", 1)[1].strip()
            return stripped[:120] or "Approved Moon delivery task"
        return "Approved Moon delivery task"


