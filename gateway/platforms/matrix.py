"""Matrix protocol platform adapter.

Connects to any Matrix homeserver using the matrix-nio async client.
Inbound messages arrive via the Matrix /sync long-poll endpoint.
Outbound messages use the Matrix Client-Server REST API.

Phase 1: Unencrypted rooms only (E2EE is a future enhancement).

Requires:
  - matrix-nio installed: pip install matrix-nio
  - MATRIX_HOMESERVER_URL and MATRIX_ACCESS_TOKEN environment variables set
  - MATRIX_USER_ID set to the bot's own Matrix ID (e.g. @hermes:matrix.org)
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — matrix-nio
# ---------------------------------------------------------------------------

try:
    from nio import (
        AsyncClient,
        AsyncClientConfig,
        DownloadResponse,
        RoomMessageText,
        RoomMessageImage,
        RoomMessageAudio,
        RoomMessageVideo,
        RoomMessageFile,
        RoomMessageNotice,
        RoomEncryptedImage,
        RoomEncryptedAudio,
        RoomEncryptedFile,
        RoomEncryptedVideo,
        SyncResponse,
        RoomSendResponse,
        UploadResponse,
        InviteEvent,
    )
    _NIO_AVAILABLE = True
except ImportError:
    _NIO_AVAILABLE = False
    # Stubs so the module still loads without matrix-nio installed
    AsyncClient = None
    AsyncClientConfig = None
    DownloadResponse = None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 32000    # Matrix has no hard limit; use a generous cap
SYNC_TIMEOUT_MS = 30_000      # 30 s long-poll
SYNC_RETRY_DELAY_INITIAL = 2.0
SYNC_RETRY_DELAY_MAX = 60.0
TYPING_TIMEOUT_MS = 10_000    # How long the typing indicator lasts (10 s)

# Regex for a Matrix user ID: @localpart:homeserver
_MATRIX_ID_RE = re.compile(r"@[^:]+:[a-zA-Z0-9.\-]+(:[0-9]+)?")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redact_matrix_id(mxid: str) -> str:
    """Redact a Matrix user ID for logging: @alice:example.org -> @al**:example.org."""
    if not mxid or not mxid.startswith("@"):
        return mxid or "<none>"
    at, rest = mxid[0], mxid[1:]
    colon = rest.find(":")
    if colon <= 0:
        return mxid
    local = rest[:colon]
    server = rest[colon:]  # includes the colon
    if len(local) <= 2:
        return f"{at}{'*' * len(local)}{server}"
    return f"{at}{local[:2]}{'*' * (len(local) - 2)}{server}"


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _is_image_type(content_type: str) -> bool:
    return content_type.startswith("image/")


def _is_audio_type(content_type: str) -> bool:
    return content_type.startswith("audio/")


def _is_video_type(content_type: str) -> bool:
    return content_type.startswith("video/")


def _markdown_to_html(text: str) -> Optional[str]:
    """Convert markdown to HTML for Matrix formatted messages.

    Returns None if the markdown library is not installed or the text has no
    markdown syntax worth converting.  Callers should fall back to plain-text
    body when this returns None.

    Matrix supports org.matrix.custom.html for rich formatting in clients
    such as Element and FluffyChat.
    """
    try:
        import markdown
        html = markdown.markdown(
            text,
            extensions=["fenced_code", "tables"],
        )
        return html
    except ImportError:
        return None


def check_matrix_requirements() -> bool:
    """Check if Matrix dependencies and credentials are available."""
    if not _NIO_AVAILABLE:
        return False
    return bool(
        os.getenv("MATRIX_HOMESERVER_URL")
        and os.getenv("MATRIX_ACCESS_TOKEN")
        and os.getenv("MATRIX_USER_ID")
    )


# ---------------------------------------------------------------------------
# Matrix Adapter
# ---------------------------------------------------------------------------

class MatrixAdapter(BasePlatformAdapter):
    """Matrix protocol adapter using matrix-nio AsyncClient.

    Connects to a Matrix homeserver and listens for messages via the /sync
    long-poll endpoint.  Sends messages and uploads media using the Matrix
    Client-Server REST API.

    Unencrypted rooms only in Phase 1.  E2EE support (via libolm) is a
    planned Phase 2 enhancement.
    """

    platform = Platform.MATRIX

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATRIX)

        extra = config.extra or {}
        self.homeserver_url = extra.get("homeserver_url", "").rstrip("/")
        self.access_token = config.token or ""
        self.user_id = extra.get("user_id", "")
        self.verify_ssl = extra.get("verify_ssl", True)

        # nio client (created in connect())
        self._client: Optional["AsyncClient"] = None

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._next_batch: Optional[str] = None  # pagination token from last sync

        # Event deduplication — Matrix can re-deliver events on reconnect.
        # Keep seen event IDs for one cleanup interval (60 s).
        self._seen_event_ids: Dict[str, float] = {}
        self._seen_cleanup_interval = 60.0

        logger.info(
            "Matrix adapter initialized: homeserver=%s user=%s ssl=%s",
            self.homeserver_url,
            _redact_matrix_id(self.user_id),
            self.verify_ssl,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the Matrix homeserver and start sync loop."""
        if not _NIO_AVAILABLE:
            logger.error("Matrix: matrix-nio is not installed. Run: pip install matrix-nio")
            return False

        if not self.homeserver_url or not self.access_token or not self.user_id:
            logger.error(
                "Matrix: MATRIX_HOMESERVER_URL, MATRIX_ACCESS_TOKEN, and "
                "MATRIX_USER_ID are all required"
            )
            return False

        # nio uses ssl=True to verify, ssl=False to skip (self-signed homeservers)
        ssl_context: Any = True if self.verify_ssl else False
        if not self.verify_ssl:
            logger.warning(
                "Matrix: SSL verification disabled — only suitable for private homeservers"
            )

        config = AsyncClientConfig(
            store_sync_tokens=False,
            encryption_enabled=False,  # Phase 1: no E2EE
        )

        self._client = AsyncClient(
            self.homeserver_url,
            self.user_id,
            config=config,
            ssl=ssl_context,
        )
        self._client.access_token = self.access_token

        # Verify connectivity with a simple whoami call
        try:
            resp = await self._client.whoami()
            if hasattr(resp, "user_id"):
                logger.info(
                    "Matrix: authenticated as %s",
                    _redact_matrix_id(resp.user_id),
                )
            else:
                logger.error("Matrix: whoami failed: %s", resp)
                await self._client.close()
                self._client = None
                return False
        except Exception as e:
            logger.error(
                "Matrix: cannot reach homeserver at %s: %s", self.homeserver_url, e
            )
            if self._client:
                await self._client.close()
                self._client = None
            return False

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info("Matrix: connected to %s", self.homeserver_url)
        return True

    async def disconnect(self) -> None:
        """Stop sync loop and close connection."""
        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        if self._client:
            await self._client.close()
            self._client = None

        logger.info("Matrix: disconnected")

    # ------------------------------------------------------------------
    # Sync Loop (inbound messages)
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Long-poll Matrix /sync endpoint and dispatch room events."""
        backoff = SYNC_RETRY_DELAY_INITIAL
        last_cleanup = time.time()

        while self._running:
            try:
                response = await self._client.sync(
                    timeout=SYNC_TIMEOUT_MS,
                    since=self._next_batch,
                    full_state=False,
                )

                if not isinstance(response, SyncResponse):
                    logger.warning(
                        "Matrix sync: unexpected response type: %s", type(response)
                    )
                    jitter = backoff * 0.2 * random.random()
                    await asyncio.sleep(backoff + jitter)
                    backoff = min(backoff * 2, SYNC_RETRY_DELAY_MAX)
                    continue

                backoff = SYNC_RETRY_DELAY_INITIAL  # reset on success
                self._next_batch = response.next_batch

                # Process room timeline events
                for room_id, room_info in response.rooms.join.items():
                    for event in room_info.timeline.events:
                        try:
                            await self._handle_room_event(room_id, event)
                        except Exception:
                            logger.exception(
                                "Matrix: error handling event in %s", room_id
                            )

                # Auto-join invited rooms (only if inviter is authorized)
                for room_id, invite_info in response.rooms.invite.items():
                    inviter = _extract_inviter(invite_info)
                    await self._handle_invite(room_id, inviter)

                # Periodic cleanup of seen event IDs
                now = time.time()
                if now - last_cleanup > self._seen_cleanup_interval:
                    cutoff = now - self._seen_cleanup_interval
                    self._seen_event_ids = {
                        k: v
                        for k, v in self._seen_event_ids.items()
                        if v > cutoff
                    }
                    last_cleanup = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.warning(
                        "Matrix sync error: %s (retrying in %.0fs)", e, backoff
                    )
                    jitter = backoff * 0.2 * random.random()
                    await asyncio.sleep(backoff + jitter)
                    backoff = min(backoff * 2, SYNC_RETRY_DELAY_MAX)

    async def _handle_invite(self, room_id: str, inviter: Optional[str]) -> None:
        """Accept a room invite — only if the inviter passes authorization.

        The full _is_user_authorized check lives in GatewayRunner, but we do
        a lightweight pre-check here: if MATRIX_ALLOWED_USERS or
        MATRIX_ALLOW_ALL_USERS is set, we can validate before joining.
        If neither is configured we defer to the gateway-level check that
        runs on the first message received in the room.
        """
        allow_all = os.getenv("MATRIX_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")
        allowed_raw = os.getenv("MATRIX_ALLOWED_USERS", "").strip()
        allowed_set = set(_parse_comma_list(allowed_raw)) if allowed_raw else None

        if not allow_all and allowed_set is not None:
            if not inviter or inviter not in allowed_set:
                logger.info(
                    "Matrix: rejecting invite to %s from unauthorized inviter %s",
                    room_id,
                    _redact_matrix_id(inviter or "<unknown>"),
                )
                # Optionally leave/reject — for now just skip the join
                return

        try:
            resp = await self._client.join(room_id)
            if hasattr(resp, "room_id"):
                logger.info("Matrix: joined room %s", room_id)
            else:
                logger.warning(
                    "Matrix: failed to join room %s: %s", room_id, resp
                )
        except Exception as e:
            logger.warning("Matrix: error joining room %s: %s", room_id, e)

    # ------------------------------------------------------------------
    # Event Handling
    # ------------------------------------------------------------------

    async def _handle_room_event(self, room_id: str, event: Any) -> None:
        """Process a single Matrix room event."""
        # Only handle message events
        if not isinstance(
            event,
            (
                RoomMessageText,
                RoomMessageImage,
                RoomMessageAudio,
                RoomMessageVideo,
                RoomMessageFile,
                RoomMessageNotice,
                RoomEncryptedImage,
                RoomEncryptedAudio,
                RoomEncryptedFile,
                RoomEncryptedVideo,
            ),
        ):
            return

        # Filter self-messages to prevent reply loops
        sender = getattr(event, "sender", "")
        if sender == self.user_id:
            return

        # Event deduplication — skip events we've already processed
        event_id = getattr(event, "event_id", "")
        if event_id and event_id in self._seen_event_ids:
            return
        if event_id:
            self._seen_event_ids[event_id] = time.time()

        # Extract text content and determine message type
        text = ""
        msg_type = MessageType.TEXT
        media_urls: List[str] = []
        media_types: List[str] = []

        if isinstance(event, (RoomMessageText, RoomMessageNotice)):
            text = getattr(event, "body", "") or ""
            msg_type = MessageType.TEXT

        elif isinstance(event, RoomMessageImage):
            text = getattr(event, "body", "") or ""
            msg_type = MessageType.PHOTO
            await self._fetch_and_cache_media(event, media_urls, media_types)

        elif isinstance(event, RoomMessageAudio):
            text = getattr(event, "body", "") or ""
            msg_type = MessageType.VOICE
            await self._fetch_and_cache_media(event, media_urls, media_types)

        elif isinstance(event, RoomMessageVideo):
            text = getattr(event, "body", "") or ""
            msg_type = MessageType.VIDEO
            await self._fetch_and_cache_media(event, media_urls, media_types)

        elif isinstance(event, RoomMessageFile):
            text = getattr(event, "body", "") or ""
            msg_type = MessageType.DOCUMENT
            await self._fetch_and_cache_media(event, media_urls, media_types)

        elif isinstance(
            event, (RoomEncryptedImage, RoomEncryptedAudio, RoomEncryptedFile, RoomEncryptedVideo)
        ):
            # E2EE messages — Phase 1 cannot decrypt
            logger.debug(
                "Matrix: received encrypted message in %s — E2EE not supported (Phase 1)",
                room_id,
            )
            return

        if not text and not media_urls:
            return

        # Determine room type (DM = 2 members, group = 3+)
        chat_type = _get_room_type(self._client, room_id)
        sender_name = _get_display_name(self._client, sender)
        room_name = _get_room_name(self._client, room_id)

        # Parse event timestamp (milliseconds since epoch)
        ts_ms = getattr(event, "server_timestamp", 0) or 0
        if ts_ms:
            try:
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                timestamp = datetime.now(tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        source = self.build_source(
            chat_id=room_id,
            chat_name=room_name or room_id,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender_name or sender,
        )

        message_event = MessageEvent(
            source=source,
            text=text,
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            message_id=event_id,
            timestamp=timestamp,
        )

        logger.debug(
            "Matrix: message from %s in %s: %s",
            _redact_matrix_id(sender),
            room_id,
            (text or "")[:60],
        )

        await self.handle_message(message_event)

    # ------------------------------------------------------------------
    # Media Handling
    # ------------------------------------------------------------------

    async def _fetch_and_cache_media(
        self,
        event: Any,
        media_urls: List[str],
        media_types: List[str],
    ) -> None:
        """Download media from an mxc:// URL and cache it locally."""
        url = getattr(event, "url", None)
        if not url or not url.startswith("mxc://"):
            return

        try:
            # Determine content type from the event's info dict
            info = getattr(event, "info", None) or {}
            mimetype = info.get("mimetype", "") or ""

            # nio download() returns a single DownloadResponse (not a tuple)
            response = await self._client.download(url)
            if not isinstance(response, DownloadResponse):
                logger.warning("Matrix: media download failed for %s: %s", url, response)
                return

            data = response.body
            content_type = mimetype or getattr(response, "content_type", "") or "application/octet-stream"
            filename = getattr(event, "body", "") or "attachment"

            # Cache based on content type
            if _is_image_type(content_type):
                ext = _ext_from_mime(content_type)
                path = cache_image_from_bytes(data, ext)
            elif _is_audio_type(content_type):
                ext = _ext_from_mime(content_type)
                path = cache_audio_from_bytes(data, ext)
            else:
                suffix = Path(filename).suffix or _ext_from_mime(content_type)
                path = cache_document_from_bytes(data, suffix)

            if path:
                media_urls.append(path)
                media_types.append(content_type)

        except Exception as e:
            logger.warning("Matrix: failed to fetch media %s: %s", url, e)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to a Matrix room."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        # Build content block with optional HTML formatting
        html = _markdown_to_html(content)
        msg_content: Dict[str, Any] = {
            "msgtype": "m.text",
            "body": content,
        }
        if html:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = html

        # Thread / reply support (metadata may carry {"reply_to_event_id": "..."})
        reply_event_id = reply_to or (metadata or {}).get("reply_to_event_id")
        if reply_event_id:
            msg_content["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_event_id}
            }

        try:
            resp = await self._client.room_send(
                room_id=chat_id,
                message_type="m.room.message",
                content=msg_content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, RoomSendResponse):
                return SendResult(success=True, message_id=resp.event_id)
            logger.warning("Matrix send failed in %s: %s", chat_id, resp)
            return SendResult(success=False, error=str(resp))
        except Exception as e:
            logger.warning("Matrix send error in %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator to a Matrix room."""
        if not self._client:
            return
        try:
            await self._client.room_typing(
                room_id=chat_id,
                typing_state=True,
                timeout=TYPING_TIMEOUT_MS,
            )
        except Exception as e:
            logger.debug(
                "Matrix typing indicator failed in %s: %s", chat_id, e
            )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image to a Matrix room by uploading to the media repo."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        # Resolve local path (support file:// and bare paths)
        file_path = image_url[7:] if image_url.startswith("file://") else image_url

        path = Path(file_path)
        if not path.exists():
            return SendResult(
                success=False, error=f"Image file not found: {file_path}"
            )

        try:
            data = path.read_bytes()
            mime = _mime_from_path(path)

            upload_resp = await self._client.upload(
                data,
                content_type=mime,
                filename=path.name,
                filesize=len(data),
            )

            if isinstance(upload_resp, tuple):
                upload_resp, _ = upload_resp
            if not isinstance(upload_resp, UploadResponse):
                return SendResult(
                    success=False, error=f"Upload failed: {upload_resp}"
                )

            mxc_uri = upload_resp.content_uri
            msg_content: Dict[str, Any] = {
                "msgtype": "m.image",
                "body": caption or path.name,
                "url": mxc_uri,
                "info": {"mimetype": mime, "size": len(data)},
            }

            resp = await self._client.room_send(
                room_id=chat_id,
                message_type="m.room.message",
                content=msg_content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, RoomSendResponse):
                return SendResult(success=True, message_id=resp.event_id)
            return SendResult(success=False, error=str(resp))

        except Exception as e:
            logger.warning("Matrix send_image error in %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image from a local file path."""
        return await self.send_image(
            chat_id, image_path, caption=caption, reply_to=reply_to
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file to a Matrix room."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        path = Path(file_path)
        if not path.exists():
            return SendResult(success=False, error=f"File not found: {file_path}")

        try:
            data = path.read_bytes()
            mime = _mime_from_path(path)
            display_name = file_name or path.name

            upload_resp = await self._client.upload(
                data,
                content_type=mime,
                filename=display_name,
                filesize=len(data),
            )

            if isinstance(upload_resp, tuple):
                upload_resp, _ = upload_resp
            if not isinstance(upload_resp, UploadResponse):
                return SendResult(
                    success=False, error=f"Upload failed: {upload_resp}"
                )

            mxc_uri = upload_resp.content_uri
            msg_content: Dict[str, Any] = {
                "msgtype": "m.file",
                "body": caption or display_name,
                "filename": display_name,
                "url": mxc_uri,
                "info": {"mimetype": mime, "size": len(data)},
            }

            resp = await self._client.room_send(
                room_id=chat_id,
                message_type="m.room.message",
                content=msg_content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, RoomSendResponse):
                return SendResult(success=True, message_id=resp.event_id)
            return SendResult(success=False, error=str(resp))

        except Exception as e:
            logger.warning("Matrix send_document error in %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    async def _send_media_as_type(
        self,
        chat_id: str,
        file_path: str,
        msgtype: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Generic helper: upload a file and send it with the given msgtype."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        path = Path(file_path)
        if not path.exists():
            return SendResult(success=False, error=f"File not found: {file_path}")

        try:
            data = path.read_bytes()
            mime = _mime_from_path(path)

            upload_resp = await self._client.upload(
                data,
                content_type=mime,
                filename=path.name,
                filesize=len(data),
            )

            if isinstance(upload_resp, tuple):
                upload_resp, _ = upload_resp
            if not isinstance(upload_resp, UploadResponse):
                return SendResult(
                    success=False, error=f"Upload failed: {upload_resp}"
                )

            mxc_uri = upload_resp.content_uri
            msg_content: Dict[str, Any] = {
                "msgtype": msgtype,
                "body": caption or path.name,
                "url": mxc_uri,
                "info": {"mimetype": mime, "size": len(data)},
            }

            resp = await self._client.room_send(
                room_id=chat_id,
                message_type="m.room.message",
                content=msg_content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, RoomSendResponse):
                return SendResult(success=True, message_id=resp.event_id)
            return SendResult(success=False, error=str(resp))

        except Exception as e:
            logger.warning("Matrix send %s error in %s: %s", msgtype, chat_id, e)
            return SendResult(success=False, error=str(e))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a voice/audio message to a Matrix room."""
        return await self._send_media_as_type(chat_id, audio_path, "m.audio", caption)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video to a Matrix room."""
        return await self._send_media_as_type(chat_id, video_path, "m.video", caption)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """Edit a previously sent message using Matrix event replacement.

        Sends a new m.room.message with m.relates_to rel_type=m.replace, which
        clients supporting Matrix MSC2676 (merged into spec) will show as an edit.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")

        html = _markdown_to_html(content)
        new_content: Dict[str, Any] = {
            "msgtype": "m.text",
            "body": content,
        }
        if html:
            new_content["format"] = "org.matrix.custom.html"
            new_content["formatted_body"] = html

        # Per spec, the top-level body must be "* <new content>" for fallback
        msg_content: Dict[str, Any] = {
            "msgtype": "m.text",
            "body": f"* {content}",
            "m.new_content": new_content,
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": message_id,
            },
        }
        if html:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = f"* {html}"

        try:
            resp = await self._client.room_send(
                room_id=chat_id,
                message_type="m.room.message",
                content=msg_content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, RoomSendResponse):
                return SendResult(success=True, message_id=resp.event_id)
            return SendResult(success=False, error=str(resp))
        except Exception as e:
            logger.warning("Matrix edit_message error in %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Chat Info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return name, type, and ID for a Matrix room."""
        return {
            "name": _get_room_name(self._client, chat_id),
            "type": _get_room_type(self._client, chat_id),
            "chat_id": chat_id,
        }


# ---------------------------------------------------------------------------
# Room helpers (synchronous — nio room state is cached locally)
# ---------------------------------------------------------------------------

def _get_room_type(client: Any, room_id: str) -> str:
    """Determine if a room is a DM (2 members) or a group chat."""
    if client is None:
        return "group"
    try:
        room = client.rooms.get(room_id)
        if room:
            return "dm" if len(room.users) <= 2 else "group"
    except Exception:
        pass
    return "group"


def _get_room_name(client: Any, room_id: str) -> str:
    """Get the human-readable display name for a room."""
    if client is None:
        return room_id
    try:
        room = client.rooms.get(room_id)
        if room and room.display_name:
            return room.display_name
    except Exception:
        pass
    return room_id


def _get_display_name(client: Any, user_id: str) -> str:
    """Get the display name for a user from cached room member state."""
    if client is None:
        return _localpart(user_id)
    try:
        for room in client.rooms.values():
            member = room.users.get(user_id)
            if member and member.display_name:
                return member.display_name
    except Exception:
        pass
    return _localpart(user_id)


def _localpart(user_id: str) -> str:
    """Extract the localpart from a Matrix user ID (@local:server -> local)."""
    match = re.match(r"@([^:]+):", user_id)
    return match.group(1) if match else user_id


def _extract_inviter(invite_info: Any) -> Optional[str]:
    """Extract the inviter's user ID from an InviteInfo object."""
    try:
        for event in invite_info.invite_state.events:
            if getattr(event, "type", "") == "m.room.member":
                content = getattr(event, "content", {}) or {}
                if content.get("membership") == "invite":
                    return getattr(event, "sender", None)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# MIME / extension helpers (no external deps)
# ---------------------------------------------------------------------------

_EXT_TO_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ogg": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".txt": "text/plain",
}

_MIME_TO_EXT: Dict[str, str] = {v: k for k, v in _EXT_TO_MIME.items()}


def _ext_from_mime(mime: str) -> str:
    """Return a file extension for a MIME type."""
    return _MIME_TO_EXT.get(mime.lower(), ".bin")


def _mime_from_path(path: Path) -> str:
    """Return MIME type for a file based on its extension."""
    return _EXT_TO_MIME.get(path.suffix.lower(), "application/octet-stream")
