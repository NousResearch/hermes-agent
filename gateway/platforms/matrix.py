"""Matrix gateway adapter.

Connects to any Matrix homeserver (self-hosted or matrix.org) via the
matrix-nio Python SDK.  Supports optional end-to-end encryption (E2EE)
when installed with ``pip install "matrix-nio[e2e]"``.

Environment variables:
    MATRIX_HOMESERVER       Homeserver URL (e.g. https://matrix.example.org)
    MATRIX_ACCESS_TOKEN     Access token (preferred auth method)
    MATRIX_USER_ID          Full user ID (@bot:server) — required for password login
    MATRIX_PASSWORD         Password (alternative to access token)
    MATRIX_ENCRYPTION       Set "true" to enable E2EE
    MATRIX_ALLOWED_USERS    Comma-separated Matrix user IDs (@user:server)
    MATRIX_HOME_ROOM        Room ID for cron/notification delivery
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Matrix message size limit (4000 chars practical, spec has no hard limit
# but clients render poorly above this).
MAX_MESSAGE_LENGTH = 4000

# Store directory for E2EE keys and sync state.
_STORE_DIR = Path.home() / ".hermes" / "matrix" / "store"

# Grace period: ignore messages older than this many seconds before startup.
_STARTUP_GRACE_SECONDS = 5


def check_matrix_requirements() -> bool:
    """Return True if the Matrix adapter can be used."""
    token = os.getenv("MATRIX_ACCESS_TOKEN", "")
    password = os.getenv("MATRIX_PASSWORD", "")
    homeserver = os.getenv("MATRIX_HOMESERVER", "")

    if not token and not password:
        logger.debug("Matrix: neither MATRIX_ACCESS_TOKEN nor MATRIX_PASSWORD set")
        return False
    if not homeserver:
        logger.warning("Matrix: MATRIX_HOMESERVER not set")
        return False
    try:
        import nio  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "Matrix: matrix-nio not installed. "
            "Run: pip install 'matrix-nio[e2e]'"
        )
        return False


class MatrixAdapter(BasePlatformAdapter):
    """Gateway adapter for Matrix (any homeserver)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATRIX)

        self._homeserver: str = (
            config.extra.get("homeserver", "")
            or os.getenv("MATRIX_HOMESERVER", "")
        ).rstrip("/")
        self._access_token: str = config.token or os.getenv("MATRIX_ACCESS_TOKEN", "")
        self._user_id: str = (
            config.extra.get("user_id", "")
            or os.getenv("MATRIX_USER_ID", "")
        )
        self._password: str = (
            config.extra.get("password", "")
            or os.getenv("MATRIX_PASSWORD", "")
        )
        self._encryption: bool = config.extra.get(
            "encryption",
            os.getenv("MATRIX_ENCRYPTION", "").lower() in ("true", "1", "yes"),
        )

        self._client: Any = None  # nio.AsyncClient
        self._sync_task: Optional[asyncio.Task] = None
        self._closing = False
        self._startup_ts: float = 0.0

        # Cache: room_id → bool (is DM)
        self._dm_rooms: Dict[str, bool] = {}
        # Set of room IDs we've joined
        self._joined_rooms: Set[str] = set()

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the Matrix homeserver and start syncing."""
        import nio

        if not self._homeserver:
            logger.error("Matrix: homeserver URL not configured")
            return False

        # Determine store path and ensure it exists.
        store_path = str(_STORE_DIR)
        _STORE_DIR.mkdir(parents=True, exist_ok=True)

        # Create the client.
        if self._encryption:
            try:
                client = nio.AsyncClient(
                    self._homeserver,
                    self._user_id or "",
                    store_path=store_path,
                )
                logger.info("Matrix: E2EE enabled (store: %s)", store_path)
            except Exception as exc:
                logger.warning(
                    "Matrix: failed to create E2EE client (%s), "
                    "falling back to plain client. Install: "
                    "pip install 'matrix-nio[e2e]'",
                    exc,
                )
                client = nio.AsyncClient(self._homeserver, self._user_id or "")
        else:
            client = nio.AsyncClient(self._homeserver, self._user_id or "")

        self._client = client

        # Authenticate.
        if self._access_token:
            client.access_token = self._access_token
            # Resolve user_id if not set.
            if not self._user_id:
                resp = await client.whoami()
                if isinstance(resp, nio.WhoamiResponse):
                    self._user_id = resp.user_id
                    client.user_id = resp.user_id
                    logger.info("Matrix: authenticated as %s", self._user_id)
                else:
                    logger.error(
                        "Matrix: whoami failed — check MATRIX_ACCESS_TOKEN and MATRIX_HOMESERVER"
                    )
                    await client.close()
                    return False
            else:
                client.user_id = self._user_id
                logger.info("Matrix: using access token for %s", self._user_id)
        elif self._password and self._user_id:
            resp = await client.login(
                self._password,
                device_name="Hermes Agent",
            )
            if isinstance(resp, nio.LoginResponse):
                logger.info("Matrix: logged in as %s", self._user_id)
            else:
                logger.error("Matrix: login failed — %s", getattr(resp, "message", resp))
                await client.close()
                return False
        else:
            logger.error("Matrix: need MATRIX_ACCESS_TOKEN or MATRIX_USER_ID + MATRIX_PASSWORD")
            await client.close()
            return False

        # If E2EE is enabled, load the crypto store.
        if self._encryption and hasattr(client, "olm"):
            try:
                if client.should_upload_keys:
                    await client.keys_upload()
                logger.info("Matrix: E2EE crypto initialized")
            except Exception as exc:
                logger.warning("Matrix: crypto init issue: %s", exc)

        # Register event callbacks.
        client.add_event_callback(self._on_room_message, nio.RoomMessageText)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageMedia)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageImage)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageAudio)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageVideo)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageFile)
        client.add_event_callback(self._on_invite, nio.InviteMemberEvent)

        # If E2EE: handle encrypted events.
        if self._encryption and hasattr(client, "olm"):
            client.add_event_callback(
                self._on_room_message, nio.MegolmEvent
            )

        # Initial sync to catch up, then start background sync.
        self._startup_ts = time.time()
        self._closing = False

        # Do an initial sync to populate room state.
        resp = await client.sync(timeout=10000, full_state=True)
        if isinstance(resp, nio.SyncResponse):
            self._joined_rooms = set(resp.rooms.join.keys())
            logger.info(
                "Matrix: initial sync complete, joined %d rooms",
                len(self._joined_rooms),
            )
            # Build DM room cache from m.direct account data.
            await self._refresh_dm_cache()
        else:
            logger.warning("Matrix: initial sync returned %s", type(resp).__name__)

        # Start the sync loop.
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Disconnect from Matrix."""
        self._closing = True

        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._client:
            await self._client.close()
            self._client = None

        logger.info("Matrix: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a Matrix room."""
        import nio

        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)

        last_event_id = None
        for chunk in chunks:
            msg_content: Dict[str, Any] = {
                "msgtype": "m.text",
                "body": chunk,
            }

            # Convert markdown to HTML for rich rendering.
            html = self._markdown_to_html(chunk)
            if html and html != chunk:
                msg_content["format"] = "org.matrix.custom.html"
                msg_content["formatted_body"] = html

            # Reply-to support.
            if reply_to:
                msg_content["m.relates_to"] = {
                    "m.in_reply_to": {"event_id": reply_to}
                }

            # Thread support: if metadata has thread_id, send as threaded reply.
            thread_id = (metadata or {}).get("thread_id")
            if thread_id:
                relates_to = msg_content.get("m.relates_to", {})
                relates_to["rel_type"] = "m.thread"
                relates_to["event_id"] = thread_id
                relates_to["is_falling_back"] = True
                if reply_to and "m.in_reply_to" not in relates_to:
                    relates_to["m.in_reply_to"] = {"event_id": reply_to}
                msg_content["m.relates_to"] = relates_to

            resp = await self._client.room_send(
                chat_id,
                "m.room.message",
                msg_content,
            )
            if isinstance(resp, nio.RoomSendResponse):
                last_event_id = resp.event_id
            else:
                err = getattr(resp, "message", str(resp))
                logger.error("Matrix: failed to send to %s: %s", chat_id, err)
                return SendResult(success=False, error=err)

        return SendResult(success=True, message_id=last_event_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return room name and type (dm/group)."""
        name = chat_id
        chat_type = "group"

        if self._client:
            room = self._client.rooms.get(chat_id)
            if room:
                name = room.display_name or room.canonical_alias or chat_id
                # Use DM cache.
                if self._dm_rooms.get(chat_id, False):
                    chat_type = "dm"
                elif room.member_count == 2:
                    chat_type = "dm"

        return {"name": name, "type": chat_type}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a typing indicator."""
        if self._client:
            try:
                await self._client.room_typing(chat_id, typing_state=True, timeout=30000)
            except Exception:
                pass

    async def edit_message(
        self, chat_id: str, message_id: str, content: str
    ) -> SendResult:
        """Edit an existing message (via m.replace)."""
        import nio

        formatted = self.format_message(content)
        msg_content: Dict[str, Any] = {
            "msgtype": "m.text",
            "body": f"* {formatted}",
            "m.new_content": {
                "msgtype": "m.text",
                "body": formatted,
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": message_id,
            },
        }

        html = self._markdown_to_html(formatted)
        if html and html != formatted:
            msg_content["m.new_content"]["format"] = "org.matrix.custom.html"
            msg_content["m.new_content"]["formatted_body"] = html
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = f"* {html}"

        resp = await self._client.room_send(chat_id, "m.room.message", msg_content)
        if isinstance(resp, nio.RoomSendResponse):
            return SendResult(success=True, message_id=resp.event_id)
        return SendResult(success=False, error=getattr(resp, "message", str(resp)))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download an image URL and upload it to Matrix."""
        try:
            # Try aiohttp first (always available), fall back to httpx
            try:
                import aiohttp as _aiohttp
                async with _aiohttp.ClientSession() as http:
                    async with http.get(image_url, timeout=_aiohttp.ClientTimeout(total=30)) as resp:
                        resp.raise_for_status()
                        data = await resp.read()
                        ct = resp.content_type or "image/png"
                        fname = image_url.rsplit("/", 1)[-1].split("?")[0] or "image.png"
            except ImportError:
                import httpx
                async with httpx.AsyncClient() as http:
                    resp = await http.get(image_url, follow_redirects=True, timeout=30)
                    resp.raise_for_status()
                    data = resp.content
                    ct = resp.headers.get("content-type", "image/png")
                    fname = image_url.rsplit("/", 1)[-1].split("?")[0] or "image.png"
        except Exception as exc:
            logger.warning("Matrix: failed to download image %s: %s", image_url, exc)
            return await self.send(chat_id, f"{caption or ''}\n{image_url}".strip(), reply_to)

        return await self._upload_and_send(chat_id, data, fname, ct, "m.image", caption, reply_to, metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local image file to Matrix."""
        return await self._send_local_file(chat_id, image_path, "m.image", caption, reply_to, metadata=metadata)

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
        return await self._send_local_file(chat_id, file_path, "m.file", caption, reply_to, file_name, metadata)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload an audio file as a voice message."""
        return await self._send_local_file(chat_id, audio_path, "m.audio", caption, reply_to, metadata=metadata)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a video file."""
        return await self._send_local_file(chat_id, video_path, "m.video", caption, reply_to, metadata=metadata)

    def format_message(self, content: str) -> str:
        """Pass-through — Matrix supports standard Markdown natively."""
        # Strip image markdown; media is uploaded separately.
        content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)
        return content

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    async def _upload_and_send(
        self,
        room_id: str,
        data: bytes,
        filename: str,
        content_type: str,
        msgtype: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload bytes to Matrix and send as a media message."""
        import nio

        # Upload to homeserver.
        resp = await self._client.upload(
            data,
            content_type=content_type,
            filename=filename,
        )
        if not isinstance(resp, nio.UploadResponse):
            err = getattr(resp, "message", str(resp))
            logger.error("Matrix: upload failed: %s", err)
            return SendResult(success=False, error=err)

        mxc_url = resp.content_uri

        # Build media message content.
        msg_content: Dict[str, Any] = {
            "msgtype": msgtype,
            "body": caption or filename,
            "url": mxc_url,
            "info": {
                "mimetype": content_type,
                "size": len(data),
            },
        }

        if reply_to:
            msg_content["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_to}
            }

        thread_id = (metadata or {}).get("thread_id")
        if thread_id:
            relates_to = msg_content.get("m.relates_to", {})
            relates_to["rel_type"] = "m.thread"
            relates_to["event_id"] = thread_id
            relates_to["is_falling_back"] = True
            msg_content["m.relates_to"] = relates_to

        resp2 = await self._client.room_send(room_id, "m.room.message", msg_content)
        if isinstance(resp2, nio.RoomSendResponse):
            return SendResult(success=True, message_id=resp2.event_id)
        return SendResult(success=False, error=getattr(resp2, "message", str(resp2)))

    async def _send_local_file(
        self,
        room_id: str,
        file_path: str,
        msgtype: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        file_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Read a local file and upload it."""
        p = Path(file_path)
        if not p.exists():
            return await self.send(
                room_id, f"{caption or ''}\n(file not found: {file_path})", reply_to
            )

        fname = file_name or p.name
        ct = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        data = p.read_bytes()

        return await self._upload_and_send(room_id, data, fname, ct, msgtype, caption, reply_to, metadata)

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Continuously sync with the homeserver."""
        while not self._closing:
            try:
                await self._client.sync(timeout=30000)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    async def _on_room_message(self, room: Any, event: Any) -> None:
        """Handle incoming text messages (and decrypted megolm events)."""
        import nio

        # Ignore own messages.
        if event.sender == self._user_id:
            return

        # Startup grace: ignore old messages from initial sync.
        event_ts = getattr(event, "server_timestamp", 0) / 1000.0
        if event_ts and event_ts < self._startup_ts - _STARTUP_GRACE_SECONDS:
            return

        # Handle decrypted MegolmEvents — extract the inner event.
        if isinstance(event, nio.MegolmEvent):
            # Failed to decrypt.
            logger.warning(
                "Matrix: could not decrypt event %s in %s",
                event.event_id, room.room_id,
            )
            return

        # Skip edits (m.replace relation).
        source_content = getattr(event, "source", {}).get("content", {})
        relates_to = source_content.get("m.relates_to", {})
        if relates_to.get("rel_type") == "m.replace":
            return

        body = getattr(event, "body", "") or ""
        if not body:
            return

        # Determine chat type.
        is_dm = self._dm_rooms.get(room.room_id, False)
        if not is_dm and room.member_count == 2:
            is_dm = True
        chat_type = "dm" if is_dm else "group"

        # Thread support.
        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")

        # Reply-to detection.
        reply_to = None
        in_reply_to = relates_to.get("m.in_reply_to", {})
        if in_reply_to:
            reply_to = in_reply_to.get("event_id")

        # Strip reply fallback from body (Matrix prepends "> ..." lines).
        if reply_to and body.startswith("> "):
            lines = body.split("\n")
            stripped = []
            past_fallback = False
            for line in lines:
                if not past_fallback:
                    if line.startswith("> ") or line == ">":
                        continue
                    if line == "":
                        past_fallback = True
                        continue
                    past_fallback = True
                stripped.append(line)
            body = "\n".join(stripped) if stripped else body

        # Message type.
        msg_type = MessageType.TEXT
        if body.startswith("!") or body.startswith("/"):
            msg_type = MessageType.COMMAND

        source = self.build_source(
            chat_id=room.room_id,
            chat_type=chat_type,
            user_id=event.sender,
            user_name=self._get_display_name(room, event.sender),
            thread_id=thread_id,
        )

        msg_event = MessageEvent(
            text=body,
            message_type=msg_type,
            source=source,
            raw_message=getattr(event, "source", {}),
            message_id=event.event_id,
            reply_to=reply_to,
        )

        await self.handle_message(msg_event)

    async def _on_room_message_media(self, room: Any, event: Any) -> None:
        """Handle incoming media messages (images, audio, video, files)."""
        import nio

        # Ignore own messages.
        if event.sender == self._user_id:
            return

        # Startup grace.
        event_ts = getattr(event, "server_timestamp", 0) / 1000.0
        if event_ts and event_ts < self._startup_ts - _STARTUP_GRACE_SECONDS:
            return

        body = getattr(event, "body", "") or ""
        url = getattr(event, "url", "")

        # Convert mxc:// to HTTP URL for downstream processing.
        http_url = ""
        if url and url.startswith("mxc://"):
            http_url = self._mxc_to_http(url)

        # Determine message type from event class.
        media_type = "document"
        msg_type = MessageType.DOCUMENT
        if isinstance(event, nio.RoomMessageImage):
            msg_type = MessageType.PHOTO
            media_type = "image"
        elif isinstance(event, nio.RoomMessageAudio):
            msg_type = MessageType.AUDIO
            media_type = "audio"
        elif isinstance(event, nio.RoomMessageVideo):
            msg_type = MessageType.VIDEO
            media_type = "video"

        is_dm = self._dm_rooms.get(room.room_id, False)
        if not is_dm and room.member_count == 2:
            is_dm = True
        chat_type = "dm" if is_dm else "group"

        # Thread/reply detection.
        source_content = getattr(event, "source", {}).get("content", {})
        relates_to = source_content.get("m.relates_to", {})
        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")

        source = self.build_source(
            chat_id=room.room_id,
            chat_type=chat_type,
            user_id=event.sender,
            user_name=self._get_display_name(room, event.sender),
            thread_id=thread_id,
        )

        msg_event = MessageEvent(
            text=body,
            message_type=msg_type,
            source=source,
            raw_message=getattr(event, "source", {}),
            message_id=event.event_id,
            media_urls=[http_url] if http_url else None,
            media_types=[media_type] if http_url else None,
        )

        await self.handle_message(msg_event)

    async def _on_invite(self, room: Any, event: Any) -> None:
        """Auto-join rooms when invited."""
        import nio

        if not isinstance(event, nio.InviteMemberEvent):
            return

        # Only process invites directed at us.
        if event.state_key != self._user_id:
            return

        if event.membership != "invite":
            return

        logger.info(
            "Matrix: invited to %s by %s — joining",
            room.room_id, event.sender,
        )
        try:
            resp = await self._client.join(room.room_id)
            if isinstance(resp, nio.JoinResponse):
                self._joined_rooms.add(room.room_id)
                logger.info("Matrix: joined %s", room.room_id)
                # Refresh DM cache since new room may be a DM.
                await self._refresh_dm_cache()
            else:
                logger.warning(
                    "Matrix: failed to join %s: %s",
                    room.room_id, getattr(resp, "message", resp),
                )
        except Exception as exc:
            logger.warning("Matrix: error joining %s: %s", room.room_id, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _refresh_dm_cache(self) -> None:
        """Refresh the DM room cache from m.direct account data.

        Tries the account_data API first, then falls back to parsing
        the sync response's account_data for robustness.
        """
        if not self._client:
            return

        dm_data: Optional[Dict] = None

        # Primary: try the dedicated account data endpoint.
        try:
            resp = await self._client.get_account_data("m.direct")
            if hasattr(resp, "content"):
                dm_data = resp.content
            elif isinstance(resp, dict):
                dm_data = resp
        except Exception as exc:
            logger.debug("Matrix: get_account_data('m.direct') failed: %s — trying sync fallback", exc)

        # Fallback: parse from the client's account_data store (populated by sync).
        if dm_data is None:
            try:
                # matrix-nio stores account data events on the client object
                ad = getattr(self._client, "account_data", None)
                if ad and isinstance(ad, dict) and "m.direct" in ad:
                    event = ad["m.direct"]
                    if hasattr(event, "content"):
                        dm_data = event.content
                    elif isinstance(event, dict):
                        dm_data = event
            except Exception:
                pass

        if dm_data is None:
            return

        dm_room_ids: Set[str] = set()
        for user_id, rooms in dm_data.items():
            if isinstance(rooms, list):
                dm_room_ids.update(rooms)

        self._dm_rooms = {
            rid: (rid in dm_room_ids)
            for rid in self._joined_rooms
        }

    def _get_display_name(self, room: Any, user_id: str) -> str:
        """Get a user's display name in a room, falling back to user_id."""
        if room and hasattr(room, "users"):
            user = room.users.get(user_id)
            if user and getattr(user, "display_name", None):
                return user.display_name
        # Strip the @...:server format to just the localpart.
        if user_id.startswith("@") and ":" in user_id:
            return user_id[1:].split(":")[0]
        return user_id

    def _mxc_to_http(self, mxc_url: str) -> str:
        """Convert mxc://server/media_id to an HTTP download URL."""
        # mxc://matrix.org/abc123 → https://matrix.org/_matrix/client/v1/media/download/matrix.org/abc123
        # Uses the authenticated client endpoint (spec v1.11+) instead of the
        # deprecated /_matrix/media/v3/download/ path.
        if not mxc_url.startswith("mxc://"):
            return mxc_url
        parts = mxc_url[6:]  # strip mxc://
        # Use our homeserver for download (federation handles the rest).
        return f"{self._homeserver}/_matrix/client/v1/media/download/{parts}"

    def _markdown_to_html(self, text: str) -> str:
        """Convert Markdown to Matrix-compatible HTML.

        Uses a simple conversion for common patterns.  For full fidelity
        a markdown-it style library could be used, but this covers the
        common cases without an extra dependency.
        """
        try:
            import markdown
            html = markdown.markdown(
                text,
                extensions=["fenced_code", "tables", "nl2br"],
            )
            # Strip wrapping <p> tags for single-paragraph messages.
            if html.count("<p>") == 1:
                html = html.replace("<p>", "").replace("</p>", "")
            return html
        except ImportError:
            pass

        # Minimal fallback: just handle bold, italic, code.
        html = text
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
        html = re.sub(r"\n", r"<br>", html)
        return html
