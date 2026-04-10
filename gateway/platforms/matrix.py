"""Matrix gateway adapter.

Connects to any Matrix homeserver (self-hosted or matrix.org) via the
matrix-nio Python SDK.  Supports optional end-to-end encryption (E2EE)
when installed with ``pip install "matrix-nio[e2e]"``.

Environment variables:
    MATRIX_HOMESERVER           Homeserver URL (e.g. https://matrix.example.org)
    MATRIX_ACCESS_TOKEN         Access token (preferred auth method)
    MATRIX_USER_ID              Full user ID (@bot:server) — required for password login
    MATRIX_PASSWORD             Password (alternative to access token)
    MATRIX_ENCRYPTION           Set "true" to enable E2EE
    MATRIX_DEVICE_ID            Stable device ID for E2EE persistence across restarts
    MATRIX_ALLOWED_USERS    Comma-separated Matrix user IDs (@user:server)
    MATRIX_HOME_ROOM        Room ID for cron/notification delivery
    MATRIX_REACTIONS        Set "false" to disable processing lifecycle reactions
                            (eyes/checkmark/cross). Default: true
    MATRIX_REQUIRE_MENTION      Require @mention in rooms (default: true)
    MATRIX_FREE_RESPONSE_ROOMS  Comma-separated room IDs exempt from mention requirement
    MATRIX_AUTO_THREAD          Auto-create threads for room messages (default: true)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from html import escape as _html_escape

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)

# Matrix message size limit (4000 chars practical, spec has no hard limit
# but clients render poorly above this).
MAX_MESSAGE_LENGTH = 4000

# Store directory for E2EE keys and sync state.
# Uses get_hermes_home() so each profile gets its own Matrix store.
from hermes_constants import get_hermes_dir as _get_hermes_dir
_STORE_DIR = _get_hermes_dir("platforms/matrix/store", "matrix/store")

# Grace period: ignore messages older than this many seconds before startup.
_STARTUP_GRACE_SECONDS = 5

# E2EE key export file for persistence across restarts.
_KEY_EXPORT_FILE = _STORE_DIR / "exported_keys.txt"
_KEY_EXPORT_PASSPHRASE = "hermes-matrix-e2ee-keys"

# Pending undecrypted events: cap and TTL for retry buffer.
_MAX_PENDING_EVENTS = 100
_PENDING_EVENT_TTL = 300  # seconds — stop retrying after 5 min


_E2EE_INSTALL_HINT = (
    "Install with: pip install 'matrix-nio[e2e]'  "
    "(requires libolm C library)"
)


def _check_e2ee_deps() -> bool:
    """Return True if matrix-nio E2EE dependencies (python-olm) are available."""
    try:
        from nio.crypto import ENCRYPTION_ENABLED
        return bool(ENCRYPTION_ENABLED)
    except (ImportError, AttributeError):
        return False


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
    except ImportError:
        logger.warning(
            "Matrix: matrix-nio not installed. "
            "Run: pip install 'matrix-nio[e2e]'"
        )
        return False

    # If encryption is requested, verify E2EE deps are available at startup
    # rather than silently degrading to plaintext-only at connect time.
    encryption_requested = os.getenv("MATRIX_ENCRYPTION", "").lower() in ("true", "1", "yes")
    if encryption_requested and not _check_e2ee_deps():
        logger.error(
            "Matrix: MATRIX_ENCRYPTION=true but E2EE dependencies are missing. %s. "
            "Without this, encrypted rooms will not work. "
            "Set MATRIX_ENCRYPTION=false to disable E2EE.",
            _E2EE_INSTALL_HINT,
        )
        return False

    return True


class MatrixAdapter(BasePlatformAdapter):
    """Gateway adapter for Matrix (any homeserver)."""

    # Threshold for detecting Matrix client-side message splits.
    # When a chunk is near the ~4000-char practical limit, a continuation
    # is almost certain.
    _SPLIT_THRESHOLD = 3900

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
        self._device_id: str = (
            config.extra.get("device_id", "")
            or os.getenv("MATRIX_DEVICE_ID", "")
        )

        self._client: Any = None  # nio.AsyncClient
        self._sync_task: Optional[asyncio.Task] = None
        self._closing = False
        self._startup_ts: float = 0.0

        # Cache: room_id → bool (is DM)
        self._dm_rooms: Dict[str, bool] = {}
        # Set of room IDs we've joined
        self._joined_rooms: Set[str] = set()
        # Event deduplication (bounded deque keeps newest entries)
        from collections import deque
        self._processed_events: deque = deque(maxlen=1000)
        self._processed_events_set: set = set()

        # Buffer for undecrypted events pending key receipt.
        # Each entry: (room, event, timestamp)
        self._pending_megolm: list = []

        # Thread participation tracking (for require_mention bypass)
        self._bot_participated_threads: set = self._load_participated_threads()
        self._MAX_TRACKED_THREADS = 500

        # Reactions: configurable via MATRIX_REACTIONS (default: true).
        self._reactions_enabled: bool = os.getenv(
            "MATRIX_REACTIONS", "true"
        ).lower() not in ("false", "0", "no")
        # Tracks the reaction event_id for in-progress (eyes) reactions.
        # Key: (room_id, message_event_id) → reaction_event_id (for the eyes reaction).
        self._pending_reactions: dict[tuple[str, str], str] = {}

        # Text batching: merge rapid successive messages (Telegram-style).
        # Matrix clients split long messages around 4000 chars.
        self._text_batch_delay_seconds = float(os.getenv("HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS", "0.6"))
        self._text_batch_split_delay_seconds = float(os.getenv("HERMES_MATRIX_TEXT_BATCH_SPLIT_DELAY_SECONDS", "2.0"))
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

    def _is_duplicate_event(self, event_id) -> bool:
        """Return True if this event was already processed. Tracks the ID otherwise."""
        if not event_id:
            return False
        if event_id in self._processed_events_set:
            return True
        if len(self._processed_events) == self._processed_events.maxlen:
            evicted = self._processed_events[0]
            self._processed_events_set.discard(evicted)
        self._processed_events.append(event_id)
        self._processed_events_set.add(event_id)
        return False

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
        # When a stable device_id is configured, pass it to the constructor
        # so matrix-nio binds to it from the start (important for E2EE
        # crypto-store persistence across restarts).
        ctor_device_id = self._device_id or None
        if self._encryption:
            if not _check_e2ee_deps():
                logger.error(
                    "Matrix: MATRIX_ENCRYPTION=true but E2EE dependencies are missing. %s. "
                    "Refusing to connect — encrypted rooms would silently fail.",
                    _E2EE_INSTALL_HINT,
                )
                return False
            try:
                client = nio.AsyncClient(
                    self._homeserver,
                    self._user_id or "",
                    device_id=ctor_device_id,
                    store_path=store_path,
                )
                logger.info(
                    "Matrix: E2EE enabled (store: %s%s)",
                    store_path,
                    f", device_id={self._device_id}" if self._device_id else "",
                )
            except Exception as exc:
                logger.error(
                    "Matrix: failed to create E2EE client: %s. %s",
                    exc, _E2EE_INSTALL_HINT,
                )
                return False
        else:
            client = nio.AsyncClient(
                self._homeserver,
                self._user_id or "",
                device_id=ctor_device_id,
            )

        self._client = client

        # Authenticate.
        if self._access_token:
            client.access_token = self._access_token

            # With access-token auth, always resolve whoami so we validate the
            # token and learn the device_id. The device_id matters for E2EE:
            # without it, matrix-nio can send plain messages but may fail to
            # decrypt inbound encrypted events or encrypt outbound room sends.
            resp = await client.whoami()
            if isinstance(resp, nio.WhoamiResponse):
                resolved_user_id = getattr(resp, "user_id", "") or self._user_id
                resolved_device_id = getattr(resp, "device_id", "")
                if resolved_user_id:
                    self._user_id = resolved_user_id

                # Prefer the user-configured device_id (MATRIX_DEVICE_ID) so
                # the bot reuses a stable identity across restarts.  Fall back
                # to whatever whoami returned.
                effective_device_id = self._device_id or resolved_device_id

                # restore_login() is the matrix-nio path that binds the access
                # token to a specific device and loads the crypto store.
                if effective_device_id and hasattr(client, "restore_login"):
                    client.restore_login(
                        self._user_id or resolved_user_id,
                        effective_device_id,
                        self._access_token,
                    )
                else:
                    if self._user_id:
                        client.user_id = self._user_id
                    if effective_device_id:
                        client.device_id = effective_device_id
                    client.access_token = self._access_token
                    if self._encryption:
                        logger.warning(
                            "Matrix: access-token login did not restore E2EE state; "
                            "encrypted rooms may fail until a device_id is available. "
                            "Set MATRIX_DEVICE_ID to a stable value."
                        )

                logger.info(
                    "Matrix: using access token for %s%s",
                    self._user_id or "(unknown user)",
                    f" (device {effective_device_id})" if effective_device_id else "",
                )
            else:
                logger.error(
                    "Matrix: whoami failed — check MATRIX_ACCESS_TOKEN and MATRIX_HOMESERVER"
                )
                await client.close()
                return False
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
        if self._encryption and getattr(client, "olm", None):
            try:
                if client.should_upload_keys:
                    await client.keys_upload()
                logger.info("Matrix: E2EE crypto initialized")
            except Exception as exc:
                logger.warning("Matrix: crypto init issue: %s", exc)

            # Import previously exported Megolm keys (survives restarts).
            if _KEY_EXPORT_FILE.exists():
                try:
                    await client.import_keys(
                        str(_KEY_EXPORT_FILE), _KEY_EXPORT_PASSPHRASE,
                    )
                    logger.info("Matrix: imported Megolm keys from backup")
                except Exception as exc:
                    logger.debug("Matrix: could not import keys: %s", exc)
        elif self._encryption:
            # E2EE was requested but the crypto store failed to load —
            # this means encrypted rooms will silently not work.  Hard-fail.
            logger.error(
                "Matrix: E2EE requested but crypto store is not loaded — "
                "cannot decrypt or encrypt messages. %s",
                _E2EE_INSTALL_HINT,
            )
            await client.close()
            return False

        # Register event callbacks.
        client.add_event_callback(self._on_room_message, nio.RoomMessageText)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageImage)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageAudio)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageVideo)
        client.add_event_callback(self._on_room_message_media, nio.RoomMessageFile)
        for encrypted_media_cls in (
            getattr(nio, "RoomEncryptedImage", None),
            getattr(nio, "RoomEncryptedAudio", None),
            getattr(nio, "RoomEncryptedVideo", None),
            getattr(nio, "RoomEncryptedFile", None),
        ):
            if encrypted_media_cls is not None:
                client.add_event_callback(self._on_room_message_media, encrypted_media_cls)
        client.add_event_callback(self._on_invite, nio.InviteMemberEvent)

        # Reaction events (m.reaction).
        if hasattr(nio, "ReactionEvent"):
            client.add_event_callback(self._on_reaction, nio.ReactionEvent)
        else:
            # Older matrix-nio versions: use UnknownEvent fallback.
            client.add_event_callback(self._on_unknown_event, nio.UnknownEvent)

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
            await self._run_e2ee_maintenance()
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

        # Export Megolm keys before closing so the next restart can decrypt
        # events that used sessions from this run.
        if self._client and self._encryption and getattr(self._client, "olm", None):
            try:
                _STORE_DIR.mkdir(parents=True, exist_ok=True)
                await self._client.export_keys(
                    str(_KEY_EXPORT_FILE), _KEY_EXPORT_PASSPHRASE,
                )
                logger.info("Matrix: exported Megolm keys for next restart")
            except Exception as exc:
                logger.debug("Matrix: could not export keys on disconnect: %s", exc)

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

            async def _room_send_once(*, ignore_unverified_devices: bool = False):
                return await asyncio.wait_for(
                    self._client.room_send(
                        chat_id,
                        "m.room.message",
                        msg_content,
                        ignore_unverified_devices=ignore_unverified_devices,
                    ),
                    timeout=45,
                )

            try:
                resp = await _room_send_once(ignore_unverified_devices=False)
            except Exception as exc:
                retryable = isinstance(exc, asyncio.TimeoutError)
                olm_unverified = getattr(nio, "OlmUnverifiedDeviceError", None)
                send_retry = getattr(nio, "SendRetryError", None)
                if isinstance(olm_unverified, type) and isinstance(exc, olm_unverified):
                    retryable = True
                if isinstance(send_retry, type) and isinstance(exc, send_retry):
                    retryable = True

                if not retryable:
                    logger.error("Matrix: failed to send to %s: %s", chat_id, exc)
                    return SendResult(success=False, error=str(exc))

                logger.warning(
                    "Matrix: initial encrypted send to %s failed (%s); "
                    "retrying after E2EE maintenance with ignored unverified devices",
                    chat_id,
                    exc,
                )
                await self._run_e2ee_maintenance()
                try:
                    resp = await _room_send_once(ignore_unverified_devices=True)
                except Exception as retry_exc:
                    logger.error("Matrix: failed to send to %s after retry: %s", chat_id, retry_exc)
                    return SendResult(success=False, error=str(retry_exc))

            if isinstance(resp, nio.RoomSendResponse):
                last_event_id = resp.event_id
                logger.info("Matrix: sent event %s to %s", last_event_id, chat_id)
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
        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            logger.warning("Matrix: blocked unsafe image URL (SSRF protection)")
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

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
        """Upload an audio file as a voice message (MSC3245 native voice)."""
        return await self._send_local_file(
            chat_id, audio_path, "m.audio", caption, reply_to, 
            metadata=metadata, is_voice=True
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
        is_voice: bool = False,
    ) -> SendResult:
        """Upload bytes to Matrix and send as a media message."""
        import nio

        # Upload to homeserver.
        # nio expects a DataProvider (callable) or file-like object, not raw bytes.
        # nio.upload() returns a tuple (UploadResponse|UploadError, Optional[Dict])
        resp, maybe_encryption_info = await self._client.upload(
            io.BytesIO(data),
            content_type=content_type,
            filename=filename,
            filesize=len(data),
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

        # Add MSC3245 voice flag for native voice messages.
        if is_voice:
            msg_content["org.matrix.msc3245.voice"] = {}

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
        is_voice: bool = False,
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

        return await self._upload_and_send(room_id, data, fname, ct, msgtype, caption, reply_to, metadata, is_voice)

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Continuously sync with the homeserver."""
        import nio

        while not self._closing:
            try:
                resp = await self._client.sync(timeout=30000)
                if isinstance(resp, nio.SyncError):
                    if self._closing:
                        return
                    err_msg = str(getattr(resp, "message", resp)).lower()
                    if "m_unknown_token" in err_msg or "m_forbidden" in err_msg or "401" in err_msg:
                        logger.error(
                            "Matrix: permanent auth error from sync: %s — stopping sync",
                            getattr(resp, "message", resp),
                        )
                        return
                    logger.warning(
                        "Matrix: sync returned %s: %s — retrying in 5s",
                        type(resp).__name__,
                        getattr(resp, "message", resp),
                    )
                    await asyncio.sleep(5)
                    continue

                await self._run_e2ee_maintenance()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Detect permanent auth/permission failures that will never
                # succeed on retry — stop syncing instead of looping forever.
                err_str = str(exc).lower()
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "forbidden" in err_str:
                    logger.error("Matrix: permanent auth error: %s — stopping sync", exc)
                    return
                logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)

    async def _run_e2ee_maintenance(self) -> None:
        """Run matrix-nio E2EE housekeeping between syncs.

        Hermes uses a custom sync loop instead of matrix-nio's sync_forever(),
        so we need to explicitly drive the key management work that sync_forever()
        normally handles for encrypted rooms.

        Also auto-trusts all devices (so senders share session keys with us)
        and retries decryption for any buffered MegolmEvents.
        """
        client = self._client
        if not client or not self._encryption or not getattr(client, "olm", None):
            return

        did_query_keys = client.should_query_keys

        tasks = [asyncio.create_task(client.send_to_device_messages())]

        if client.should_upload_keys:
            tasks.append(asyncio.create_task(client.keys_upload()))

        if did_query_keys:
            tasks.append(asyncio.create_task(client.keys_query()))

        if client.should_claim_keys:
            users = client.get_users_for_key_claiming()
            if users:
                tasks.append(asyncio.create_task(client.keys_claim(users)))

        for task in asyncio.as_completed(tasks):
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Matrix: E2EE maintenance task failed: %s", exc)

        # After key queries, auto-trust all devices so senders share keys with
        # us.  For a bot this is the right default — we want to decrypt
        # everything, not enforce manual verification.
        if did_query_keys:
            self._auto_trust_devices()

        # Retry any buffered undecrypted events now that new keys may have
        # arrived (from key requests, key queries, or to-device forwarding).
        if self._pending_megolm:
            await self._retry_pending_decryptions()

    def _auto_trust_devices(self) -> None:
        """Trust/verify all unverified devices we know about.

        When other clients see our device as verified, they proactively share
        Megolm session keys with us.  Without this, many clients will refuse
        to include an unverified device in key distributions.
        """
        client = self._client
        if not client:
            return

        device_store = getattr(client, "device_store", None)
        if not device_store:
            return

        own_device = getattr(client, "device_id", None)
        trusted_count = 0

        try:
            # DeviceStore.__iter__ yields OlmDevice objects directly.
            for device in device_store:
                if getattr(device, "device_id", None) == own_device:
                    continue
                if not getattr(device, "verified", False):
                    client.verify_device(device)
                    trusted_count += 1
        except Exception as exc:
            logger.debug("Matrix: auto-trust error: %s", exc)

        if trusted_count:
            logger.info("Matrix: auto-trusted %d new device(s)", trusted_count)

    async def _retry_pending_decryptions(self) -> None:
        """Retry decrypting buffered MegolmEvents after new keys arrive."""
        import nio

        client = self._client
        if not client or not self._pending_megolm:
            return

        now = time.time()
        still_pending: list = []

        for room, event, ts in self._pending_megolm:
            # Drop events that have aged past the TTL.
            if now - ts > _PENDING_EVENT_TTL:
                logger.debug(
                    "Matrix: dropping expired pending event %s (age %.0fs)",
                    getattr(event, "event_id", "?"), now - ts,
                )
                continue

            try:
                decrypted = client.decrypt_event(event)
            except Exception:
                # Still missing the key — keep in buffer.
                still_pending.append((room, event, ts))
                continue

            if isinstance(decrypted, nio.MegolmEvent):
                # decrypt_event returned the same undecryptable event.
                still_pending.append((room, event, ts))
                continue

            logger.info(
                "Matrix: decrypted buffered event %s (%s)",
                getattr(event, "event_id", "?"),
                type(decrypted).__name__,
            )

            # Route to the appropriate handler based on decrypted type.
            try:
                if isinstance(decrypted, nio.RoomMessageText):
                    await self._on_room_message(room, decrypted)
                elif isinstance(
                    decrypted,
                    (nio.RoomMessageImage, nio.RoomMessageAudio,
                     nio.RoomMessageVideo, nio.RoomMessageFile),
                ):
                    await self._on_room_message_media(room, decrypted)
                else:
                    logger.debug(
                        "Matrix: decrypted event %s has unhandled type %s",
                        getattr(event, "event_id", "?"),
                        type(decrypted).__name__,
                    )
            except Exception as exc:
                logger.warning(
                    "Matrix: error processing decrypted event %s: %s",
                    getattr(event, "event_id", "?"), exc,
                )

        self._pending_megolm = still_pending

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    async def _on_room_message(self, room: Any, event: Any) -> None:
        """Handle incoming text messages (and decrypted megolm events)."""
        import nio

        # Ignore own messages.
        if event.sender == self._user_id:
            return

        # Deduplicate by event ID (nio can fire the same event more than once).
        if self._is_duplicate_event(getattr(event, "event_id", None)):
            return

        # Startup grace: ignore old messages from initial sync.
        event_ts = getattr(event, "server_timestamp", 0) / 1000.0
        if event_ts and event_ts < self._startup_ts - _STARTUP_GRACE_SECONDS:
            return

        # Handle undecryptable MegolmEvents: request the missing session key
        # and buffer the event for retry once the key arrives.
        if isinstance(event, nio.MegolmEvent):
            logger.warning(
                "Matrix: could not decrypt event %s in %s — requesting key",
                event.event_id, room.room_id,
            )

            # Ask other devices in the room to forward the session key.
            try:
                resp = await self._client.request_room_key(event)
                if hasattr(resp, "event_id") or not isinstance(resp, Exception):
                    logger.debug(
                        "Matrix: room key request sent for session %s",
                        getattr(event, "session_id", "?"),
                    )
            except Exception as exc:
                logger.debug("Matrix: room key request failed: %s", exc)

            # Buffer for retry on next maintenance cycle.
            self._pending_megolm.append((room, event, time.time()))
            if len(self._pending_megolm) > _MAX_PENDING_EVENTS:
                self._pending_megolm = self._pending_megolm[-_MAX_PENDING_EVENTS:]
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

        # Require-mention gating.
        if not is_dm:
            free_rooms_raw = os.getenv("MATRIX_FREE_RESPONSE_ROOMS", "")
            free_rooms = {r.strip() for r in free_rooms_raw.split(",") if r.strip()}
            require_mention = os.getenv("MATRIX_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no")
            is_free_room = room.room_id in free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._bot_participated_threads)

            formatted_body = source_content.get("formatted_body")
            if require_mention and not is_free_room and not in_bot_thread:
                if not self._is_bot_mentioned(body, formatted_body):
                    return

        # Strip mention from body when present (including in DMs).
        if self._is_bot_mentioned(body, source_content.get("formatted_body")):
            body = self._strip_mention(body)

        # Auto-thread: create a thread for non-DM, non-threaded messages.
        if not is_dm and not thread_id:
            auto_thread = os.getenv("MATRIX_AUTO_THREAD", "true").lower() in ("true", "1", "yes")
            if auto_thread:
                thread_id = event.event_id
                self._track_thread(thread_id)

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
        if body.startswith(("!", "/")):
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
            reply_to_message_id=reply_to,
        )

        if thread_id:
            self._track_thread(thread_id)

        # Acknowledge receipt so the room shows as read (fire-and-forget).
        self._background_read_receipt(room.room_id, event.event_id)

        # Only batch plain text messages — commands dispatch immediately.
        if msg_type == MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    # ------------------------------------------------------------------
    # Text message aggregation (handles Matrix client-side splits)
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        from gateway.session import build_session_key
        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text event and reset the flush timer.

        When a Matrix client splits a long message, the chunks arrive within
        a few hundred milliseconds.  This merges them into a single event
        before dispatching.
        """
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        chunk_len = len(event.text or "")
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = f"{existing.text}\n{event.text}" if existing.text else event.text
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            # Merge any media that might be attached
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)

        # Cancel any pending flush and restart the timer
        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text.

        Uses a longer delay when the latest chunk is near Matrix's ~4000-char
        split point, since a continuation chunk is almost certain.
        """
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info(
                "[Matrix] Flushing text batch %s (%d chars)",
                key, len(event.text or ""),
            )
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    async def _on_room_message_media(self, room: Any, event: Any) -> None:
        """Handle incoming media messages (images, audio, video, files)."""
        import nio

        # Ignore own messages.
        if event.sender == self._user_id:
            return

        # Deduplicate by event ID.
        if self._is_duplicate_event(getattr(event, "event_id", None)):
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
        # Use the MIME type from the event's content info when available,
        # falling back to category-level MIME types for downstream matching
        # (gateway/run.py checks startswith("image/"), startswith("audio/"), etc.)
        source_content = getattr(event, "source", {}).get("content", {})
        if not isinstance(source_content, dict):
            source_content = {}
        event_content = getattr(event, "content", {})
        if not isinstance(event_content, dict):
            event_content = {}
        content_info = event_content.get("info") if isinstance(event_content, dict) else {}
        if not isinstance(content_info, dict) or not content_info:
            content_info = source_content.get("info", {}) if isinstance(source_content, dict) else {}
        event_mimetype = (
            (content_info.get("mimetype") if isinstance(content_info, dict) else None)
            or getattr(event, "mimetype", "")
            or ""
        )
        # For encrypted media, the URL may be in file.url instead of event.url.
        file_content = source_content.get("file", {}) if isinstance(source_content, dict) else {}
        if not url and isinstance(file_content, dict):
            url = file_content.get("url", "") or ""
            if url and url.startswith("mxc://"):
                http_url = self._mxc_to_http(url)

        media_type = "application/octet-stream"
        msg_type = MessageType.DOCUMENT

        # Safely resolve encrypted media classes — they may not exist on older
        # nio versions, and in test environments nio may be mocked (MagicMock
        # auto-attributes are not valid types for isinstance).
        def _safe_isinstance(obj, cls_name):
            cls = getattr(nio, cls_name, None)
            if cls is None or not isinstance(cls, type):
                return False
            return isinstance(obj, cls)

        is_encrypted_image = _safe_isinstance(event, "RoomEncryptedImage")
        is_encrypted_audio = _safe_isinstance(event, "RoomEncryptedAudio")
        is_encrypted_video = _safe_isinstance(event, "RoomEncryptedVideo")
        is_encrypted_file = _safe_isinstance(event, "RoomEncryptedFile")
        is_encrypted_media = any((is_encrypted_image, is_encrypted_audio, is_encrypted_video, is_encrypted_file))
        is_voice_message = False

        if isinstance(event, nio.RoomMessageImage) or is_encrypted_image:
            msg_type = MessageType.PHOTO
            media_type = event_mimetype or "image/png"
        elif isinstance(event, nio.RoomMessageAudio) or is_encrypted_audio:
            if source_content.get("org.matrix.msc3245.voice") is not None:
                is_voice_message = True
                msg_type = MessageType.VOICE
            else:
                msg_type = MessageType.AUDIO
            media_type = event_mimetype or "audio/ogg"
        elif isinstance(event, nio.RoomMessageVideo) or is_encrypted_video:
            msg_type = MessageType.VIDEO
            media_type = event_mimetype or "video/mp4"
        elif event_mimetype:
            media_type = event_mimetype

        # Cache media locally when downstream tools need a real file path:
        # - photos (vision tools can't access MXC URLs)
        # - voice messages (transcription tools need local files)
        # - any encrypted media (HTTP fallback would point at ciphertext)
        cached_path = None
        should_cache_locally = (
            msg_type == MessageType.PHOTO or is_voice_message or is_encrypted_media
        )
        if should_cache_locally and url:
            try:
                if is_voice_message:
                    download_resp = await self._client.download(mxc=url)
                else:
                    download_resp = await self._client.download(url)
                file_bytes = getattr(download_resp, "body", None)
                if file_bytes is not None:
                    if is_encrypted_media:
                        from nio.crypto.attachments import decrypt_attachment

                        hashes_value = getattr(event, "hashes", None)
                        if hashes_value is None and isinstance(file_content, dict):
                            hashes_value = file_content.get("hashes")
                        hash_value = hashes_value.get("sha256") if isinstance(hashes_value, dict) else None

                        key_value = getattr(event, "key", None)
                        if key_value is None and isinstance(file_content, dict):
                            key_value = file_content.get("key")
                        if isinstance(key_value, dict):
                            key_value = key_value.get("k")

                        iv_value = getattr(event, "iv", None)
                        if iv_value is None and isinstance(file_content, dict):
                            iv_value = file_content.get("iv")

                        if key_value and hash_value and iv_value:
                            file_bytes = decrypt_attachment(file_bytes, key_value, hash_value, iv_value)
                        else:
                            logger.warning(
                                "[Matrix] Encrypted media event missing decryption metadata for %s",
                                event.event_id,
                            )
                            file_bytes = None

                    if file_bytes is not None:
                        from gateway.platforms.base import (
                            cache_audio_from_bytes,
                            cache_document_from_bytes,
                            cache_image_from_bytes,
                        )

                        if msg_type == MessageType.PHOTO:
                            ext_map = {
                                "image/jpeg": ".jpg",
                                "image/png": ".png",
                                "image/gif": ".gif",
                                "image/webp": ".webp",
                            }
                            ext = ext_map.get(media_type, ".jpg")
                            cached_path = cache_image_from_bytes(file_bytes, ext=ext)
                            logger.info("[Matrix] Cached user image at %s", cached_path)
                        elif msg_type in (MessageType.AUDIO, MessageType.VOICE):
                            ext = Path(body or ("voice.ogg" if is_voice_message else "audio.ogg")).suffix or ".ogg"
                            cached_path = cache_audio_from_bytes(file_bytes, ext=ext)
                        else:
                            filename = body or (
                                "video.mp4" if msg_type == MessageType.VIDEO else "document"
                            )
                            cached_path = cache_document_from_bytes(file_bytes, filename)
            except Exception as e:
                logger.warning("[Matrix] Failed to cache media: %s", e)

        is_dm = self._dm_rooms.get(room.room_id, False)
        if not is_dm and room.member_count == 2:
            is_dm = True
        chat_type = "dm" if is_dm else "group"

        # Thread/reply detection.
        relates_to = source_content.get("m.relates_to", {})
        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")

        # Require-mention gating (media messages).
        if not is_dm:
            free_rooms_raw = os.getenv("MATRIX_FREE_RESPONSE_ROOMS", "")
            free_rooms = {r.strip() for r in free_rooms_raw.split(",") if r.strip()}
            require_mention = os.getenv("MATRIX_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no")
            is_free_room = room.room_id in free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._bot_participated_threads)

            if require_mention and not is_free_room and not in_bot_thread:
                formatted_body = source_content.get("formatted_body")
                if not self._is_bot_mentioned(body, formatted_body):
                    return

        # Strip mention from body when present (including in DMs).
        if self._is_bot_mentioned(body, source_content.get("formatted_body")):
            body = self._strip_mention(body)

        # Auto-thread: create a thread for non-DM, non-threaded messages.
        if not is_dm and not thread_id:
            auto_thread = os.getenv("MATRIX_AUTO_THREAD", "true").lower() in ("true", "1", "yes")
            if auto_thread:
                thread_id = event.event_id
                self._track_thread(thread_id)

        source = self.build_source(
            chat_id=room.room_id,
            chat_type=chat_type,
            user_id=event.sender,
            user_name=self._get_display_name(room, event.sender),
            thread_id=thread_id,
        )

        allow_http_fallback = bool(http_url) and not is_encrypted_media
        media_urls = [cached_path] if cached_path else ([http_url] if allow_http_fallback else None)
        media_types = [media_type] if media_urls else None

        msg_event = MessageEvent(
            text=body,
            message_type=msg_type,
            source=source,
            raw_message=getattr(event, "source", {}),
            message_id=event.event_id,
            media_urls=media_urls,
            media_types=media_types,
        )

        if thread_id:
            self._track_thread(thread_id)

        # Acknowledge receipt so the room shows as read (fire-and-forget).
        self._background_read_receipt(room.room_id, event.event_id)

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
    # Reactions (send, receive, processing lifecycle)
    # ------------------------------------------------------------------

    async def _send_reaction(
        self, room_id: str, event_id: str, emoji: str,
    ) -> Optional[str]:
        """Send an emoji reaction to a message in a room.
        Returns the reaction event_id on success, None on failure.
        """
        import nio

        if not self._client:
            return None
        content = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": event_id,
                "key": emoji,
            }
        }
        try:
            resp = await self._client.room_send(
                room_id, "m.reaction", content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, nio.RoomSendResponse):
                logger.debug("Matrix: sent reaction %s to %s", emoji, event_id)
                return resp.event_id
            logger.debug("Matrix: reaction send failed: %s", resp)
            return None
        except Exception as exc:
            logger.debug("Matrix: reaction send error: %s", exc)
            return None

    async def _redact_reaction(
        self, room_id: str, reaction_event_id: str, reason: str = "",
    ) -> bool:
        """Remove a reaction by redacting its event."""
        return await self.redact_message(room_id, reaction_event_id, reason)

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add eyes reaction when the agent starts processing a message."""
        if not self._reactions_enabled:
            return
        msg_id = event.message_id
        room_id = event.source.chat_id
        if msg_id and room_id:
            reaction_event_id = await self._send_reaction(room_id, msg_id, "\U0001f440")
            if reaction_event_id:
                self._pending_reactions[(room_id, msg_id)] = reaction_event_id

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome,
    ) -> None:
        """Replace eyes with checkmark (success) or cross (failure)."""
        if not self._reactions_enabled:
            return
        msg_id = event.message_id
        room_id = event.source.chat_id
        if not msg_id or not room_id:
            return
        if outcome == ProcessingOutcome.CANCELLED:
            return
        # Remove the eyes reaction first, if we tracked its event_id.
        reaction_key = (room_id, msg_id)
        if reaction_key in self._pending_reactions:
            eyes_event_id = self._pending_reactions.pop(reaction_key)
            if not await self._redact_reaction(room_id, eyes_event_id):
                logger.debug("Matrix: failed to redact eyes reaction %s", eyes_event_id)
        await self._send_reaction(
            room_id,
            msg_id,
            "\u2705" if outcome == ProcessingOutcome.SUCCESS else "\u274c",
        )

    async def _on_reaction(self, room: Any, event: Any) -> None:
        """Handle incoming reaction events."""
        if event.sender == self._user_id:
            return
        if self._is_duplicate_event(getattr(event, "event_id", None)):
            return
        # Log for now; future: trigger agent actions based on emoji.
        reacts_to = getattr(event, "reacts_to", "")
        key = getattr(event, "key", "")
        logger.info(
            "Matrix: reaction %s from %s on %s in %s",
            key, event.sender, reacts_to, room.room_id,
        )

    async def _on_unknown_event(self, room: Any, event: Any) -> None:
        """Fallback handler for events not natively parsed by matrix-nio.

        Catches m.reaction on older nio versions that lack ReactionEvent.
        """
        source = getattr(event, "source", {})
        if source.get("type") != "m.reaction":
            return
        content = source.get("content", {})
        relates_to = content.get("m.relates_to", {})
        if relates_to.get("rel_type") != "m.annotation":
            return
        if source.get("sender") == self._user_id:
            return
        logger.info(
            "Matrix: reaction %s from %s on %s in %s",
            relates_to.get("key", "?"),
            source.get("sender", "?"),
            relates_to.get("event_id", "?"),
            room.room_id,
        )

    # ------------------------------------------------------------------
    # Read receipts
    # ------------------------------------------------------------------

    def _background_read_receipt(self, room_id: str, event_id: str) -> None:
        """Fire-and-forget read receipt with error logging."""
        async def _send() -> None:
            try:
                await self.send_read_receipt(room_id, event_id)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("Matrix: background read receipt failed: %s", exc)
        asyncio.ensure_future(_send())

    async def send_read_receipt(self, room_id: str, event_id: str) -> bool:
        """Send a read receipt (m.read) for an event.

        Also sets the fully-read marker so the room is marked as read
        in all clients.
        """
        if not self._client:
            return False
        try:
            if hasattr(self._client, "room_read_markers"):
                await self._client.room_read_markers(
                    room_id,
                    fully_read_event=event_id,
                    read_event=event_id,
                )
            else:
                # Fallback for older matrix-nio.
                await self._client.room_send(
                    room_id, "m.receipt", {"event_id": event_id},
                )
            logger.debug("Matrix: sent read receipt for %s in %s", event_id, room_id)
            return True
        except Exception as exc:
            logger.debug("Matrix: read receipt failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Message redaction
    # ------------------------------------------------------------------

    async def redact_message(
        self, room_id: str, event_id: str, reason: str = "",
    ) -> bool:
        """Redact (delete) a message or event from a room."""
        import nio

        if not self._client:
            return False
        try:
            resp = await self._client.room_redact(
                room_id, event_id, reason=reason,
            )
            if isinstance(resp, nio.RoomRedactResponse):
                logger.info("Matrix: redacted %s in %s", event_id, room_id)
                return True
            logger.warning("Matrix: redact failed: %s", resp)
            return False
        except Exception as exc:
            logger.warning("Matrix: redact error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Room history
    # ------------------------------------------------------------------

    async def fetch_room_history(
        self,
        room_id: str,
        limit: int = 50,
        start: str = "",
    ) -> list:
        """Fetch recent messages from a room.

        Returns a list of dicts with keys: event_id, sender, body,
        timestamp, type.  Uses the ``room_messages()`` API.
        """
        import nio

        if not self._client:
            return []
        try:
            resp = await self._client.room_messages(
                room_id,
                start=start or "",
                limit=limit,
                direction=nio.Api.MessageDirection.back
                if hasattr(nio.Api, "MessageDirection")
                else "b",
            )
        except Exception as exc:
            logger.warning("Matrix: room_messages failed for %s: %s", room_id, exc)
            return []

        if not isinstance(resp, nio.RoomMessagesResponse):
            logger.warning("Matrix: room_messages returned %s", type(resp).__name__)
            return []

        messages = []
        for event in reversed(resp.chunk):
            body = getattr(event, "body", "") or ""
            messages.append({
                "event_id": getattr(event, "event_id", ""),
                "sender": getattr(event, "sender", ""),
                "body": body,
                "timestamp": getattr(event, "server_timestamp", 0),
                "type": type(event).__name__,
            })
        return messages

    # ------------------------------------------------------------------
    # Room creation & management
    # ------------------------------------------------------------------

    async def create_room(
        self,
        name: str = "",
        topic: str = "",
        invite: Optional[list] = None,
        is_direct: bool = False,
        preset: str = "private_chat",
    ) -> Optional[str]:
        """Create a new Matrix room.

        Args:
            name: Human-readable room name.
            topic: Room topic.
            invite: List of user IDs to invite.
            is_direct: Mark as a DM room.
            preset: One of private_chat, public_chat, trusted_private_chat.

        Returns the room_id on success, None on failure.
        """
        import nio

        if not self._client:
            return None
        try:
            resp = await self._client.room_create(
                name=name or None,
                topic=topic or None,
                invite=invite or [],
                is_direct=is_direct,
                preset=getattr(
                    nio.Api.RoomPreset if hasattr(nio.Api, "RoomPreset") else type("", (), {}),
                    preset, None,
                ) or preset,
            )
            if isinstance(resp, nio.RoomCreateResponse):
                room_id = resp.room_id
                self._joined_rooms.add(room_id)
                logger.info("Matrix: created room %s (%s)", room_id, name or "unnamed")
                return room_id
            logger.warning("Matrix: room_create failed: %s", resp)
            return None
        except Exception as exc:
            logger.warning("Matrix: room_create error: %s", exc)
            return None

    async def invite_user(self, room_id: str, user_id: str) -> bool:
        """Invite a user to a room."""
        import nio

        if not self._client:
            return False
        try:
            resp = await self._client.room_invite(room_id, user_id)
            if isinstance(resp, nio.RoomInviteResponse):
                logger.info("Matrix: invited %s to %s", user_id, room_id)
                return True
            logger.warning("Matrix: invite failed: %s", resp)
            return False
        except Exception as exc:
            logger.warning("Matrix: invite error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Presence
    # ------------------------------------------------------------------

    _VALID_PRESENCE_STATES = frozenset(("online", "offline", "unavailable"))

    async def set_presence(self, state: str = "online", status_msg: str = "") -> bool:
        """Set the bot's presence status."""
        if not self._client:
            return False
        if state not in self._VALID_PRESENCE_STATES:
            logger.warning("Matrix: invalid presence state %r", state)
            return False
        try:
            if hasattr(self._client, "set_presence"):
                await self._client.set_presence(state, status_msg=status_msg or None)
                logger.debug("Matrix: presence set to %s", state)
                return True
        except Exception as exc:
            logger.debug("Matrix: set_presence failed: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Emote & notice message types
    # ------------------------------------------------------------------

    async def send_emote(
        self, chat_id: str, text: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an emote message (/me style action)."""
        import nio

        if not self._client or not text:
            return SendResult(success=False, error="No client or empty text")

        msg_content: Dict[str, Any] = {
            "msgtype": "m.emote",
            "body": text,
        }
        html = self._markdown_to_html(text)
        if html and html != text:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = html

        try:
            resp = await self._client.room_send(
                chat_id, "m.room.message", msg_content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, nio.RoomSendResponse):
                return SendResult(success=True, message_id=resp.event_id)
            return SendResult(success=False, error=str(resp))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_notice(
        self, chat_id: str, text: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a notice message (bot-appropriate, non-alerting)."""
        import nio

        if not self._client or not text:
            return SendResult(success=False, error="No client or empty text")

        msg_content: Dict[str, Any] = {
            "msgtype": "m.notice",
            "body": text,
        }
        html = self._markdown_to_html(text)
        if html and html != text:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = html

        try:
            resp = await self._client.room_send(
                chat_id, "m.room.message", msg_content,
                ignore_unverified_devices=True,
            )
            if isinstance(resp, nio.RoomSendResponse):
                return SendResult(success=True, message_id=resp.event_id)
            return SendResult(success=False, error=str(resp))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

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

    # ------------------------------------------------------------------
    # Thread participation tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _thread_state_path() -> Path:
        """Path to the persisted thread participation set."""
        from hermes_cli.config import get_hermes_home
        return get_hermes_home() / "matrix_threads.json"

    @classmethod
    def _load_participated_threads(cls) -> set:
        """Load persisted thread IDs from disk."""
        path = cls._thread_state_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return set(data)
        except Exception as e:
            logger.debug("Could not load matrix thread state: %s", e)
        return set()

    def _save_participated_threads(self) -> None:
        """Persist the current thread set to disk (best-effort)."""
        path = self._thread_state_path()
        try:
            thread_list = list(self._bot_participated_threads)
            if len(thread_list) > self._MAX_TRACKED_THREADS:
                thread_list = thread_list[-self._MAX_TRACKED_THREADS:]
                self._bot_participated_threads = set(thread_list)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(thread_list), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not save matrix thread state: %s", e)

    def _track_thread(self, thread_id: str) -> None:
        """Add a thread to the participation set and persist."""
        if thread_id not in self._bot_participated_threads:
            self._bot_participated_threads.add(thread_id)
            self._save_participated_threads()

    # ------------------------------------------------------------------
    # Mention detection helpers
    # ------------------------------------------------------------------

    def _is_bot_mentioned(self, body: str, formatted_body: Optional[str] = None) -> bool:
        """Return True if the bot is mentioned in the message."""
        if not body and not formatted_body:
            return False
        # Check for full @user:server in body
        if self._user_id and self._user_id in body:
            return True
        # Check for localpart with word boundaries (case-insensitive)
        if self._user_id and ":" in self._user_id:
            localpart = self._user_id.split(":")[0].lstrip("@")
            if localpart and re.search(r'\b' + re.escape(localpart) + r'\b', body, re.IGNORECASE):
                return True
        # Check formatted_body for Matrix pill
        if formatted_body and self._user_id:
            if f"matrix.to/#/{self._user_id}" in formatted_body:
                return True
        return False

    def _strip_mention(self, body: str) -> str:
        """Remove bot mention from message body."""
        # Remove full @user:server
        if self._user_id:
            body = body.replace(self._user_id, "")
        # If still contains localpart mention, remove it
        if self._user_id and ":" in self._user_id:
            localpart = self._user_id.split(":")[0].lstrip("@")
            if localpart:
                body = re.sub(r'\b' + re.escape(localpart) + r'\b', '', body, flags=re.IGNORECASE)
        return body.strip()

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
        """Convert Markdown to Matrix-compatible HTML (org.matrix.custom.html).

        Uses the ``markdown`` library when available (installed with the
        ``matrix`` extra).  Falls back to a comprehensive regex converter
        that handles fenced code blocks, inline code, headers, bold,
        italic, strikethrough, links, blockquotes, lists, and horizontal
        rules — everything the Matrix HTML spec allows.
        """
        try:
            import markdown as _md

            md = _md.Markdown(
                extensions=["fenced_code", "tables", "nl2br", "sane_lists"],
            )
            # Remove the raw HTML preprocessor so <script> etc. in the
            # source are escaped rather than passed through.
            if "html_block" in md.preprocessors:
                md.preprocessors.deregister("html_block")

            html = md.convert(text)
            md.reset()

            # Strip wrapping <p> tags for single-paragraph messages so
            # clients don't add extra spacing around short replies.
            if html.count("<p>") == 1:
                html = html.replace("<p>", "").replace("</p>", "")
            return html
        except ImportError:
            pass

        return self._markdown_to_html_fallback(text)

    # ------------------------------------------------------------------
    # Regex-based Markdown -> HTML (no extra dependencies)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_link_url(url: str) -> str:
        """Sanitize a URL for use in an href attribute.

        Rejects dangerous URI schemes (javascript:, data:, vbscript:) and
        escapes double-quotes to prevent attribute breakout.
        """
        stripped = url.strip()
        scheme = stripped.split(":", 1)[0].lower().strip() if ":" in stripped else ""
        if scheme in ("javascript", "data", "vbscript"):
            return ""
        # Escape double quotes to prevent href attribute breakout.
        return stripped.replace('"', "&quot;")

    @staticmethod
    def _markdown_to_html_fallback(text: str) -> str:
        """Comprehensive regex Markdown-to-HTML for Matrix.

        Handles fenced code blocks, inline code, headers, bold, italic,
        strikethrough, links, blockquotes, ordered/unordered lists, and
        horizontal rules.  Code regions are extracted first to prevent
        inner transformations from mangling them.

        Security: all non-code text is HTML-escaped before markdown
        transforms to prevent HTML injection via crafted input.  Link
        URLs are sanitized against dangerous URI schemes.
        """
        placeholders: list = []

        def _protect_html(html_fragment: str) -> str:
            idx = len(placeholders)
            placeholders.append(html_fragment)
            return f"\x00PROTECTED{idx}\x00"

        # Fenced code blocks: ```lang\n...\n```
        result = re.sub(
            r"```(\w*)\n(.*?)```",
            lambda m: _protect_html(
                f'<pre><code class="language-{_html_escape(m.group(1))}">'
                f"{_html_escape(m.group(2))}</code></pre>"
                if m.group(1)
                else f"<pre><code>{_html_escape(m.group(2))}</code></pre>"
            ),
            text,
            flags=re.DOTALL,
        )

        # Inline code: `code`
        result = re.sub(
            r"`([^`\n]+)`",
            lambda m: _protect_html(
                f"<code>{_html_escape(m.group(1))}</code>"
            ),
            result,
        )

        # Extract and protect markdown links before escaping.
        result = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: _protect_html(
                '<a href="{}">{}</a>'.format(
                    MatrixAdapter._sanitize_link_url(m.group(2)),
                    _html_escape(m.group(1)),
                )
            ),
            result,
        )

        # HTML-escape remaining text (neutralises <script>, <img onerror=...>).
        parts = re.split(r"(\x00PROTECTED\d+\x00)", result)
        for idx, part in enumerate(parts):
            if not part.startswith("\x00PROTECTED"):
                parts[idx] = _html_escape(part)
        result = "".join(parts)

        # Block-level transforms (line-oriented).
        lines = result.split("\n")
        out_lines: list = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Horizontal rule
            if re.match(r"^[\s]*([-*_])\s*\1\s*\1[\s\-*_]*$", line):
                out_lines.append("<hr>")
                i += 1
                continue

            # Headers
            hdr = re.match(r"^(#{1,6})\s+(.+)$", line)
            if hdr:
                level = len(hdr.group(1))
                out_lines.append(f"<h{level}>{hdr.group(2).strip()}</h{level}>")
                i += 1
                continue

            # Blockquote (> may be escaped to &gt; by html.escape)
            if line.startswith("&gt; ") or line == "&gt;" or line.startswith("> ") or line == ">":
                bq_lines = []
                while i < len(lines) and (
                    lines[i].startswith("&gt; ") or lines[i] == "&gt;"
                    or lines[i].startswith("> ") or lines[i] == ">"
                ):
                    ln = lines[i]
                    if ln.startswith("&gt; "):
                        bq_lines.append(ln[5:])
                    elif ln.startswith("> "):
                        bq_lines.append(ln[2:])
                    else:
                        bq_lines.append("")
                    i += 1
                out_lines.append(f"<blockquote>{'<br>'.join(bq_lines)}</blockquote>")
                continue

            # Unordered list
            ul_match = re.match(r"^[\s]*[-*+]\s+(.+)$", line)
            if ul_match:
                items = []
                while i < len(lines) and re.match(r"^[\s]*[-*+]\s+(.+)$", lines[i]):
                    items.append(re.match(r"^[\s]*[-*+]\s+(.+)$", lines[i]).group(1))
                    i += 1
                li = "".join(f"<li>{item}</li>" for item in items)
                out_lines.append(f"<ul>{li}</ul>")
                continue

            # Ordered list
            ol_match = re.match(r"^[\s]*\d+[.)]\s+(.+)$", line)
            if ol_match:
                items = []
                while i < len(lines) and re.match(r"^[\s]*\d+[.)]\s+(.+)$", lines[i]):
                    items.append(re.match(r"^[\s]*\d+[.)]\s+(.+)$", lines[i]).group(1))
                    i += 1
                li = "".join(f"<li>{item}</li>" for item in items)
                out_lines.append(f"<ol>{li}</ol>")
                continue

            out_lines.append(line)
            i += 1

        result = "\n".join(out_lines)

        # Inline transforms.
        result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result, flags=re.DOTALL)
        result = re.sub(r"__(.+?)__", r"<strong>\1</strong>", result, flags=re.DOTALL)
        result = re.sub(r"\*(.+?)\*", r"<em>\1</em>", result, flags=re.DOTALL)
        result = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", result, flags=re.DOTALL)
        result = re.sub(r"~~(.+?)~~", r"<del>\1</del>", result, flags=re.DOTALL)
        result = re.sub(r"\n", "<br>\n", result)
        # Clean up excessive <br> around block elements.
        result = re.sub(r"<br>\n(</?(?:pre|blockquote|h[1-6]|ul|ol|li|hr))", r"\n\1", result)
        result = re.sub(r"(</(?:pre|blockquote|h[1-6]|ul|ol|li)>)<br>", r"\1", result)

        # Restore protected regions.
        for idx, original in enumerate(placeholders):
            result = result.replace(f"\x00PROTECTED{idx}\x00", original)

        return result
