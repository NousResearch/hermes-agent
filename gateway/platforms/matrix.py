"""Matrix gateway adapter.

Connects to any Matrix homeserver (self-hosted or matrix.org) via the
mautrix Python SDK.  Supports optional end-to-end encryption (E2EE)
when installed with ``pip install "mautrix[encryption]"``.

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
    MATRIX_DM_MENTION_THREADS   Create a thread when bot is @mentioned in a DM (default: false)
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
from typing import Any, Dict, Optional, Set

from html import escape as _html_escape

from mautrix.types import (
    ContentURI,
    EventID,
    EventType,
    PaginationDirection,
    PresenceState,
    RoomCreatePreset,
    RoomID,
    SyncToken,
    TrustState,
    UserID,
)

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
_CRYPTO_PICKLE_PATH = _STORE_DIR / "crypto_store.pickle"

# Grace period: ignore messages older than this many seconds before startup.
_STARTUP_GRACE_SECONDS = 5

# Pending undecrypted events: cap and TTL for retry buffer.
_MAX_PENDING_EVENTS = 100
_PENDING_EVENT_TTL = 300  # seconds — stop retrying after 5 min


_E2EE_INSTALL_HINT = (
    "Install with: pip install 'mautrix[encryption]'  "
    "(requires libolm C library)"
)


def _check_e2ee_deps() -> bool:
    """Return True if mautrix E2EE dependencies (python-olm) are available."""
    try:
        from mautrix.crypto import OlmMachine  # noqa: F401
        return True
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
        import mautrix  # noqa: F401
    except ImportError:
        logger.warning(
            "Matrix: mautrix not installed. "
            "Run: pip install 'mautrix[encryption]'"
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

        self._client: Any = None  # mautrix.client.Client
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
        # Each entry: (room_id, event, timestamp)
        self._pending_megolm: list = []

        # Thread participation tracking (for require_mention bypass)
        self._bot_participated_threads: set = self._load_participated_threads()
        self._MAX_TRACKED_THREADS = 500

        # Mention/thread gating — parsed once from env vars.
        self._require_mention: bool = os.getenv("MATRIX_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no")
        free_rooms_raw = os.getenv("MATRIX_FREE_RESPONSE_ROOMS", "")
        self._free_rooms: Set[str] = {r.strip() for r in free_rooms_raw.split(",") if r.strip()}
        self._auto_thread: bool = os.getenv("MATRIX_AUTO_THREAD", "true").lower() in ("true", "1", "yes")
        self._dm_mention_threads: bool = os.getenv("MATRIX_DM_MENTION_THREADS", "false").lower() in ("true", "1", "yes")

        # Reactions: configurable via MATRIX_REACTIONS (default: true).
        self._reactions_enabled: bool = os.getenv(
            "MATRIX_REACTIONS", "true"
        ).lower() not in ("false", "0", "no")
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
        from mautrix.api import HTTPAPI
        from mautrix.client import Client
        from mautrix.client.state_store import MemoryStateStore, MemorySyncStore

        if not self._homeserver:
            logger.error("Matrix: homeserver URL not configured")
            return False

        # Ensure store dir exists for E2EE key persistence.
        _STORE_DIR.mkdir(parents=True, exist_ok=True)

        # Create the HTTP API layer.
        api = HTTPAPI(
            base_url=self._homeserver,
            token=self._access_token or "",
        )

        # Create the client.
        state_store = MemoryStateStore()
        sync_store = MemorySyncStore()
        client = Client(
            mxid=UserID(self._user_id) if self._user_id else UserID(""),
            device_id=self._device_id or None,
            api=api,
            state_store=state_store,
            sync_store=sync_store,
        )

        self._client = client

        # Authenticate.
        if self._access_token:
            api.token = self._access_token

            # Validate the token and learn user_id / device_id.
            try:
                resp = await client.whoami()
                resolved_user_id = getattr(resp, "user_id", "") or self._user_id
                resolved_device_id = getattr(resp, "device_id", "")
                if resolved_user_id:
                    self._user_id = str(resolved_user_id)
                    client.mxid = UserID(self._user_id)

                # Prefer user-configured device_id for stable E2EE identity.
                effective_device_id = self._device_id or resolved_device_id
                if effective_device_id:
                    client.device_id = effective_device_id

                logger.info(
                    "Matrix: using access token for %s%s",
                    self._user_id or "(unknown user)",
                    f" (device {effective_device_id})" if effective_device_id else "",
                )
            except Exception as exc:
                logger.error(
                    "Matrix: whoami failed — check MATRIX_ACCESS_TOKEN and MATRIX_HOMESERVER: %s",
                    exc,
                )
                await api.session.close()
                return False
        elif self._password and self._user_id:
            try:
                resp = await client.login(
                    identifier=self._user_id,
                    password=self._password,
                    device_name="Hermes Agent",
                    device_id=self._device_id or None,
                )
                if resp and hasattr(resp, "device_id"):
                    client.device_id = resp.device_id
                logger.info("Matrix: logged in as %s", self._user_id)
            except Exception as exc:
                logger.error("Matrix: login failed — %s", exc)
                await api.session.close()
                return False
        else:
            logger.error("Matrix: need MATRIX_ACCESS_TOKEN or MATRIX_USER_ID + MATRIX_PASSWORD")
            await api.session.close()
            return False

        # Set up E2EE if requested.
        if self._encryption:
            if not _check_e2ee_deps():
                logger.error(
                    "Matrix: MATRIX_ENCRYPTION=true but E2EE dependencies are missing. %s. "
                    "Refusing to connect — encrypted rooms would silently fail.",
                    _E2EE_INSTALL_HINT,
                )
                await api.session.close()
                return False
            try:
                from mautrix.crypto import OlmMachine
                from mautrix.crypto.store import MemoryCryptoStore

                crypto_store = MemoryCryptoStore()

                # Restore persisted crypto state from a previous run.
                # Uses HMAC to verify integrity before unpickling.
                pickle_path = _CRYPTO_PICKLE_PATH
                if pickle_path.exists():
                    try:
                        import hashlib, hmac, pickle
                        raw = pickle_path.read_bytes()
                        # Format: 32-byte HMAC-SHA256 signature + pickle data.
                        if len(raw) > 32:
                            sig, payload = raw[:32], raw[32:]
                            # Key is derived from the device_id + user_id (stable per install).
                            hmac_key = f"{self._user_id}:{self._device_id}".encode()
                            expected = hmac.new(hmac_key, payload, hashlib.sha256).digest()
                            if hmac.compare_digest(sig, expected):
                                saved = pickle.loads(payload)  # noqa: S301
                                if isinstance(saved, MemoryCryptoStore):
                                    crypto_store = saved
                                    logger.info("Matrix: restored E2EE crypto store from %s", pickle_path)
                            else:
                                logger.warning("Matrix: crypto store HMAC mismatch — ignoring stale/tampered file")
                    except Exception as exc:
                        logger.warning("Matrix: could not restore crypto store: %s", exc)

                olm = OlmMachine(client, crypto_store, state_store)

                # Set trust policy: accept unverified devices so senders
                # share Megolm session keys with us automatically.
                olm.share_keys_min_trust = TrustState.UNVERIFIED
                olm.send_keys_min_trust = TrustState.UNVERIFIED

                await olm.load()
                client.crypto = olm
                logger.info(
                    "Matrix: E2EE enabled (store: %s%s)",
                    str(_STORE_DIR),
                    f", device_id={client.device_id}" if client.device_id else "",
                )
            except Exception as exc:
                logger.error(
                    "Matrix: failed to create E2EE client: %s. %s",
                    exc, _E2EE_INSTALL_HINT,
                )
                await api.session.close()
                return False

        # Register event handlers.
        from mautrix.client import InternalEventType as IntEvt

        client.add_event_handler(EventType.ROOM_MESSAGE, self._on_room_message)
        client.add_event_handler(EventType.REACTION, self._on_reaction)
        client.add_event_handler(IntEvt.INVITE, self._on_invite)

        if self._encryption and getattr(client, "crypto", None):
            client.add_event_handler(EventType.ROOM_ENCRYPTED, self._on_encrypted_event)

        # Initial sync to catch up, then start background sync.
        self._startup_ts = time.time()
        self._closing = False

        try:
            sync_data = await client.sync(timeout=10000, full_state=True)
            if isinstance(sync_data, dict):
                rooms_join = sync_data.get("rooms", {}).get("join", {})
                self._joined_rooms = set(rooms_join.keys())
                logger.info(
                    "Matrix: initial sync complete, joined %d rooms",
                    len(self._joined_rooms),
                )
                # Build DM room cache from m.direct account data.
                await self._refresh_dm_cache()
            else:
                logger.warning("Matrix: initial sync returned unexpected type %s", type(sync_data).__name__)
        except Exception as exc:
            logger.warning("Matrix: initial sync error: %s", exc)

        # Share keys after initial sync if E2EE is enabled.
        if self._encryption and getattr(client, "crypto", None):
            try:
                await client.crypto.share_keys()
            except Exception as exc:
                logger.warning("Matrix: initial key share failed: %s", exc)

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

        # Persist E2EE crypto store before closing so the next restart
        # can decrypt events using sessions from this run.
        if self._client and self._encryption and getattr(self._client, "crypto", None):
            try:
                import hashlib, hmac, pickle
                crypto_store = self._client.crypto.crypto_store
                _STORE_DIR.mkdir(parents=True, exist_ok=True)
                pickle_path = _CRYPTO_PICKLE_PATH
                payload = pickle.dumps(crypto_store)
                hmac_key = f"{self._user_id}:{self._device_id}".encode()
                sig = hmac.new(hmac_key, payload, hashlib.sha256).digest()
                pickle_path.write_bytes(sig + payload)
                logger.info("Matrix: persisted E2EE crypto store to %s", pickle_path)
            except Exception as exc:
                logger.debug("Matrix: could not persist crypto store on disconnect: %s", exc)

        if self._client:
            try:
                await self._client.api.session.close()
            except Exception:
                pass
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

            try:
                event_id = await asyncio.wait_for(
                    self._client.send_message_event(
                        RoomID(chat_id),
                        EventType.ROOM_MESSAGE,
                        msg_content,
                    ),
                    timeout=45,
                )
                last_event_id = str(event_id)
                logger.info("Matrix: sent event %s to %s", last_event_id, chat_id)
            except Exception as exc:
                # On E2EE errors, retry after sharing keys.
                if self._encryption and getattr(self._client, "crypto", None):
                    try:
                        await self._client.crypto.share_keys()
                        event_id = await asyncio.wait_for(
                            self._client.send_message_event(
                                RoomID(chat_id),
                                EventType.ROOM_MESSAGE,
                                msg_content,
                            ),
                            timeout=45,
                        )
                        last_event_id = str(event_id)
                        logger.info("Matrix: sent event %s to %s (after key share)", last_event_id, chat_id)
                        continue
                    except Exception as retry_exc:
                        logger.error("Matrix: failed to send to %s after retry: %s", chat_id, retry_exc)
                        return SendResult(success=False, error=str(retry_exc))
                logger.error("Matrix: failed to send to %s: %s", chat_id, exc)
                return SendResult(success=False, error=str(exc))

        return SendResult(success=True, message_id=last_event_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return room name and type (dm/group)."""
        name = chat_id
        chat_type = "dm" if await self._is_dm_room(chat_id) else "group"

        if self._client:
            try:
                name_evt = await self._client.get_state_event(
                    RoomID(chat_id), EventType.ROOM_NAME,
                )
                if name_evt and hasattr(name_evt, "name") and name_evt.name:
                    name = name_evt.name
            except Exception:
                pass

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
                await self._client.set_typing(RoomID(chat_id), timeout=30000)
            except Exception:
                pass

    async def edit_message(
        self, chat_id: str, message_id: str, content: str
    ) -> SendResult:
        """Edit an existing message (via m.replace)."""

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

        try:
            event_id = await self._client.send_message_event(
                RoomID(chat_id), EventType.ROOM_MESSAGE, msg_content,
            )
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

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

        # Upload to homeserver.
        try:
            mxc_url = await self._client.upload_media(
                data,
                mime_type=content_type,
                filename=filename,
            )
        except Exception as exc:
            logger.error("Matrix: upload failed: %s", exc)
            return SendResult(success=False, error=str(exc))

        # Build media message content.
        msg_content: Dict[str, Any] = {
            "msgtype": msgtype,
            "body": caption or filename,
            "url": str(mxc_url),
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

        try:
            event_id = await self._client.send_message_event(
                RoomID(room_id), EventType.ROOM_MESSAGE, msg_content,
            )
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

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
        while not self._closing:
            try:
                sync_data = await self._client.sync(timeout=30000)
                if isinstance(sync_data, dict):
                    # Update joined rooms from sync response.
                    rooms_join = sync_data.get("rooms", {}).get("join", {})
                    if rooms_join:
                        self._joined_rooms.update(rooms_join.keys())

                # Share keys periodically if E2EE is enabled.
                if self._encryption and getattr(self._client, "crypto", None):
                    try:
                        await self._client.crypto.share_keys()
                    except Exception as exc:
                        logger.warning("Matrix: E2EE key share failed: %s", exc)

                # Retry any buffered undecrypted events.
                if self._pending_megolm:
                    await self._retry_pending_decryptions()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Detect permanent auth/permission failures.
                err_str = str(exc).lower()
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "forbidden" in err_str:
                    logger.error("Matrix: permanent auth error: %s — stopping sync", exc)
                    return
                logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)

    async def _retry_pending_decryptions(self) -> None:
        """Retry decrypting buffered encrypted events after new keys arrive."""
        client = self._client
        if not client or not self._pending_megolm:
            return
        crypto = getattr(client, "crypto", None)
        if not crypto:
            return

        now = time.time()
        still_pending: list = []

        for room_id, event, ts in self._pending_megolm:
            # Drop events that have aged past the TTL.
            if now - ts > _PENDING_EVENT_TTL:
                logger.debug(
                    "Matrix: dropping expired pending event %s (age %.0fs)",
                    getattr(event, "event_id", "?"), now - ts,
                )
                continue

            try:
                decrypted = await crypto.decrypt_megolm_event(event)
            except Exception:
                still_pending.append((room_id, event, ts))
                continue

            if decrypted is None or decrypted is event:
                still_pending.append((room_id, event, ts))
                continue

            logger.info(
                "Matrix: decrypted buffered event %s",
                getattr(event, "event_id", "?"),
            )

            # Route to the appropriate handler.
            # Remove from dedup set so _on_room_message doesn't drop it
            # (the encrypted event ID was already registered by _on_encrypted_event).
            decrypted_id = str(getattr(decrypted, "event_id", getattr(event, "event_id", "")))
            if decrypted_id:
                self._processed_events_set.discard(decrypted_id)
            try:
                await self._on_room_message(decrypted)
            except Exception as exc:
                logger.warning(
                    "Matrix: error processing decrypted event %s: %s",
                    getattr(event, "event_id", "?"), exc,
                )

        self._pending_megolm = still_pending

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    async def _on_room_message(self, event: Any) -> None:
        """Handle incoming room message events (text, media)."""
        room_id = str(getattr(event, "room_id", ""))
        sender = str(getattr(event, "sender", ""))

        # Ignore own messages.
        if sender == self._user_id:
            return

        # Deduplicate by event ID.
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            return

        # Startup grace: ignore old messages from initial sync.
        raw_ts = getattr(event, "timestamp", None) or getattr(event, "server_timestamp", None) or 0
        event_ts = raw_ts / 1000.0 if raw_ts else 0.0
        if event_ts and event_ts < self._startup_ts - _STARTUP_GRACE_SECONDS:
            return

        # Extract content from the event.
        content = getattr(event, "content", None)
        if content is None:
            return

        # Get msgtype — either from content object or raw dict.
        if hasattr(content, "msgtype"):
            msgtype = str(content.msgtype)
        elif isinstance(content, dict):
            msgtype = content.get("msgtype", "")
        else:
            msgtype = ""

        # Determine source content dict for relation/thread extraction.
        if isinstance(content, dict):
            source_content = content
        elif hasattr(content, "serialize"):
            source_content = content.serialize()
        else:
            source_content = {}

        relates_to = source_content.get("m.relates_to", {})

        # Skip edits (m.replace relation).
        if relates_to.get("rel_type") == "m.replace":
            return

        # Ignore m.notice to prevent bot-to-bot loops (m.notice is the
        # conventional msgtype for bot responses in the Matrix ecosystem).
        if msgtype == "m.notice":
            return

        # Dispatch by msgtype.
        media_msgtypes = ("m.image", "m.audio", "m.video", "m.file")
        if msgtype in media_msgtypes:
            await self._handle_media_message(room_id, sender, event_id, event_ts, source_content, relates_to, msgtype)
        elif msgtype == "m.text":
            await self._handle_text_message(room_id, sender, event_id, event_ts, source_content, relates_to)

    async def _resolve_message_context(
        self,
        room_id: str,
        sender: str,
        event_id: str,
        body: str,
        source_content: dict,
        relates_to: dict,
    ) -> Optional[tuple]:
        """Shared mention/thread/DM gating for text and media handlers.

        Returns (body, is_dm, chat_type, thread_id, display_name, source)
        or None if the message should be dropped (mention gating).
        """
        is_dm = await self._is_dm_room(room_id)
        chat_type = "dm" if is_dm else "group"

        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")

        formatted_body = source_content.get("formatted_body")
        is_mentioned = self._is_bot_mentioned(body, formatted_body)

        # Require-mention gating.
        if not is_dm:
            is_free_room = room_id in self._free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._bot_participated_threads)
            if self._require_mention and not is_free_room and not in_bot_thread:
                if not is_mentioned:
                    return None

        # DM mention-thread.
        if is_dm and not thread_id and self._dm_mention_threads and is_mentioned:
            thread_id = event_id
            self._track_thread(thread_id)

        # Strip mention from body.
        if is_mentioned:
            body = self._strip_mention(body)

        # Auto-thread.
        if not is_dm and not thread_id and self._auto_thread:
            thread_id = event_id
            self._track_thread(thread_id)

        display_name = await self._get_display_name(room_id, sender)
        source = self.build_source(
            chat_id=room_id,
            chat_type=chat_type,
            user_id=sender,
            user_name=display_name,
            thread_id=thread_id,
        )

        if thread_id:
            self._track_thread(thread_id)

        self._background_read_receipt(room_id, event_id)

        return body, is_dm, chat_type, thread_id, display_name, source

    async def _handle_text_message(
        self,
        room_id: str,
        sender: str,
        event_id: str,
        event_ts: float,
        source_content: dict,
        relates_to: dict,
    ) -> None:
        """Process a text message event."""
        body = source_content.get("body", "") or ""
        if not body:
            return

        ctx = await self._resolve_message_context(
            room_id, sender, event_id, body, source_content, relates_to,
        )
        if ctx is None:
            return
        body, is_dm, chat_type, thread_id, display_name, source = ctx

        # Reply-to detection.
        reply_to = None
        in_reply_to = relates_to.get("m.in_reply_to", {})
        if in_reply_to:
            reply_to = in_reply_to.get("event_id")

        # Strip reply fallback from body.
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

        msg_type = MessageType.TEXT
        if body.startswith(("!", "/")):
            msg_type = MessageType.COMMAND

        msg_event = MessageEvent(
            text=body,
            message_type=msg_type,
            source=source,
            raw_message=source_content,
            message_id=event_id,
            reply_to_message_id=reply_to,
        )

        if msg_type == MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    async def _handle_media_message(
        self,
        room_id: str,
        sender: str,
        event_id: str,
        event_ts: float,
        source_content: dict,
        relates_to: dict,
        msgtype: str,
    ) -> None:
        """Process a media message event (image, audio, video, file)."""
        body = source_content.get("body", "") or ""
        url = source_content.get("url", "")

        # Convert mxc:// to HTTP URL for downstream processing.
        http_url = ""
        if url and url.startswith("mxc://"):
            http_url = self._mxc_to_http(url)

        # Extract MIME type from content info.
        content_info = source_content.get("info", {})
        if not isinstance(content_info, dict):
            content_info = {}
        event_mimetype = content_info.get("mimetype", "")

        # For encrypted media, the URL may be in file.url.
        file_content = source_content.get("file", {})
        if not url and isinstance(file_content, dict):
            url = file_content.get("url", "") or ""
            if url and url.startswith("mxc://"):
                http_url = self._mxc_to_http(url)

        is_encrypted_media = bool(file_content and isinstance(file_content, dict) and file_content.get("url"))

        media_type = "application/octet-stream"
        msg_type = MessageType.DOCUMENT
        is_voice_message = False

        if msgtype == "m.image":
            msg_type = MessageType.PHOTO
            media_type = event_mimetype or "image/png"
        elif msgtype == "m.audio":
            if source_content.get("org.matrix.msc3245.voice") is not None:
                is_voice_message = True
                msg_type = MessageType.VOICE
            else:
                msg_type = MessageType.AUDIO
            media_type = event_mimetype or "audio/ogg"
        elif msgtype == "m.video":
            msg_type = MessageType.VIDEO
            media_type = event_mimetype or "video/mp4"
        elif event_mimetype:
            media_type = event_mimetype

        # Cache media locally when downstream tools need a real file path.
        cached_path = None
        should_cache_locally = (
            msg_type == MessageType.PHOTO or is_voice_message or is_encrypted_media
        )
        if should_cache_locally and url:
            try:
                file_bytes = await self._client.download_media(ContentURI(url))
                if file_bytes is not None:
                    if is_encrypted_media:
                        from mautrix.crypto.attachments import decrypt_attachment

                        hashes_value = file_content.get("hashes") if isinstance(file_content, dict) else None
                        hash_value = hashes_value.get("sha256") if isinstance(hashes_value, dict) else None

                        key_value = file_content.get("key") if isinstance(file_content, dict) else None
                        if isinstance(key_value, dict):
                            key_value = key_value.get("k")

                        iv_value = file_content.get("iv") if isinstance(file_content, dict) else None

                        if key_value and hash_value and iv_value:
                            file_bytes = decrypt_attachment(file_bytes, key_value, hash_value, iv_value)
                        else:
                            logger.warning(
                                "[Matrix] Encrypted media event missing decryption metadata for %s",
                                event_id,
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

        ctx = await self._resolve_message_context(
            room_id, sender, event_id, body, source_content, relates_to,
        )
        if ctx is None:
            return
        body, is_dm, chat_type, thread_id, display_name, source = ctx

        allow_http_fallback = bool(http_url) and not is_encrypted_media
        media_urls = [cached_path] if cached_path else ([http_url] if allow_http_fallback else None)
        media_types = [media_type] if media_urls else None

        msg_event = MessageEvent(
            text=body,
            message_type=msg_type,
            source=source,
            raw_message=source_content,
            message_id=event_id,
            media_urls=media_urls,
            media_types=media_types,
        )

        await self.handle_message(msg_event)

    async def _on_encrypted_event(self, event: Any) -> None:
        """Handle encrypted events that could not be auto-decrypted."""
        room_id = str(getattr(event, "room_id", ""))
        event_id = str(getattr(event, "event_id", ""))

        if self._is_duplicate_event(event_id):
            return

        logger.warning(
            "Matrix: could not decrypt event %s in %s — buffering for retry",
            event_id, room_id,
        )

        self._pending_megolm.append((room_id, event, time.time()))
        if len(self._pending_megolm) > _MAX_PENDING_EVENTS:
            self._pending_megolm = self._pending_megolm[-_MAX_PENDING_EVENTS:]

    async def _on_invite(self, event: Any) -> None:
        """Auto-join rooms when invited."""

        room_id = str(getattr(event, "room_id", ""))

        logger.info(
            "Matrix: invited to %s — joining",
            room_id,
        )
        try:
            await self._client.join_room(RoomID(room_id))
            self._joined_rooms.add(room_id)
            logger.info("Matrix: joined %s", room_id)
            await self._refresh_dm_cache()
        except Exception as exc:
            logger.warning("Matrix: error joining %s: %s", room_id, exc)

    # ------------------------------------------------------------------
    # Reactions (send, receive, processing lifecycle)
    # ------------------------------------------------------------------

    async def _send_reaction(
        self, room_id: str, event_id: str, emoji: str,
    ) -> Optional[str]:
        """Send an emoji reaction to a message in a room.
        Returns the reaction event_id on success, None on failure.
        """

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
            resp_event_id = await self._client.send_message_event(
                RoomID(room_id), EventType.REACTION, content,
            )
            logger.debug("Matrix: sent reaction %s to %s", emoji, event_id)
            return str(resp_event_id)
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

    async def _on_reaction(self, event: Any) -> None:
        """Handle incoming reaction events."""
        sender = str(getattr(event, "sender", ""))
        if sender == self._user_id:
            return
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            return

        room_id = str(getattr(event, "room_id", ""))
        content = getattr(event, "content", None)
        if content:
            relates_to = content.get("m.relates_to", {}) if isinstance(content, dict) else getattr(content, "relates_to", {})
            reacts_to = ""
            key = ""
            if isinstance(relates_to, dict):
                reacts_to = relates_to.get("event_id", "")
                key = relates_to.get("key", "")
            elif hasattr(relates_to, "event_id"):
                reacts_to = str(getattr(relates_to, "event_id", ""))
                key = str(getattr(relates_to, "key", ""))
            logger.info(
                "Matrix: reaction %s from %s on %s in %s",
                key, sender, reacts_to, room_id,
            )

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
        """Buffer a text event and reset the flush timer."""
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
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)

        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
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
        """Send a read receipt (m.read) for an event."""
        if not self._client:
            return False
        try:
            await self._client.set_read_markers(
                RoomID(room_id),
                fully_read_event=EventID(event_id),
                read_receipt=EventID(event_id),
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
        if not self._client:
            return False
        try:
            await self._client.redact(
                RoomID(room_id), EventID(event_id), reason=reason or None,
            )
            logger.info("Matrix: redacted %s in %s", event_id, room_id)
            return True
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
        """Fetch recent messages from a room."""
        if not self._client:
            return []
        try:
            resp = await self._client.get_messages(
                RoomID(room_id),
                direction=PaginationDirection.BACKWARD,
                from_token=SyncToken(start) if start else None,
                limit=limit,
            )
        except Exception as exc:
            logger.warning("Matrix: get_messages failed for %s: %s", room_id, exc)
            return []

        if not resp:
            return []

        events = getattr(resp, "chunk", []) or (resp.get("chunk", []) if isinstance(resp, dict) else [])
        messages = []
        for event in reversed(events):
            body = ""
            content = getattr(event, "content", None)
            if content:
                if hasattr(content, "body"):
                    body = content.body or ""
                elif isinstance(content, dict):
                    body = content.get("body", "")
            messages.append({
                "event_id": str(getattr(event, "event_id", "")),
                "sender": str(getattr(event, "sender", "")),
                "body": body,
                "timestamp": getattr(event, "timestamp", 0) or getattr(event, "server_timestamp", 0),
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
        """Create a new Matrix room."""
        if not self._client:
            return None
        try:
            preset_enum = {
                "private_chat": RoomCreatePreset.PRIVATE,
                "public_chat": RoomCreatePreset.PUBLIC,
                "trusted_private_chat": RoomCreatePreset.TRUSTED_PRIVATE,
            }.get(preset, RoomCreatePreset.PRIVATE)
            invitees = [UserID(u) for u in (invite or [])]
            room_id = await self._client.create_room(
                name=name or None,
                topic=topic or None,
                invitees=invitees,
                is_direct=is_direct,
                preset=preset_enum,
            )
            room_id_str = str(room_id)
            self._joined_rooms.add(room_id_str)
            logger.info("Matrix: created room %s (%s)", room_id_str, name or "unnamed")
            return room_id_str
        except Exception as exc:
            logger.warning("Matrix: create_room error: %s", exc)
            return None

    async def invite_user(self, room_id: str, user_id: str) -> bool:
        """Invite a user to a room."""
        if not self._client:
            return False
        try:
            await self._client.invite_user(RoomID(room_id), UserID(user_id))
            logger.info("Matrix: invited %s to %s", user_id, room_id)
            return True
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
            presence_map = {
                "online": PresenceState.ONLINE,
                "offline": PresenceState.OFFLINE,
                "unavailable": PresenceState.UNAVAILABLE,
            }
            await self._client.set_presence(
                presence=presence_map[state],
                status=status_msg or None,
            )
            logger.debug("Matrix: presence set to %s", state)
            return True
        except Exception as exc:
            logger.debug("Matrix: set_presence failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Emote & notice message types
    # ------------------------------------------------------------------

    async def _send_simple_message(
        self, chat_id: str, text: str, msgtype: str,
    ) -> SendResult:
        """Send a simple message (emote, notice) with optional HTML formatting."""
        if not self._client or not text:
            return SendResult(success=False, error="No client or empty text")

        msg_content: Dict[str, Any] = {"msgtype": msgtype, "body": text}
        html = self._markdown_to_html(text)
        if html and html != text:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = html

        try:
            event_id = await self._client.send_message_event(
                RoomID(chat_id), EventType.ROOM_MESSAGE, msg_content,
            )
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_emote(
        self, chat_id: str, text: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an emote message (/me style action)."""
        return await self._send_simple_message(chat_id, text, "m.emote")

    async def send_notice(
        self, chat_id: str, text: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a notice message (bot-appropriate, non-alerting)."""
        return await self._send_simple_message(chat_id, text, "m.notice")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _is_dm_room(self, room_id: str) -> bool:
        """Check if a room is a DM."""
        if self._dm_rooms.get(room_id, False):
            return True
        # Fallback: check member count via state store.
        state_store = getattr(self._client, "state_store", None) if self._client else None
        if state_store:
            try:
                members = await state_store.get_members(room_id)
                if members and len(members) == 2:
                    return True
            except Exception:
                pass
        return False

    async def _refresh_dm_cache(self) -> None:
        """Refresh the DM room cache from m.direct account data."""
        if not self._client:
            return

        dm_data: Optional[Dict] = None

        try:
            resp = await self._client.get_account_data("m.direct")
            if hasattr(resp, "content"):
                dm_data = resp.content
            elif isinstance(resp, dict):
                dm_data = resp
        except Exception as exc:
            logger.debug("Matrix: get_account_data('m.direct') failed: %s", exc)

        if dm_data is None:
            return

        dm_room_ids: Set[str] = set()
        for user_id, rooms in dm_data.items():
            if isinstance(rooms, list):
                dm_room_ids.update(str(r) for r in rooms)

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
        if self._user_id and self._user_id in body:
            return True
        if self._user_id and ":" in self._user_id:
            localpart = self._user_id.split(":")[0].lstrip("@")
            if localpart and re.search(r'\b' + re.escape(localpart) + r'\b', body, re.IGNORECASE):
                return True
        if formatted_body and self._user_id:
            if f"matrix.to/#/{self._user_id}" in formatted_body:
                return True
        return False

    def _strip_mention(self, body: str) -> str:
        """Remove bot mention from message body."""
        if self._user_id:
            body = body.replace(self._user_id, "")
        if self._user_id and ":" in self._user_id:
            localpart = self._user_id.split(":")[0].lstrip("@")
            if localpart:
                body = re.sub(r'\b' + re.escape(localpart) + r'\b', '', body, flags=re.IGNORECASE)
        return body.strip()

    async def _get_display_name(self, room_id: str, user_id: str) -> str:
        """Get a user's display name in a room, falling back to user_id."""
        state_store = getattr(self._client, "state_store", None) if self._client else None
        if state_store:
            try:
                member = await state_store.get_member(room_id, user_id)
                if member and getattr(member, "displayname", None):
                    return member.displayname
            except Exception:
                pass
        # Strip the @...:server format to just the localpart.
        if user_id.startswith("@") and ":" in user_id:
            return user_id[1:].split(":")[0]
        return user_id

    def _mxc_to_http(self, mxc_url: str) -> str:
        """Convert mxc://server/media_id to an HTTP download URL."""
        if not mxc_url.startswith("mxc://"):
            return mxc_url
        parts = mxc_url[6:]  # strip mxc://
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
            if "html_block" in md.preprocessors:
                md.preprocessors.deregister("html_block")

            html = md.convert(text)
            md.reset()

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
        """Sanitize a URL for use in an href attribute."""
        stripped = url.strip()
        scheme = stripped.split(":", 1)[0].lower().strip() if ":" in stripped else ""
        if scheme in ("javascript", "data", "vbscript"):
            return ""
        return stripped.replace('"', "&quot;")

    @staticmethod
    def _markdown_to_html_fallback(text: str) -> str:
        """Comprehensive regex Markdown-to-HTML for Matrix."""
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

        # HTML-escape remaining text.
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

            # Blockquote
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
        result = re.sub(r"<br>\n(</?(?:pre|blockquote|h[1-6]|ul|ol|li|hr))", r"\n\1", result)
        result = re.sub(r"(</(?:pre|blockquote|h[1-6]|ul|ol|li)>)<br>", r"\1", result)

        # Restore protected regions.
        for idx, original in enumerate(placeholders):
            result = result.replace(f"\x00PROTECTED{idx}\x00", original)

        return result
