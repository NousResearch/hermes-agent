"""SimpleX Chat platform adapter (Hermes plugin).

Connects to a simplex-chat daemon running in WebSocket mode.
Inbound messages arrive via a persistent WebSocket connection.
Outbound messages use the same WebSocket with JSON commands.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/simplex/``. The Hermes plugin loader scans the
directory at startup, calls ``register(ctx)``, and the platform
becomes available to ``gateway/run.py`` and ``tools/send_message_tool``
through the registry — no edits to core files are required.

SimpleX chat daemon setup:
    simplex-chat -p 5225          # start daemon on port 5225
    # or via Docker:
    # docker run -p 5225:5225 simplexchat/simplex-chat-cli -p 5225

Required environment variables:
    SIMPLEX_WS_URL             WebSocket URL of the daemon
                               (default: ws://127.0.0.1:5225)

Optional environment variables:
    SIMPLEX_ALLOWED_USERS      Comma-separated allowlist. Each entry may be
                               either a numeric contactId (stable across
                               renames; visible via `/contacts` in the CLI)
                               or a contact display name (what the SimpleX
                               UI shows). Both forms are accepted.
    SIMPLEX_ALLOW_ALL_USERS    Set 'true' to allow all contacts
    SIMPLEX_AUTO_ACCEPT        Set 'false' to disable contact-request auto-accept
                               (default: 'true')
    SIMPLEX_GROUP_ALLOWED      Comma-separated group IDs to monitor, or '*'
                               for any group. Omit to disable groups entirely.
    SIMPLEX_HOME_CHANNEL       Default contact/group ID for cron delivery
    SIMPLEX_HOME_CHANNEL_NAME  Human label for the home channel
    HERMES_SIMPLEX_TEXT_BATCH_DELAY
                               Quiet-period seconds (default: 0.8) used to
                               concatenate rapid-fire inbound text messages
                               into a single MessageEvent — same pattern as
                               Telegram's text batching.

The ``websockets`` Python package is imported lazily — the plugin is
discoverable and ``hermes setup`` can describe it even when websockets is
not installed. ``check_requirements()`` returns False until the package
is present, so the gateway will not attempt to instantiate the adapter.
"""

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy import: BasePlatformAdapter and friends live in the main repo.
# Imported at module top because they're stdlib-only inside Hermes — no
# external dependency that would block the plugin from loading.
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 8000  # SimpleX has no hard limit; chunk for sanity
WS_RETRY_DELAY_INITIAL = 2.0
WS_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0
HEALTH_CHECK_STALE_THRESHOLD = 300.0

# Correlation ID prefix for requests we send so we can ignore our own echoes.
_CORR_PREFIX = "hermes-"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a stripped list."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _redact_id(contact_id: str) -> str:
    """Redact a contact/group ID for logging."""
    if not contact_id:
        return "<none>"
    s = str(contact_id)
    if len(s) <= 4:
        return s
    return s[:2] + "**" + s[-2:]


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes."""
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"GIF8":
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"%PDF":
        return ".pdf"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return ".mp4"
    if data[:4] == b"OggS":
        return ".ogg"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".opus"}


# ---------------------------------------------------------------------------
# SimpleX Adapter
# ---------------------------------------------------------------------------

class SimplexAdapter(BasePlatformAdapter):
    """SimpleX Chat adapter using the simplex-chat daemon WebSocket API.

    Instantiated by the ``adapter_factory`` passed to
    ``ctx.register_platform()`` in :func:`register`.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig, **kwargs):
        platform = Platform("simplex")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self.ws_url = extra.get("ws_url", "ws://127.0.0.1:5225").rstrip("/")

        # Contact-request auto-accept (on by default — matches the way most
        # bot deployments expect to behave). Read from env first, then fall
        # back to the value seeded by ``_env_enablement``.
        env_auto = os.getenv("SIMPLEX_AUTO_ACCEPT")
        if env_auto is not None:
            self.auto_accept = env_auto.strip().lower() not in {"0", "false", "no", ""}
        else:
            self.auto_accept = bool(extra.get("auto_accept", True))

        # Group allowlist. Without ``SIMPLEX_GROUP_ALLOWED``, group messages
        # are ignored entirely (safer default — a bot in a group otherwise
        # processes every member's traffic). Use ``*`` to accept any group.
        group_allowed_str = os.getenv("SIMPLEX_GROUP_ALLOWED", "") or extra.get(
            "group_allowed", ""
        )
        self.group_allow_from = set(_parse_comma_list(group_allowed_str))

        # Running state
        self._ws = None  # websockets connection
        self._ws_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_ws_activity = 0.0

        # Track sent correlation IDs to filter echoes
        self._pending_corr_ids: set = set()
        self._max_pending_corr = 200

        # File transfers awaiting rcvFileComplete (keyed by fileId). Populated
        # when a newChatItems event carries an unfinished rcvFileTransfer,
        # consumed when the file finishes downloading.
        self._pending_file_transfers: Dict[int, dict] = {}

        # Correlation tracking for ``_send_command``. Separate from
        # ``_pending_corr_ids`` (which is the upstream cosmetic echo filter)
        # because we actually await responses to commands we send.
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._corr_counter = 0

        # Text message batching — concatenate rapid-fire messages into one
        # event before dispatching, mirroring Telegram's batching.
        self._text_batch_delay = float(
            os.getenv("HERMES_SIMPLEX_TEXT_BATCH_DELAY", "0.8")
        )
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "SimpleX adapter initialized: url=%s auto_accept=%s groups=%s",
            self.ws_url,
            self.auto_accept,
            "enabled" if self.group_allow_from else "disabled",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the simplex-chat daemon and start the WebSocket listener."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.error(
                "SimpleX: 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return False

        if not self.ws_url:
            logger.error("SimpleX: SIMPLEX_WS_URL is required")
            return False

        # Quick connectivity check — try to open and immediately close
        try:
            import websockets as _wsclient
            async with _wsclient.connect(self.ws_url, open_timeout=10):
                pass
        except Exception as e:
            logger.error("SimpleX: cannot reach daemon at %s: %s", self.ws_url, e)
            return False

        self._running = True
        self._last_ws_activity = time.time()
        self._ws_task = asyncio.create_task(self._ws_listener())
        self._health_task = asyncio.create_task(self._health_monitor())

        if hasattr(self, "_mark_connected"):
            self._mark_connected()
        logger.info("SimpleX: connected to %s", self.ws_url)
        return True

    async def disconnect(self) -> None:
        """Stop WebSocket listener and clean up."""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Cancel pending text-batch flush timers
        for task in list(self._pending_text_batch_tasks.values()):
            if not task.done():
                task.cancel()
        self._pending_text_batch_tasks.clear()
        self._pending_text_batches.clear()

        # Cancel pending command futures
        for fut in self._pending_responses.values():
            if not fut.done():
                fut.cancel()
        self._pending_responses.clear()

        if hasattr(self, "_mark_disconnected"):
            self._mark_disconnected()
        logger.info("SimpleX: disconnected")

    # ------------------------------------------------------------------
    # WebSocket listener
    # ------------------------------------------------------------------

    async def _ws_listener(self) -> None:
        """Maintain a persistent WebSocket connection to the daemon."""
        import websockets as _wsclient
        from websockets.exceptions import ConnectionClosed

        backoff = WS_RETRY_DELAY_INITIAL

        while self._running:
            try:
                logger.debug("SimpleX WS: connecting to %s", self.ws_url)
                async with _wsclient.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    backoff = WS_RETRY_DELAY_INITIAL
                    self._last_ws_activity = time.time()
                    logger.info("SimpleX WS: connected")

                    async for raw in ws:
                        if not self._running:
                            break
                        self._last_ws_activity = time.time()
                        try:
                            msg = json.loads(raw)
                            await self._handle_event(msg)
                        except json.JSONDecodeError:
                            logger.debug("SimpleX WS: invalid JSON: %.100s", raw)
                        except Exception:
                            logger.exception("SimpleX WS: error handling event")

            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: connection closed: %s (reconnecting in %.0fs)",
                        e, backoff,
                    )
            except Exception as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: unexpected error: %s (reconnecting in %.0fs)",
                        e, backoff,
                    )
            finally:
                self._ws = None

            if self._running:
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, WS_RETRY_DELAY_MAX)

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Observe WebSocket idleness without reconnecting healthy quiet links.

        simplex-chat can legitimately stay application-silent for long periods
        when no messages arrive. The websockets client already sends protocol
        pings (see _ws_listener ping_interval/ping_timeout), so treating lack of
        chat events as a stale connection causes needless reconnect churn.
        """
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break
            elapsed = time.time() - self._last_ws_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.debug("SimpleX: WS application-idle for %.0fs", elapsed)

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        """Dispatch a daemon event to the appropriate handler."""
        # simplex-chat WebSocket messages are usually shaped as:
        #   {"corrId": "...", "resp": {"type": "newChatItems", ...}}
        # Older/examples may put the response fields at top-level. Normalize
        # both forms before dispatching, otherwise inbound chatItems are lost.
        resp = event.get("resp") if isinstance(event.get("resp"), dict) else event
        corr_id = event.get("corrId")

        # Handle correlated responses (replies to our own commands)
        if corr_id and corr_id in self._pending_responses:
            fut = self._pending_responses.pop(corr_id)
            if not fut.done():
                fut.set_result(resp)
            return

        # Cosmetic echo filter: prefixed corrIds are ours but didn't make it
        # into _pending_responses (e.g. fire-and-forget).
        if corr_id and isinstance(corr_id, str) and corr_id.startswith(_CORR_PREFIX):
            self._pending_corr_ids.discard(corr_id)
            return

        resp_type = resp.get("type") or event.get("type", "")

        # Auto-accept contact requests
        if resp_type == "contactRequest" and self.auto_accept:
            contact_req = resp.get("contactRequest", {}) or {}
            contact_req_id = contact_req.get("contactRequestId")
            if contact_req_id is not None:
                logger.info(
                    "SimpleX: auto-accepting contact request %s",
                    _redact_id(str(contact_req_id)),
                )
                await self._send_command(f"/accept {contact_req_id}")
            return

        # Early file-descriptor ready: simplex fires this before newChatItems
        # for some file types (especially large files and voice messages
        # transferred via XFTP). Send /freceive immediately so the download
        # starts; the chat item arrives in a subsequent newChatItems event.
        if resp_type == "rcvFileDescrReady":
            rcv_file = resp.get("rcvFileTransfer", {}) or {}
            file_id = rcv_file.get("fileId") if isinstance(rcv_file, dict) else None
            if file_id is not None:
                logger.debug(
                    "SimpleX: rcvFileDescrReady for fileId=%s — sending /freceive",
                    file_id,
                )
                await self._send_fire_and_forget(f"/freceive {file_id}")
            return

        # New messages — simplex-chat sends "newChatItems" with an array
        if resp_type == "newChatItems":
            chat_items = resp.get("chatItems", []) or []
            if not isinstance(chat_items, list):
                chat_items = [chat_items]
            for item in chat_items:
                try:
                    await self._handle_chat_item(item)
                except Exception:
                    logger.exception("SimpleX: error processing chat item")
            return

        # Singular variant — some daemon versions emit this
        if resp_type == "newChatItem":
            try:
                await self._handle_chat_item(resp)
            except Exception:
                logger.exception("SimpleX: error processing chat item")
            return

        # File transfer completion — deliver any deferred chat item
        if resp_type == "rcvFileComplete":
            chat_item = resp.get("chatItem", {}) or {}
            chat_item_data = chat_item.get("chatItem", {}) or {}
            file_info = chat_item_data.get("file", {}) or {}
            file_id = file_info.get("fileId") if isinstance(file_info, dict) else None
            if file_id is not None and file_id in self._pending_file_transfers:
                pending = self._pending_file_transfers.pop(file_id)
                file_source = file_info.get("fileSource", {}) or {}
                file_path = (
                    file_source.get("filePath")
                    if isinstance(file_source, dict)
                    else None
                )
                if file_path:
                    pending_item_data = pending.get("chatItem", {}) or {}
                    pending_item_data.setdefault("file", {})["fileSource"] = {
                        "filePath": file_path
                    }
                    pending["chatItem"] = pending_item_data
                    try:
                        await self._handle_chat_item(pending)
                    except Exception:
                        logger.exception(
                            "SimpleX: error processing deferred file message"
                        )
            return

        if resp_type:
            logger.debug("SimpleX: unhandled event type: %s", resp_type)

    async def _handle_chat_item(self, chat_item: dict) -> None:
        """Process a single chat item from a newChatItems event."""
        chat_info = chat_item.get("chatInfo", {}) or {}
        chat_item_data = chat_item.get("chatItem", {}) or {}

        chat_type = chat_info.get("type", "")

        meta = chat_item_data.get("meta", {}) or {}
        content = chat_item_data.get("content", {}) or {}
        msg_content = content.get("msgContent", {}) or {}

        # Filter out our own messages
        item_direction = chat_item_data.get("chatDir", {}) or {}
        direction_type = (
            item_direction.get("type", "") if isinstance(item_direction, dict) else ""
        )
        if direction_type in ("directSnd", "groupSnd"):
            return

        # Only process received messages
        content_type = content.get("type", "") if isinstance(content, dict) else ""
        if content_type != "rcvMsgContent":
            return

        # Text content
        text = ""
        msg_type_str = (
            msg_content.get("type", "") if isinstance(msg_content, dict) else ""
        )
        if msg_type_str in ("text", "file", "image", "voice", "link", "video"):
            text = msg_content.get("text", "")

        if not text and msg_type_str not in ("image", "file", "voice"):
            return

        # Sender + chat IDs
        sender_id = ""
        sender_name = ""
        chat_id = ""
        is_group = False

        if chat_type == "direct":
            contact = chat_info.get("contact", {}) or {}
            sender_id = str(contact.get("contactId", ""))
            sender_name = contact.get("localDisplayName", "") or contact.get(
                "profile", {}
            ).get("displayName", "")
            chat_id = sender_id
        elif chat_type == "group":
            group_info = chat_info.get("groupInfo", {}) or {}
            group_id = str(group_info.get("groupId", ""))
            chat_id = f"group:{group_id}"
            is_group = True

            member = item_direction.get("groupMember", {}) or {}
            sender_id = str(member.get("memberId", ""))
            sender_name = member.get("localDisplayName", "") or member.get(
                "memberProfile", {}
            ).get("displayName", "")

            # Group allowlist
            if self.group_allow_from:
                if (
                    "*" not in self.group_allow_from
                    and group_id not in self.group_allow_from
                ):
                    logger.debug(
                        "SimpleX: group %s not in allowlist",
                        _redact_id(group_id),
                    )
                    return
            else:
                logger.debug(
                    "SimpleX: ignoring group message (no SIMPLEX_GROUP_ALLOWED)"
                )
                return
        else:
            logger.debug("SimpleX: unhandled chat type: %s", chat_type)
            return

        if not sender_id:
            logger.debug("SimpleX: ignoring message with no sender")
            return

        # File / image / voice attachment handling. File info is at
        # chatItem.chatItem.file (sibling of meta, content, chatDir).
        media_urls: List[str] = []
        media_types: List[str] = []
        file_info = chat_item_data.get("file")

        if file_info and isinstance(file_info, dict):
            file_source = file_info.get("fileSource", {}) or {}
            file_path = (
                file_source.get("filePath")
                if isinstance(file_source, dict)
                else None
            )
            file_name = file_info.get("fileName", "")
            file_id = file_info.get("fileId")

            ext = ""
            if file_path:
                ext = Path(file_path).suffix.lower()
            if not ext and file_name:
                ext = Path(file_name).suffix.lower()

            # Voice notes typically arrive before the file finishes
            # downloading. Defer the message until rcvFileComplete fires.
            if not file_path and _is_audio_ext(ext) and file_id is not None:
                logger.info(
                    "SimpleX: voice file %d not yet received, accepting transfer",
                    file_id,
                )
                self._pending_file_transfers[file_id] = chat_item
                # Fire-and-forget: simplex-chat does not return a corrId reply
                # for /freceive, so awaiting one would block the event loop.
                await self._send_fire_and_forget(f"/freceive {file_id}")
                return

            if file_path:
                ext = Path(file_path).suffix.lower() or (
                    Path(file_name).suffix.lower() if file_name else ""
                )
                if _is_image_ext(ext):
                    media_urls.append(file_path)
                    media_types.append(f"image/{ext.lstrip('.')}")
                elif _is_audio_ext(ext):
                    media_urls.append(file_path)
                    media_types.append(f"audio/{ext.lstrip('.')}")
                else:
                    media_urls.append(file_path)
                    media_types.append("application/octet-stream")

        # Source
        chat_name = sender_name
        if is_group:
            group_info = chat_info.get("groupInfo", {}) or {}
            chat_name = group_info.get("localDisplayName", "") or group_info.get(
                "groupProfile", {}
            ).get("displayName", chat_id)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type="group" if is_group else "dm",
            user_id=sender_id,
            user_name=sender_name or sender_id,
        )

        # Message type
        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO
            else:
                # Catch-all: non-image/non-audio files (tagged
                # application/octet-stream above) are documents so run.py's
                # document-context injection surfaces the file to the agent.
                msg_type = MessageType.DOCUMENT

        # Timestamp
        ts_str = meta.get("itemTs") or meta.get("createdAt", "")
        try:
            if ts_str:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(tz=timezone.utc)
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        msg_event = MessageEvent(
            source=source,
            text=text or "",
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message=chat_item,
        )

        logger.debug(
            "SimpleX: message from %s in %s: %s",
            _redact_id(sender_id),
            chat_id[:20],
            (text or "")[:50],
        )

        # Batch consecutive text messages so the agent sees one combined
        # message instead of dropping earlier ones when the user pastes
        # several lines in quick succession.
        if msg_type == MessageType.TEXT and text:
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    # ------------------------------------------------------------------
    # Text message batching
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        return f"{event.source.platform.value}:{event.source.chat_id}"

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text event and reset the flush timer."""
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        if existing is None:
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = (
                    f"{existing.text}\n{event.text}" if existing.text else event.text
                )
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
            await asyncio.sleep(self._text_batch_delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info(
                "[SimpleX] Flushing text batch %s (%d chars)",
                key,
                len(event.text or ""),
            )
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    # ------------------------------------------------------------------
    # Command interface
    # ------------------------------------------------------------------

    def _make_corr_id(self) -> str:
        """Mint a new correlation ID and remember it for echo-filtering.

        We add every minted id to ``_pending_corr_ids`` so the inbound
        event loop can drop the daemon's echo of our own commands without
        ever invoking ``_handle_chat_item``. The set is bounded — when
        it grows past ``_max_pending_corr``, the oldest entries are
        evicted in a single sweep.
        """
        self._corr_counter += 1
        corr_id = f"{_CORR_PREFIX}{self._corr_counter}-{int(time.time() * 1000)}"
        self._pending_corr_ids.add(corr_id)
        if len(self._pending_corr_ids) > self._max_pending_corr:
            overflow = len(self._pending_corr_ids) - self._max_pending_corr
            for _ in range(overflow):
                try:
                    self._pending_corr_ids.pop()
                except KeyError:
                    break
        return corr_id

    async def _send_ws(self, payload: dict) -> None:
        """Fire-and-forget JSON payload write.

        Drops cleanly when the WebSocket is missing or already closed; the
        caller never has to handle reconnection — the ``_ws_listener``
        loop does that out of band.
        """
        ws = self._ws
        if not ws:
            logger.debug("SimpleX: WS send dropped (not connected)")
            return
        try:
            await ws.send(json.dumps(payload))
        except Exception as e:
            logger.warning("SimpleX: WS send error: %s", e)

    async def _send_command(
        self, command: str, timeout: float = 30.0
    ) -> Optional[dict]:
        """Send a command and await the correlated response."""
        ws = self._ws
        if not ws:
            logger.warning("SimpleX: command sent but WebSocket not connected")
            return None

        corr_id = self._make_corr_id()
        payload = json.dumps({"corrId": corr_id, "cmd": command})

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_responses[corr_id] = fut

        try:
            await ws.send(payload)
            result = await asyncio.wait_for(fut, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning("SimpleX: command timed out: %s", command[:50])
            self._pending_responses.pop(corr_id, None)
            return None
        except Exception as e:
            logger.warning("SimpleX: command failed: %s — %s", command[:50], e)
            self._pending_responses.pop(corr_id, None)
            return None

    async def _send_fire_and_forget(self, command: str) -> None:
        """Send a command without waiting for a correlated response.

        Use this for commands the daemon never sends a corrId reply for,
        such as ``/freceive``. Awaiting a corr-id reply on those would
        stall the event loop for the full command timeout.
        """
        corr_id = self._make_corr_id()
        await self._send_ws({"corrId": corr_id, "cmd": command})

    # ------------------------------------------------------------------
    # Outbound — text
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message.

        If *content* contains ``MEDIA:<path>`` tags (embedded by TTS / audio
        tools to signal file attachments), they are stripped from the text
        body and sent as native voice notes or documents.

        Groups use the structured ``/_send #<id> json [...]`` form
        because the bracket chat-command syntax (``#[<id>] text``) is
        parsed by the daemon as a display-name lookup, which silently
        drops when the group's display name isn't the literal ID. DMs
        use the simple ``@<id> text`` form which has always worked in
        production.

        The call is fire-and-forget at the WebSocket level: the daemon
        doesn't always return a corrId reply for chat commands, and
        waiting for one would serialise all outbound traffic behind a
        30-second timeout.
        """
        _voice_exts = {".ogg", ".mp3", ".wav", ".m4a", ".opus"}
        media_paths = re.findall(r"MEDIA:(\S+)", content)
        if media_paths:
            content = re.sub(r"MEDIA:\S+", "", content).strip()

        if content:
            corr_id = self._make_corr_id()
            if chat_id.startswith("group:"):
                # Structured form: addresses by numeric ID, and json.dumps
                # escapes newlines + special chars correctly.
                composed = json.dumps(
                    [{"msgContent": {"type": "text", "text": content}}]
                )
                cmd_str = f"/_send #{chat_id[6:]} json {composed}"
            else:
                cmd_str = f"@{chat_id} {content}"

            await self._send_ws({"corrId": corr_id, "cmd": cmd_str})

        for path in media_paths:
            is_voice = os.path.splitext(path)[1].lower() in _voice_exts
            if is_voice:
                media_result = await self.send_voice(chat_id, path)
            else:
                media_result = await self.send_document(chat_id, path)
            if not media_result.success:
                return media_result

        return SendResult(success=True)

    # ------------------------------------------------------------------
    # Outbound — media
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(file_path: str) -> tuple[str, str]:
        """Ensure *file_path* is a PNG and return ``(png_path, thumb_data_uri)``.

        SimpleX clients can't display WebP and a few other formats inline.
        This converts to PNG when needed and generates a small JPEG thumbnail
        for the ``image`` field in the ``/_send`` payload so the chat shows
        an inline preview. Uses Pillow when available, falls back to
        ImageMagick ``convert``.
        """
        import subprocess
        import tempfile

        p = Path(file_path)
        png_path = file_path
        thumb_uri = ""

        try:
            from PIL import Image

            img = Image.open(file_path)
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                png_path = str(p.with_suffix(".png"))
                img.save(png_path, "PNG")
            thumb = img.copy()
            thumb.thumbnail((128, 128))
            import io

            buf = io.BytesIO()
            thumb.save(buf, "JPEG", quality=70)
            thumb_uri = (
                "data:image/jpg;base64,"
                + base64.b64encode(buf.getvalue()).decode()
            )
        except ImportError:
            try:
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    png_path = str(p.with_suffix(".png"))
                    subprocess.run(
                        ["convert", file_path, png_path],
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(
                    [
                        "convert",
                        file_path,
                        "-resize",
                        "128x128",
                        "-quality",
                        "70",
                        tmp_path,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                with open(tmp_path, "rb") as f:
                    thumb_uri = (
                        "data:image/jpg;base64," + base64.b64encode(f.read()).decode()
                    )
                os.remove(tmp_path)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                logger.warning("SimpleX: image conversion unavailable: %s", exc)

        return png_path, thumb_uri

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image. Supports ``file://`` URLs and ``http(s)://`` URLs."""
        from urllib.parse import unquote

        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            try:
                from gateway.platforms.base import cache_image_from_url

                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("SimpleX: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))

        if not file_path or not Path(file_path).exists():
            return SendResult(success=False, error="Image file not found")

        png_path, thumb_uri = self._prepare_image(file_path)

        # /_send addresses by numeric ID; /f only accepts display names which
        # breaks for group IDs.
        composed = json.dumps(
            [
                {
                    "filePath": png_path,
                    "msgContent": {
                        "type": "image",
                        "image": thumb_uri,
                        "text": caption or "",
                    },
                }
            ]
        )

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/_send #{group_id} json {composed}"
        else:
            command = f"/_send @{chat_id} json {composed}"

        result = await self._send_command(command)
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Failed to send image")

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file via SimpleX."""
        return await self.send_image(
            chat_id, f"file://{image_path}", caption=caption, **kwargs
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file via SimpleX (as a file attachment)."""
        return await self.send_document(chat_id, video_path, caption=caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        if not Path(file_path).exists():
            return SendResult(success=False, error="File not found")

        composed = json.dumps(
            [
                {
                    "filePath": file_path,
                    "msgContent": {"type": "file", "text": caption or ""},
                }
            ]
        )

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/_send #{group_id} json {composed}"
        else:
            command = f"/_send @{chat_id} json {composed}"

        result = await self._send_command(command)
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Failed to send document")

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        duration: int = 0,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a SimpleX voice note (plays inline).

        SimpleX distinguishes a generic file attachment (``type: "file"``)
        from an inline voice note (``type: "voice"``). ``/f`` would deliver
        a downloadable file; the structured ``/_send`` form with
        ``msgContent.type == "voice"`` produces the voice-note player.
        """
        if not Path(audio_path).exists():
            return SendResult(success=False, error="Voice file not found")

        composed = json.dumps(
            [
                {
                    "msgContent": {
                        "type": "voice",
                        "text": caption or "",
                        "duration": duration,
                    },
                    "fileSource": {"filePath": audio_path},
                }
            ]
        )

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/_send #{group_id} json {composed}"
        else:
            command = f"/_send @{chat_id} json {composed}"

        result = await self._send_command(command)
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Failed to send voice message")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """SimpleX has no typing-indicator API — no-op."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic chat info."""
        if chat_id.startswith("group:"):
            return {"chat_id": chat_id, "type": "group", "name": chat_id[6:]}
        return {"chat_id": chat_id, "type": "dm", "name": chat_id}


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: require SIMPLEX_WS_URL AND the websockets package.

    Returning False keeps the platform out of ``get_connected_platforms()``
    so the gateway never instantiates the adapter when the dependency is
    missing or no daemon URL is configured.
    """
    if not os.getenv("SIMPLEX_WS_URL"):
        return False
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def is_connected(config) -> bool:
    """Check whether SimpleX is configured (env or config.yaml)."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the WebSocket
    client. Returns ``None`` when SimpleX isn't minimally configured.

    The special ``home_channel`` key is handled by the core hook — it
    becomes a proper ``HomeChannel`` dataclass on the ``PlatformConfig``
    rather than being merged into ``extra``.
    """
    ws_url = os.getenv("SIMPLEX_WS_URL", "").strip()
    if not ws_url:
        return None
    seed: dict = {"ws_url": ws_url}

    auto_accept = os.getenv("SIMPLEX_AUTO_ACCEPT", "").strip().lower()
    if auto_accept:
        seed["auto_accept"] = auto_accept not in {"0", "false", "no"}

    group_allowed = os.getenv("SIMPLEX_GROUP_ALLOWED", "").strip()
    if group_allowed:
        seed["group_allowed"] = group_allowed

    home = os.getenv("SIMPLEX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("SIMPLEX_HOME_CHANNEL_NAME", "").strip() or home,
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
    """Open an ephemeral WebSocket to the daemon, send, and close.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (e.g. ``hermes cron`` running as a
    separate process from ``hermes gateway``). Without this hook,
    ``deliver=simplex`` cron jobs fail with "No live adapter for platform".

    ``thread_id`` and ``force_document`` are accepted for signature parity
    with other plugins but are not meaningful here. ``media_files`` is
    accepted but only the text body is delivered — SimpleX file transfers
    require the daemon's filesystem-backed flow, which an ephemeral
    connection cannot drive safely.
    """
    try:
        import websockets as _wsclient
    except ImportError:
        return {"error": "websockets not installed. Run: pip install websockets"}

    extra = getattr(pconfig, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get(
        "ws_url", "ws://127.0.0.1:5225"
    )
    if not ws_url:
        return {"error": "SimpleX standalone send: SIMPLEX_WS_URL is required"}

    try:
        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            composed = json.dumps(
                [{"msgContent": {"type": "text", "text": message}}]
            )
            cmd_str = f"/_send #{group_id} json {composed}"
        else:
            # Direct contacts are addressed by display name without brackets.
            cmd_str = f"@{chat_id} {message}"

        payload = {
            "corrId": f"{_CORR_PREFIX}snd-{int(time.time() * 1000)}",
            "cmd": cmd_str,
        }

        async with _wsclient.connect(
            ws_url, open_timeout=10, close_timeout=5
        ) as ws:
            await ws.send(json.dumps(payload))
            # Give the daemon a moment to process the command before closing.
            await asyncio.sleep(0.5)

        return {"success": True, "platform": "simplex", "chat_id": chat_id}
    except Exception as e:
        return {"error": f"SimpleX send failed: {e}"}


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup gateway`` → SimpleX.

    Prompts for the WebSocket URL and the optional allowlist / groups /
    auto-accept / home channel. Writes to ``~/.hermes/.env`` via
    ``hermes_cli.config``.
    """
    print()
    print("SimpleX Chat setup")
    print("------------------")
    print("Requirements:")
    print("  1. simplex-chat daemon running (e.g. `simplex-chat -p 5225`).")
    print("  2. Python package `websockets` installed (`pip install websockets`).")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print(
            "hermes_cli.config not available; set SIMPLEX_* vars manually in "
            "~/.hermes/.env"
        )
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("SIMPLEX_WS_URL", "Daemon WebSocket URL (default ws://127.0.0.1:5225)")
    _prompt("SIMPLEX_ALLOWED_USERS", "Allowed contactIds or display names (comma-separated; blank=skip)")
    _prompt(
        "SIMPLEX_GROUP_ALLOWED",
        "Allowed group IDs (comma-separated, or '*' for any; blank=disable groups)",
    )
    _prompt(
        "SIMPLEX_AUTO_ACCEPT",
        "Auto-accept incoming contact requests? (true/false, default true)",
    )
    _prompt("SIMPLEX_HOME_CHANNEL", "Home channel contact/group ID (or empty)")
    print(
        "Done. Make sure the simplex-chat daemon is running before starting "
        "the gateway."
    )


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: SimplexAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["SIMPLEX_WS_URL"],
        install_hint=(
            "pip install websockets   # SimpleX adapter requires the "
            "websockets package"
        ),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="SIMPLEX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🔒",
        # SimpleX uses opaque contact IDs only — no phone numbers or email
        # addresses to redact.
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via SimpleX Chat, a private decentralised "
            "messenger. Contacts are identified by opaque internal IDs, "
            "not phone numbers or usernames. SimpleX supports standard "
            "markdown formatting. There is no typing indicator and no "
            "hard message length limit, but keep responses conversational. "
            "You can attach native images, voice notes, and arbitrary "
            "files; the adapter handles MEDIA:<path> tags by sending them "
            "as inline voice notes (audio extensions) or documents."
        ),
    )
