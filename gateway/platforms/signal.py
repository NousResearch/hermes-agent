"""Signal messenger platform adapter.

Connects to a signal-cli daemon running in HTTP mode.
Inbound messages arrive via SSE (Server-Sent Events) streaming.
Outbound messages and actions use JSON-RPC 2.0 over HTTP.

Based on PR #268 by ibhagwan, rebuilt with bug fixes.

Requires:
  - signal-cli installed and running: signal-cli daemon --http 127.0.0.1:8080
  - SIGNAL_HTTP_URL and SIGNAL_ACCOUNT environment variables set
"""

import asyncio
import base64
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
)
from gateway.platforms.helpers import redact_phone
from gateway.platforms.signal_rate_limit import (
    SIGNAL_BATCH_PACING_NOTICE_THRESHOLD,
    SIGNAL_MAX_ATTACHMENTS_PER_MSG,
    SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
    SignalRateLimitError,
    _extract_retry_after_seconds,
    _format_wait,
    _is_signal_rate_limit_error,
    _signal_send_timeout,
    get_scheduler,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_MESSAGE_LENGTH = 8000  # Signal message size limit
TYPING_INTERVAL = 8.0  # seconds between typing indicator refreshes
SSE_RETRY_DELAY_INITIAL = 2.0
SSE_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0  # seconds between health checks
HEALTH_CHECK_STALE_THRESHOLD = 120.0  # seconds without SSE activity before concern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


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
    if data[:2] == b"PK":
        return ".zip"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp")


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in (".mp3", ".wav", ".ogg", ".m4a", ".aac")


_EXT_TO_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
    ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".pdf": "application/pdf", ".zip": "application/zip",
}


def _ext_to_mime(ext: str) -> str:
    """Map file extension to MIME type."""
    return _EXT_TO_MIME.get(ext.lower(), "application/octet-stream")


def _render_mentions(text: str, mentions: list) -> str:
    """Replace Signal mention placeholders (\\uFFFC) with readable @identifiers.

    Signal encodes @mentions as the Unicode object replacement character
    with out-of-band metadata containing the mentioned user's UUID/number.
    """
    if not mentions or "\uFFFC" not in text:
        return text
    # Sort mentions by start position (reverse) to replace from end to start
    # so indices don't shift as we replace
    sorted_mentions = sorted(mentions, key=lambda m: m.get("start", 0), reverse=True)
    for mention in sorted_mentions:
        start = mention.get("start", 0)
        length = mention.get("length", 1)
        # Use the mention's number or UUID as the replacement
        identifier = mention.get("number") or mention.get("uuid") or "user"
        replacement = f"@{identifier}"
        text = text[:start] + replacement + text[start + length:]
    return text


def _is_signal_service_id(value: str) -> bool:
    """Return True if *value* already looks like a Signal service identifier."""
    if not value:
        return False
    if value.startswith("PNI:") or value.startswith("u:"):
        return True
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _looks_like_e164_number(value: str) -> bool:
    """Return True for a plausible E.164 phone number."""
    if not value or not value.startswith("+"):
        return False
    digits = value[1:]
    return digits.isdigit() and 7 <= len(digits) <= 15


def check_signal_requirements() -> bool:
    """Check if Signal is configured (has URL and account)."""
    return bool(os.getenv("SIGNAL_HTTP_URL") and os.getenv("SIGNAL_ACCOUNT"))


# ---------------------------------------------------------------------------
# Signal Adapter
# ---------------------------------------------------------------------------

class SignalAdapter(BasePlatformAdapter):
    """Signal messenger adapter using signal-cli HTTP daemon."""

    platform = Platform.SIGNAL
    # Signal has no real edit API for already-sent messages. Mark it explicitly
    # so streaming suppresses the visible cursor instead of leaving a stale tofu
    # square behind in chat clients when edit attempts fail.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SIGNAL)

        extra = config.extra or {}
        self.http_url = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/")
        self.account = extra.get("account", "")
        self.ignore_stories = extra.get("ignore_stories", True)

        # Parse allowlists — group policy is derived from presence of group allowlist
        group_allowed_str = os.getenv("SIGNAL_GROUP_ALLOWED_USERS", "")
        self.group_allow_from = set(_parse_comma_list(group_allowed_str))

        # DM allowlist — mirrors SIGNAL_ALLOWED_USERS checked by run.py.
        # Stored here so the reaction hooks can skip unauthorized senders
        # (reactions fire before run.py's auth gate, so without this check
        # every inbound DM from any contact gets a 👀 reaction).
        # "*" means all users allowed (open mode); empty means no restriction
        # recorded at adapter level (run.py still enforces auth separately).
        dm_allowed_str = os.getenv("SIGNAL_ALLOWED_USERS", "*")
        self.dm_allow_from = set(_parse_comma_list(dm_allowed_str))

        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None

        # Background tasks
        self._sse_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        # Per-chat typing-indicator backoff. When signal-cli reports
        # NETWORK_FAILURE (recipient offline / unroutable), base.py's
        # _keep_typing refresh loop would otherwise hammer sendTyping every
        # ~2s indefinitely, producing WARNING-level log spam and pointless
        # RPC traffic. We track consecutive failures per chat and skip the
        # RPC during a cooldown window instead.
        self._typing_failures: Dict[str, int] = {}
        self._typing_skip_until: Dict[str, float] = {}
        self._running = False
        self._last_sse_activity = 0.0
        self._sse_response: Optional[httpx.Response] = None

        # Normalize account for self-message filtering
        self._account_normalized = self.account.strip()

        # Track recently sent message timestamps to prevent echo-back loops
        # in Note to Self / self-chat mode (mirrors WhatsApp recentlySentIds)
        self._recent_sent_timestamps: set = set()
        self._max_recent_timestamps = 50
        # Signal increasingly exposes ACI/PNI UUIDs as stable recipient IDs.
        # Keep a best-effort mapping so outbound sends can upgrade from a
        # phone number to the corresponding UUID when signal-cli prefers it.
        self._recipient_uuid_by_number: Dict[str, str] = {}
        self._recipient_number_by_uuid: Dict[str, str] = {}
        self._recipient_cache_lock = asyncio.Lock()

        logger.info("Signal adapter initialized: url=%s account=%s groups=%s",
                     self.http_url, redact_phone(self.account),
                     "enabled" if self.group_allow_from else "disabled")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to signal-cli daemon and start SSE listener."""
        if not self.http_url or not self.account:
            logger.error("Signal: SIGNAL_HTTP_URL and SIGNAL_ACCOUNT are required")
            return False

        # Acquire scoped lock to prevent duplicate Signal listeners for the same phone
        lock_acquired = False
        try:
            if not self._acquire_platform_lock('signal-phone', self.account, 'Signal account'):
                return False
            lock_acquired = True
        except Exception as e:
            logger.warning("Signal: Could not acquire phone lock (non-fatal): %s", e)

        # Tighter keepalive so idle CLOSE_WAIT drains promptly (#18451).
        from gateway.platforms._http_client_limits import platform_httpx_limits
        self.client = httpx.AsyncClient(timeout=30.0, limits=platform_httpx_limits())
        try:
            # Health check — verify signal-cli daemon is reachable
            try:
                resp = await self.client.get(f"{self.http_url}/api/v1/check", timeout=10.0)
                if resp.status_code != 200:
                    logger.error("Signal: health check failed (status %d)", resp.status_code)
                    return False
            except Exception as e:
                logger.error("Signal: cannot reach signal-cli at %s: %s", self.http_url, e)
                return False

            self._running = True
            self._last_sse_activity = time.time()
            self._sse_task = asyncio.create_task(self._sse_listener())
            self._health_monitor_task = asyncio.create_task(self._health_monitor())

            logger.info("Signal: connected to %s", self.http_url)
            return True
        finally:
            if not self._running:
                if self.client:
                    await self.client.aclose()
                    self.client = None
                if lock_acquired:
                    self._release_platform_lock()

    async def disconnect(self) -> None:
        """Stop SSE listener and clean up."""
        self._running = False

        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # Cancel all typing tasks
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        if self.client:
            await self.client.aclose()
            self.client = None

        self._release_platform_lock()

        logger.info("Signal: disconnected")

    # ------------------------------------------------------------------
    # SSE Streaming (inbound messages)
    # ------------------------------------------------------------------

    async def _sse_listener(self) -> None:
        """Listen for SSE events from signal-cli daemon."""
        url = f"{self.http_url}/api/v1/events?account={quote(self.account, safe='')}"
        backoff = SSE_RETRY_DELAY_INITIAL

        while self._running:
            try:
                logger.debug("Signal SSE: connecting to %s", url)
                async with self.client.stream(
                    "GET", url,
                    headers={"Accept": "text/event-stream"},
                    timeout=None,
                ) as response:
                    self._sse_response = response
                    backoff = SSE_RETRY_DELAY_INITIAL  # Reset on successful connection
                    self._last_sse_activity = time.time()
                    logger.info("Signal SSE: connected")

                    buffer = ""
                    async for chunk in response.aiter_text():
                        if not self._running:
                            break
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            # SSE keepalive comments (":") prove the connection
                            # is alive — update activity so the health monitor
                            # doesn't report false idle warnings.
                            if line.startswith(":"):
                                self._last_sse_activity = time.time()
                                continue
                            # Parse SSE data lines
                            if line.startswith("data:"):
                                data_str = line[5:].strip()
                                if not data_str:
                                    continue
                                self._last_sse_activity = time.time()
                                try:
                                    data = json.loads(data_str)
                                    await self._handle_envelope(data)
                                except json.JSONDecodeError:
                                    logger.debug("Signal SSE: invalid JSON: %s", data_str[:100])
                                except Exception:
                                    logger.exception("Signal SSE: error handling event")

            except asyncio.CancelledError:
                break
            except httpx.HTTPError as e:
                if self._running:
                    logger.warning("Signal SSE: HTTP error: %s (reconnecting in %.0fs)", e, backoff)
            except Exception as e:
                if self._running:
                    logger.warning("Signal SSE: error: %s (reconnecting in %.0fs)", e, backoff)

            if self._running:
                # Add 20% jitter to prevent thundering herd on reconnection
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)

        self._sse_response = None

    # ------------------------------------------------------------------
    # Health Monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Monitor SSE connection health and force reconnect if stale."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break

            elapsed = time.time() - self._last_sse_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.warning("Signal: SSE idle for %.0fs, checking daemon health", elapsed)
                try:
                    resp = await self.client.get(
                        f"{self.http_url}/api/v1/check", timeout=10.0
                    )
                    if resp.status_code == 200:
                        # Daemon is alive but SSE is idle — update activity to
                        # avoid repeated warnings (connection may just be quiet)
                        self._last_sse_activity = time.time()
                        logger.debug("Signal: daemon healthy, SSE idle")
                    else:
                        logger.warning("Signal: health check failed (%d), forcing reconnect", resp.status_code)
                        self._force_reconnect()
                except Exception as e:
                    logger.warning("Signal: health check error: %s, forcing reconnect", e)
                    self._force_reconnect()

    def _force_reconnect(self) -> None:
        """Force SSE reconnection by closing the current response."""
        if self._sse_response and not self._sse_response.is_stream_consumed:
            try:
                task = asyncio.create_task(self._sse_response.aclose())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                pass
            self._sse_response = None

    # ------------------------------------------------------------------
    # Message Handling
    # ------------------------------------------------------------------

    async def _handle_envelope(self, envelope: dict) -> None:
        """Process an incoming signal-cli envelope."""
        # Unwrap nested envelope if present
        envelope_data = envelope.get("envelope", envelope)

        # Handle syncMessage: extract "Note to Self" messages (sent to own account)
        # while still filtering other sync events (read receipts, typing, etc.)
        is_note_to_self = False
        if "syncMessage" in envelope_data:
            sync_msg = envelope_data.get("syncMessage")
            if sync_msg and isinstance(sync_msg, dict):
                sent_msg = sync_msg.get("sentMessage")
                if sent_msg and isinstance(sent_msg, dict):
                    dest = sent_msg.get("destinationNumber") or sent_msg.get("destination")
                    sent_ts = sent_msg.get("timestamp")
                    if dest == self._account_normalized:
                        # Check if this is an echo of our own outbound reply
                        if sent_ts and sent_ts in self._recent_sent_timestamps:
                            self._recent_sent_timestamps.discard(sent_ts)
                            return
                        # Genuine user Note to Self — promote to dataMessage
                        is_note_to_self = True
                        envelope_data = {**envelope_data, "dataMessage": sent_msg}
            if not is_note_to_self:
                return

        # Extract sender info
        sender = (
            envelope_data.get("sourceNumber")
            or envelope_data.get("sourceUuid")
            or envelope_data.get("source")
        )
        sender_name = envelope_data.get("sourceName", "")
        sender_uuid = envelope_data.get("sourceUuid", "")
        self._remember_recipient_identifiers(sender, sender_uuid)

        if not sender:
            logger.debug("Signal: ignoring envelope with no sender")
            return

        # Self-message filtering — prevent reply loops (but allow Note to Self)
        if self._account_normalized and sender == self._account_normalized and not is_note_to_self:
            return

        # Filter stories
        if self.ignore_stories and envelope_data.get("storyMessage"):
            return

        # Get data message — also check editMessage (edited messages contain
        # their updated dataMessage inside editMessage.dataMessage)
        data_message = (
            envelope_data.get("dataMessage")
            or (envelope_data.get("editMessage") or {}).get("dataMessage")
        )
        if not data_message:
            return

        # Check for group message
        group_info = data_message.get("groupInfo")
        group_id = group_info.get("groupId") if group_info else None
        is_group = bool(group_id)

        # Group message filtering — derived from SIGNAL_GROUP_ALLOWED_USERS:
        # - No env var set → groups disabled (default safe behavior)
        # - Env var set with group IDs → only those groups allowed
        # - Env var set with "*" → all groups allowed
        # DM auth is fully handled by run.py (_is_user_authorized)
        if is_group:
            if not self.group_allow_from:
                logger.debug("Signal: ignoring group message (no SIGNAL_GROUP_ALLOWED_USERS)")
                return
            if "*" not in self.group_allow_from and group_id not in self.group_allow_from:
                logger.debug("Signal: group %s not in allowlist", group_id[:8] if group_id else "?")
                return

        # Build chat info
        chat_id = sender if not is_group else f"group:{group_id}"
        chat_type = "group" if is_group else "dm"

        # Extract text and render mentions
        text = data_message.get("message", "")
        mentions = data_message.get("mentions", [])
        if text and mentions:
            text = _render_mentions(text, mentions)

        # Extract quote (reply-to) context from Signal dataMessage
        quote_data = data_message.get("quote") or {}
        reply_to_id = str(quote_data.get("id")) if quote_data.get("id") else None
        reply_to_text = quote_data.get("text")

        # Process attachments
        attachments_data = data_message.get("attachments", [])
        media_urls = []
        media_types = []

        if attachments_data and not getattr(self, "ignore_attachments", False):
            for att in attachments_data:
                att_id = att.get("id")
                att_size = att.get("size", 0)
                if not att_id:
                    continue
                if att_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                    logger.warning("Signal: attachment too large (%d bytes), skipping", att_size)
                    continue
                try:
                    cached_path, ext = await self._fetch_attachment(att_id)
                    if cached_path:
                        # Use contentType from Signal if available, else map from extension
                        content_type = att.get("contentType") or _ext_to_mime(ext)
                        media_urls.append(cached_path)
                        media_types.append(content_type)
                except Exception:
                    logger.exception("Signal: failed to fetch attachment %s", att_id)

        # Skip envelopes with no meaningful content (no text, no attachments).
        # Catches profile key updates, empty messages, and other metadata-only
        # envelopes that still carry a dataMessage wrapper but have nothing
        # worth processing. See issue: signal-cli logs "Profile key update" +
        # Hermes receives msg='' triggering a full agent turn for nothing.
        if (not text or not text.strip()) and not media_urls:
            logger.debug(
                "Signal: skipping contentless envelope from %s (%d attachments)",
                redact_phone(sender), len(media_urls) if media_urls else 0,
            )
            return

        # Build session source
        source = self.build_source(
            chat_id=chat_id,
            chat_name=group_info.get("groupName") if group_info else sender_name,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender_name or sender,
            user_id_alt=sender_uuid if sender_uuid else None,
            chat_id_alt=group_id if is_group else None,
        )

        # Determine message type from media
        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO

        # Parse timestamp from envelope data (milliseconds since epoch)
        ts_ms = envelope_data.get("timestamp", 0)
        if ts_ms:
            try:
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                timestamp = datetime.now(tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        # Build and dispatch event.
        # Store raw envelope data in raw_message so on_processing_start/complete
        # can extract targetAuthor + targetTimestamp for sendReaction.
        event = MessageEvent(
            source=source,
            text=text or "",
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message={"sender": sender, "timestamp_ms": ts_ms},
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
        )

        logger.debug("Signal: message from %s in %s: %s",
                      redact_phone(sender), chat_id[:20], (text or "")[:50])

        await self.handle_message(event)

    def _remember_recipient_identifiers(self, number: Optional[str], service_id: Optional[str]) -> None:
        """Cache any number↔UUID mapping observed from Signal envelopes."""
        if not number or not service_id or not _is_signal_service_id(service_id):
            return
        self._recipient_uuid_by_number[number] = service_id
        self._recipient_number_by_uuid[service_id] = number

    def _extract_contact_uuid(self, contact: Any, phone_number: str) -> Optional[str]:
        """Best-effort extraction of a Signal service ID from listContacts output."""
        if not isinstance(contact, dict):
            return None

        number = contact.get("number")
        recipient = contact.get("recipient")
        service_id = contact.get("uuid") or contact.get("serviceId")
        if not service_id:
            profile = contact.get("profile")
            if isinstance(profile, dict):
                service_id = profile.get("serviceId") or profile.get("uuid")

        if service_id and _is_signal_service_id(service_id):
            matches_number = number == phone_number or recipient == phone_number
            if matches_number:
                return service_id
        return None

    async def _resolve_recipient(self, chat_id: str) -> str:
        """Return the preferred Signal recipient identifier for a direct chat."""
        if (
            not chat_id
            or chat_id.startswith("group:")
            or _is_signal_service_id(chat_id)
            or not _looks_like_e164_number(chat_id)
        ):
            return chat_id

        cached = self._recipient_uuid_by_number.get(chat_id)
        if cached:
            return cached

        async with self._recipient_cache_lock:
            cached = self._recipient_uuid_by_number.get(chat_id)
            if cached:
                return cached

            contacts = await self._rpc("listContacts", {
                "account": self.account,
                "allRecipients": True,
            })
            if isinstance(contacts, list):
                for contact in contacts:
                    number = contact.get("number") if isinstance(contact, dict) else None
                    service_id = self._extract_contact_uuid(contact, chat_id)
                    if number and service_id:
                        self._remember_recipient_identifiers(number, service_id)

            return self._recipient_uuid_by_number.get(chat_id, chat_id)

    # ------------------------------------------------------------------
    # Attachment Handling
    # ------------------------------------------------------------------

    async def _fetch_attachment(self, attachment_id: str) -> tuple:
        """Fetch an attachment via JSON-RPC and cache it. Returns (path, ext)."""
        result = await self._rpc("getAttachment", {
            "account": self.account,
            "id": attachment_id,
        })

        if not result:
            return None, ""

        # Handle dict response (signal-cli returns {"data": "base64..."})
        if isinstance(result, dict):
            result = result.get("data")
            if not result:
                logger.warning("Signal: attachment response missing 'data' key")
                return None, ""

        # Result is base64-encoded file content
        raw_data = base64.b64decode(result)
        ext = _guess_extension(raw_data)

        if _is_image_ext(ext):
            path = cache_image_from_bytes(raw_data, ext)
        elif _is_audio_ext(ext):
            path = cache_audio_from_bytes(raw_data, ext)
        else:
            path = cache_document_from_bytes(raw_data, ext)

        return path, ext

    # ------------------------------------------------------------------
    # JSON-RPC Communication
    # ------------------------------------------------------------------

    async def _rpc(
        self,
        method: str,
        params: dict,
        rpc_id: str = None,
        *,
        log_failures: bool = True,
        raise_on_rate_limit: bool = False,
        timeout: float = 30.0,
    ) -> Any:
        """Send a JSON-RPC 2.0 request to signal-cli daemon.

        When ``log_failures=False``, error and exception paths log at DEBUG
        instead of WARNING — used by the typing-indicator path to silence
        repeated NETWORK_FAILURE spam for unreachable recipients while
        still preserving visibility for the first occurrence and for
        unrelated RPCs.

        When ``raise_on_rate_limit=True``, a Signal ``[429]`` /
        ``RateLimitException`` response raises ``SignalRateLimitError``
        instead of being swallowed — lets callers (multi-attachment send)
        opt into backoff-retry without changing default behaviour.
        """
        if not self.client:
            logger.warning("Signal: RPC called but client not connected")
            return None

        if rpc_id is None:
            rpc_id = f"{method}_{int(time.time() * 1000)}"

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": rpc_id,
        }

        try:
            resp = await self.client.post(
                f"{self.http_url}/api/v1/rpc",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                err = data["error"]
                if raise_on_rate_limit:
                    if _is_signal_rate_limit_error(err):
                        err_msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
                        retry_after = _extract_retry_after_seconds(err)
                        raise SignalRateLimitError(err_msg, retry_after=retry_after)
                if log_failures:
                    logger.warning("Signal RPC error (%s): %s", method, err)
                else:
                    logger.debug("Signal RPC error (%s): %s", method, err)
                return None

            return data.get("result")

        except SignalRateLimitError:
            raise
        except Exception as e:
            if log_failures:
                logger.warning("Signal RPC %s failed: %s", method, e)
            else:
                logger.debug("Signal RPC %s failed: %s", method, e)
            return None

    # ------------------------------------------------------------------
    # Formatting — markdown → Signal body ranges
    # ------------------------------------------------------------------

    @staticmethod
    def _markdown_to_signal(text: str) -> tuple:
        """Convert markdown to plain text + Signal textStyles list.

        Signal doesn't render markdown.  Instead it uses ``bodyRanges``
        (exposed by signal-cli as ``textStyle`` / ``textStyles`` params)
        with the format ``start:length:STYLE``.

        Positions are measured in **UTF-16 code units** (not Python code
        points) because that's what the Signal protocol uses.

        Supported styles: BOLD, ITALIC, STRIKETHROUGH, MONOSPACE.
        (Signal's SPOILER style is not currently mapped — no standard
        markdown syntax for it; would need ``||spoiler||`` parsing.)

        Returns ``(plain_text, styles_list)`` where *styles_list* may be
        empty if there's nothing to format.
        """
        import re

        def _utf16_len(s: str) -> int:
            """Length of *s* in UTF-16 code units."""
            return len(s.encode("utf-16-le")) // 2

        # Pre-process: normalize whitespace before any position tracking
        # so later operations don't invalidate recorded offsets.
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        styles: list = []

        # --- Phase 1: fenced code blocks  ```...``` → MONOSPACE ---
        _CB = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
        while m := _CB.search(text):
            inner = m.group(1).rstrip("\n")
            start = m.start()
            text = text[: m.start()] + inner + text[m.end() :]
            styles.append((start, len(inner), "MONOSPACE"))

        # --- Phase 2: heading markers  # Foo → Foo (BOLD) ---
        _HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
        new_text = ""
        last_end = 0
        for m in _HEADING.finditer(text):
            new_text += text[last_end : m.start()]
            last_end = m.end()
            eol = text.find("\n", m.end())
            if eol == -1:
                eol = len(text)
            heading_text = text[m.end() : eol]
            start = len(new_text)
            new_text += heading_text
            styles.append((start, len(heading_text), "BOLD"))
            last_end = eol
        new_text += text[last_end:]
        text = new_text

        # --- Phase 3: inline patterns (single-pass to avoid offset drift) ---
        # The old code processed each pattern sequentially, stripping markers
        # and recording positions per-pass.  Later passes shifted text without
        # adjusting earlier positions → bold/italic landed mid-word.
        #
        # Fix: collect ALL non-overlapping matches first, then strip every
        # marker in one pass so positions are computed against the final text.
        _PATTERNS = [
            (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), "BOLD"),
            (re.compile(r"__(.+?)__", re.DOTALL), "BOLD"),
            (re.compile(r"~~(.+?)~~", re.DOTALL), "STRIKETHROUGH"),
            (re.compile(r"`(.+?)`"), "MONOSPACE"),
            (re.compile(r"(?<!\*)\*(?!\*| )(.+?)(?<!\*)\*(?!\*)"), "ITALIC"),
            (re.compile(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)"), "ITALIC"),
        ]

        # Collect all non-overlapping matches (earlier patterns win ties).
        all_matches: list = []  # (start, end, g1_start, g1_end, style)
        occupied: list = []     # (start, end) intervals already claimed
        for pat, style in _PATTERNS:
            for m in pat.finditer(text):
                ms, me = m.start(), m.end()
                if not any(ms < oe and me > os for os, oe in occupied):
                    all_matches.append((ms, me, m.start(1), m.end(1), style))
                    occupied.append((ms, me))
        all_matches.sort()

        # Build removal list so we can adjust Phase 1/2 styles.
        # Each match removes its prefix markers (start..g1_start) and
        # suffix markers (g1_end..end).
        removals: list = []  # (position, length) sorted
        for ms, me, g1s, g1e, _ in all_matches:
            if g1s > ms:
                removals.append((ms, g1s - ms))
            if me > g1e:
                removals.append((g1e, me - g1e))
        removals.sort()

        # Adjust Phase 1/2 styles for characters about to be removed.
        def _adj(pos: int) -> int:
            shift = 0
            for rp, rl in removals:
                if rp < pos:
                    shift += min(rl, pos - rp)
                else:
                    break
            return pos - shift

        adjusted_prior: list = []
        for s, l, st in styles:
            ns = _adj(s)
            ne = _adj(s + l)
            if ne > ns:
                adjusted_prior.append((ns, ne - ns, st))

        # Strip all inline markers in one pass → positions are correct.
        result = ""
        last_end = 0
        inline_styles: list = []
        for ms, me, g1s, g1e, sty in all_matches:
            result += text[last_end:ms]
            pos = len(result)
            inner = text[g1s:g1e]
            result += inner
            inline_styles.append((pos, len(inner), sty))
            last_end = me
        result += text[last_end:]
        text = result

        styles = adjusted_prior + inline_styles

        # Convert code-point offsets → UTF-16 code-unit offsets
        style_strings = []
        for cp_start, cp_len, stype in sorted(styles):
            # Safety: skip any out-of-bounds styles
            if cp_start < 0 or cp_start + cp_len > len(text):
                continue
            u16_start = _utf16_len(text[:cp_start])
            u16_len = _utf16_len(text[cp_start : cp_start + cp_len])
            style_strings.append(f"{u16_start}:{u16_len}:{stype}")

        return text, style_strings

    def format_message(self, content: str) -> str:
        """Strip markdown for plain-text fallback (used by base class).

        The actual rich formatting happens in send() via _markdown_to_signal().
        """
        # This is only called if someone uses the base-class send path.
        # Our send() override bypasses this entirely.
        return content

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
        """Send a text message with native Signal formatting."""
        await self._stop_typing_indicator(chat_id)

        plain_text, text_styles = self._markdown_to_signal(content)

        params: Dict[str, Any] = {
            "account": self.account,
            "message": plain_text,
        }

        if text_styles:
            if len(text_styles) == 1:
                params["textStyle"] = text_styles[0]
            else:
                params["textStyles"] = text_styles

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        result = await self._rpc("send", params)

        if result is not None:
            self._track_sent_timestamp(result)
            # Signal has no editable message identifier. Returning None keeps the
            # stream consumer on the non-edit fallback path instead of pretending
            # future edits can remove an in-progress cursor from the chat thread.
            return SendResult(success=True, message_id=None)
        return SendResult(success=False, error="RPC send failed")

    def _track_sent_timestamp(self, rpc_result) -> None:
        """Record outbound message timestamp for echo-back filtering."""
        ts = rpc_result.get("timestamp") if isinstance(rpc_result, dict) else None
        if ts:
            self._recent_sent_timestamps.add(ts)
            if len(self._recent_sent_timestamps) > self._max_recent_timestamps:
                self._recent_sent_timestamps.pop()

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator.

        base.py's ``_keep_typing`` refresh loop calls this every ~2s while
        the agent is processing. If signal-cli returns NETWORK_FAILURE for
        this recipient (offline, unroutable, group membership lost, etc.)
        the unmitigated behaviour is: a WARNING log every 2 seconds for as
        long as the agent keeps running. Instead we:

        - silence the WARNING after the first consecutive failure (subsequent
          attempts log at DEBUG) so transport issues are still visible once
          but don't flood the log,
        - skip the RPC entirely during an exponential cooldown window once
          three consecutive failures have happened, so we stop hammering
          signal-cli with requests it can't deliver.

        A successful sendTyping clears the counters.
        """
        now = time.monotonic()
        skip_until = self._typing_skip_until.get(chat_id, 0.0)
        if now < skip_until:
            return

        params: Dict[str, Any] = {
            "account": self.account,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        fails = self._typing_failures.get(chat_id, 0)
        result = await self._rpc(
            "sendTyping",
            params,
            rpc_id="typing",
            log_failures=(fails == 0),
        )

        if result is None:
            fails += 1
            self._typing_failures[chat_id] = fails
            # After 3 consecutive failures, back off exponentially (16s,
            # 32s, 60s cap) to stop spamming signal-cli for a recipient
            # that clearly isn't reachable right now.
            if fails >= 3:
                backoff = min(60.0, 16.0 * (2 ** (fails - 3)))
                self._typing_skip_until[chat_id] = now + backoff
        else:
            self._typing_failures.pop(chat_id, None)
            self._typing_skip_until.pop(chat_id, None)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images via chunked Signal RPC calls.

        Per-image alt texts are dropped — Signal's send RPC only carries
        one shared message body. Bad images (download failure, missing
        file, oversize) are skipped with a warning so one bad URL
        doesn't lose the rest of the batch. ``human_delay`` is ignored:
        the rate-limit scheduler handles inter-batch pacing.
        """
        if not images:
            return

        scheduler = get_scheduler()
        logger.info(
            "Signal send_multiple_images: received %d image(s) for %s — "
            "scheduler state: %s",
            len(images), chat_id[:30], scheduler.state(),
        )

        await self._stop_typing_indicator(chat_id)

        attachments: List[str] = []
        skipped_download = 0
        skipped_missing = 0
        skipped_oversize = 0
        for image_url, _alt_text in images:
            if image_url.startswith("file://"):
                file_path = unquote(image_url[7:])
            else:
                try:
                    file_path = await cache_image_from_url(image_url)
                except Exception as e:
                    logger.warning("Signal: failed to download image %s: %s", image_url, e)
                    skipped_download += 1
                    continue

            if not file_path or not Path(file_path).exists():
                logger.warning("Signal: image file not found for %s", image_url)
                skipped_missing += 1
                continue

            file_size = Path(file_path).stat().st_size
            if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                logger.warning(
                    "Signal: image too large (%d bytes), skipping %s", file_size, image_url
                )
                skipped_oversize += 1
                continue

            attachments.append(file_path)

        if not attachments:
            logger.error(
                "Signal: no valid images in batch of %d "
                "(download=%d missing=%d oversize=%d)",
                len(images), skipped_download, skipped_missing, skipped_oversize,
            )
            return

        logger.info(
            "Signal send_multiple_images: %d/%d images valid, sending in chunks",
            len(attachments), len(images),
        )

        base_params: Dict[str, Any] = {
            "account": self.account,
            "message": "",
        }
        if chat_id.startswith("group:"):
            base_params["groupId"] = chat_id[6:]
        else:
            base_params["recipient"] = [await self._resolve_recipient(chat_id)]

        att_batches = [
            attachments[i:i + SIGNAL_MAX_ATTACHMENTS_PER_MSG]
            for i in range(0, len(attachments), SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ]

        for idx, att_batch in enumerate(att_batches):
            n = len(att_batch)
            estimated = scheduler.estimate_wait(n)
            logger.debug(
                "Signal batch %d/%d: %d attachments, estimated wait=%.1fs",
                idx + 1, len(att_batches), n, estimated,
            )
            if estimated >= SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                await self._notify_batch_pacing(
                    chat_id, idx + 1, len(att_batches), estimated
                )

            params = dict(base_params, attachments=att_batch)
            send_timeout = _signal_send_timeout(n)

            for attempt in range(1, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS + 1):
                await scheduler.acquire(n)
                try:
                    _rpc_t0 = time.monotonic()
                    result = await self._rpc(
                        "send", params, raise_on_rate_limit=True, timeout=send_timeout,
                    )
                    _rpc_duration = time.monotonic() - _rpc_t0
                    if result is not None:
                        self._track_sent_timestamp(result)
                        await scheduler.report_rpc_duration(_rpc_duration, n)
                        logger.info(
                            "Signal batch %d/%d: %d attachments sent in %.1fs "
                            "(attempt %d/%d)",
                            idx + 1, len(att_batches), n, _rpc_duration,
                            attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                        )
                    else:
                        # Assume the server didn't accept the batch, don't deduce tokens
                        logger.error(
                            "Signal: RPC send failed for batch %d/%d (%d attachments, "
                            "attempt %d/%d, rpc_duration=%.1fs)",
                            idx + 1, len(att_batches), n,
                            attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                            _rpc_duration,
                        )
                        # Retry transient (non-rate-limit) failures once
                        if attempt < SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                            backoff = 2.0 ** attempt
                            logger.info(
                                "Signal: retrying batch %d/%d after %.1fs backoff",
                                idx + 1, len(att_batches), backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                    break
                except SignalRateLimitError as e:
                    scheduler.feedback(e.retry_after, n)
                    if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                        logger.error(
                            "Signal: rate-limit retries exhausted on batch %d/%d "
                            "(%d attachments lost, server retry_after=%s)",
                            idx + 1, len(att_batches), n,
                            f"{e.retry_after:.0f}s" if e.retry_after else "unknown",
                        )
                        break
                    logger.warning(
                        "Signal: rate-limited on batch %d/%d "
                        "(attempt %d/%d, server retry_after=%s); "
                        "scheduler will pace the retry",
                        idx + 1, len(att_batches),
                        attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                        f"{e.retry_after:.0f}s" if e.retry_after else "unknown",
                    )

    async def _notify_batch_pacing(
        self,
        chat_id: str,
        next_batch_idx: int,
        total_batches: int,
        wait_s: float,
    ) -> None:
        """Inform the user when an inter-batch pacing wait crosses the
        notice threshold. Best-effort; logs and continues on failure."""
        try:
            await self.send(
                chat_id,
                f"(More images coming — pausing ~{_format_wait(wait_s)} "
                f"for Signal rate limit, batch {next_batch_idx}/{total_batches}.)",
            )
        except Exception as e:
            logger.warning("Signal: failed to send pacing notice: %s", e)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image. Supports http(s):// and file:// URLs."""
        await self._stop_typing_indicator(chat_id)

        # Resolve image to local path
        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            # Download remote image to cache
            try:
                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("Signal: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))

        if not file_path or not Path(file_path).exists():
            return SendResult(success=False, error="Image file not found")

        # Validate size
        file_size = Path(file_path).stat().st_size
        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return SendResult(success=False, error=f"Image too large ({file_size} bytes)")

        params: Dict[str, Any] = {
            "account": self.account,
            "message": caption or "",
            "attachments": [file_path],
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        result = await self._rpc("send", params)
        if result is not None:
            self._track_sent_timestamp(result)
            return SendResult(success=True)
        return SendResult(success=False, error="RPC send with attachment failed")

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        media_label: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Send any file as a Signal attachment via RPC.

        Shared implementation for send_document, send_image_file, send_voice,
        and send_video — avoids duplicating the validation/routing/RPC logic.
        """
        await self._stop_typing_indicator(chat_id)

        try:
            file_size = Path(file_path).stat().st_size
        except FileNotFoundError:
            return SendResult(success=False, error=f"{media_label} file not found: {file_path}")

        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return SendResult(success=False, error=f"{media_label} too large ({file_size} bytes)")

        params: Dict[str, Any] = {
            "account": self.account,
            "message": caption or "",
            "attachments": [file_path],
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        result = await self._rpc("send", params)
        if result is not None:
            self._track_sent_timestamp(result)
            return SendResult(success=True)
        return SendResult(success=False, error=f"RPC send {media_label.lower()} failed")

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        return await self._send_attachment(chat_id, file_path, "File", caption)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file as a native Signal attachment.

        Called by the gateway media delivery flow when MEDIA: tags containing
        image paths are extracted from agent responses.
        """
        return await self._send_attachment(chat_id, image_path, "Image", caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a Signal attachment.

        Signal does not distinguish voice messages from file attachments at
        the API level, so this routes through the same RPC send path.
        """
        return await self._send_attachment(chat_id, audio_path, "Audio", caption)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file as a Signal attachment."""
        return await self._send_attachment(chat_id, video_path, "Video", caption)

    # ------------------------------------------------------------------
    # Typing Indicators
    # ------------------------------------------------------------------

    async def _stop_typing_indicator(self, chat_id: str) -> None:
        """Stop a typing indicator loop for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Reset per-chat typing backoff state so the next agent turn starts
        # fresh rather than inheriting a cooldown from a prior conversation.
        self._typing_failures.pop(chat_id, None)
        self._typing_skip_until.pop(chat_id, None)

    async def stop_typing(self, chat_id: str) -> None:
        """Public interface for stopping typing — called by base adapter's
        _keep_typing finally block to clean up platform-level typing tasks."""
        await self._stop_typing_indicator(chat_id)

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    async def send_reaction(
        self,
        chat_id: str,
        emoji: str,
        target_author: str,
        target_timestamp: int,
    ) -> bool:
        """Send a reaction emoji to a specific message via signal-cli RPC.

        Args:
            chat_id: The chat (phone number or "group:<id>")
            emoji: Reaction emoji string (e.g. "👀", "✅")
            target_author: Phone number / UUID of the message author
            target_timestamp: Signal timestamp (ms) of the message to react to
        """
        params: Dict[str, Any] = {
            "account": self.account,
            "emoji": emoji,
            "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [chat_id]

        result = await self._rpc("sendReaction", params)
        if result is not None:
            return True
        logger.debug("Signal: sendReaction failed (chat=%s, emoji=%s)", chat_id[:20], emoji)
        return False

    async def remove_reaction(
        self,
        chat_id: str,
        target_author: str,
        target_timestamp: int,
    ) -> bool:
        """Remove a reaction by sending an empty-string emoji."""
        params: Dict[str, Any] = {
            "account": self.account,
            "emoji": "",
            "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
            "remove": True,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [chat_id]

        result = await self._rpc("sendReaction", params)
        return result is not None

    # ------------------------------------------------------------------
    # Processing Lifecycle Hooks (reactions as progress indicators)
    # ------------------------------------------------------------------

    def _extract_reaction_target(self, event: MessageEvent) -> Optional[tuple]:
        """Extract (target_author, target_timestamp) from a MessageEvent.

        Returns None if the event doesn't carry the raw Signal envelope data
        needed for sendReaction.
        """
        raw = event.raw_message
        if not isinstance(raw, dict):
            return None
        author = raw.get("sender")
        ts = raw.get("timestamp_ms")
        if not author or not ts:
            return None
        return (author, ts)

    def _reactions_enabled(self, event: "MessageEvent" = None) -> bool:
        """Check if message reactions are enabled for this event.

        Two gates:
        1. SIGNAL_REACTIONS env var — set to false/0/no to disable globally.
        2. DM allowlist — if SIGNAL_ALLOWED_USERS is set, only react to
           messages from senders in that list.  This prevents unauthorized
           contacts from seeing the 👀 reaction (which fires before run.py's
           auth gate and would otherwise reveal that a bot is listening).
        """
        if os.getenv("SIGNAL_REACTIONS", "true").lower() in ("false", "0", "no"):
            return False
        if event is not None:
            sender = getattr(getattr(event, "source", None), "user_id", None)
            if sender and "*" not in self.dm_allow_from and sender not in self.dm_allow_from:
                return False
        return True

    async def on_processing_start(self, event: MessageEvent) -> None:
        """React with 👀 when processing begins."""
        if not self._reactions_enabled(event):
            return
        target = self._extract_reaction_target(event)
        if target:
            await self.send_reaction(event.source.chat_id, "👀", *target)

    async def on_processing_complete(self, event: MessageEvent, outcome: "ProcessingOutcome") -> None:
        """Swap the 👀 reaction for ✅ (success) or ❌ (failure).

        On CANCELLED we leave the 👀 in place — no terminal outcome means
        the reaction should keep reflecting "in progress" (matches Telegram).
        """
        if not self._reactions_enabled(event):
            return
        if outcome == ProcessingOutcome.CANCELLED:
            return
        target = self._extract_reaction_target(event)
        if not target:
            return
        chat_id = event.source.chat_id
        # Remove the in-progress reaction, then add the final one
        await self.remove_reaction(chat_id, *target)
        if outcome == ProcessingOutcome.SUCCESS:
            await self.send_reaction(chat_id, "✅", *target)
        elif outcome == ProcessingOutcome.FAILURE:
            await self.send_reaction(chat_id, "❌", *target)

    # ------------------------------------------------------------------
    # Chat Info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat/contact."""
        if chat_id.startswith("group:"):
            return {
                "name": chat_id,
                "type": "group",
                "chat_id": chat_id,
            }

        # Try to resolve contact name
        result = await self._rpc("getContact", {
            "account": self.account,
            "contactAddress": chat_id,
        })

        name = chat_id
        if result and isinstance(result, dict):
            name = result.get("name") or result.get("profileName") or chat_id

        return {
            "name": name,
            "type": "dm",
            "chat_id": chat_id,
        }
