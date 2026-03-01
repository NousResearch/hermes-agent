"""
Signal platform adapter using signal-cli daemon HTTP/JSON-RPC.

Connects to a running `signal-cli daemon --http <host:port>`:
- Receives messages via SSE at /api/v1/events?account=<phone>
- Sends messages via JSON-RPC at /api/v1/rpc
- Sends typing indicators every 8 seconds
- Handles attachments (fetch/send with size validation)
"""

import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
import traceback

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
    get_image_cache_dir,
)

logger = logging.getLogger(__name__)

# Constants
SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024  # 100MB limit
TYPING_INTERVAL = 8.0  # seconds
SSE_RETRY_DELAY_INITIAL = 2.0
SSE_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0  # seconds between health checks
HEALTH_CHECK_STALE_THRESHOLD = (
    120.0  # seconds without data before considering connection stale
)


class SignalAdapter(BasePlatformAdapter):
    """
    Signal platform adapter using signal-cli daemon.

    Features:
    - Continuous SSE streaming with exponential backoff reconnection
    - Typing indicators (8s interval)
    - Attachment upload/download with size validation
    - DM pairing integration
    - Group message filtering
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SIGNAL)

        # Enable debug logging if SIGNAL_DEBUG is set
        if os.getenv("SIGNAL_DEBUG", "").lower() in ("true", "1", "yes"):
            logger.setLevel(logging.DEBUG)
            logger.debug("[Signal] Debug logging enabled")

        # Extract config from extra dict
        self.http_url = config.extra.get("http_url", "http://127.0.0.1:8080")
        self.account = config.extra.get("account", "")
        self.allowed_users_str = config.extra.get("allowed_users", "")
        self.group_allow_from_str = config.extra.get("group_allow_from", "")
        self.dm_policy = config.extra.get("dm_policy", "pairing")
        self.group_policy = config.extra.get("group_policy", "disabled")
        self.ignore_attachments = config.extra.get("ignore_attachments", False)
        self.ignore_stories = config.extra.get("ignore_stories", True)

        # Parse allowlists
        self.allowed_users = self._parse_comma_separated(self.allowed_users_str)
        self.group_allow_from = self._parse_comma_separated(self.group_allow_from_str)

        # HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)

        # SSE listener task
        self._sse_task: Optional[asyncio.Task] = None

        # Typing indicator tasks (per chat)
        self._typing_tasks: Dict[str, asyncio.Task] = {}

        # Health monitor task
        self._health_monitor_task: Optional[asyncio.Task] = None

        # Running state
        self._running = False

        # SSE activity tracking
        self._last_sse_activity: float = 0.0
        self._sse_response: Optional[httpx.Response] = None

        # Pairing store (inherited from base)
        from gateway.pairing import PairingStore

        self.pairing_store = PairingStore()

        logger.debug(
            f"[Signal] Config: http_url={self.http_url}, account={self.account}"
        )

    @staticmethod
    def _parse_comma_separated(value: str) -> List[str]:
        """Parse comma-separated string into list, stripping whitespace."""
        if not value or not value.strip():
            return []
        return [v.strip() for v in value.split(",") if v.strip()]

    @staticmethod
    def _redact_phone(phone: str) -> str:
        """Redact phone number for logging: +1234567890 -> +12****90"""
        if not phone:
            return "<none>"
        if len(phone) <= 8:
            return phone[:2] + "****" + phone[-2:] if len(phone) > 4 else "****"
        return phone[:4] + "****" + phone[-4:]

    @staticmethod
    def _redact_phone_list(phones: list) -> list:
        """Redact a list of phone numbers for logging."""
        return [SignalAdapter._redact_phone(p) for p in phones]

    async def connect(self) -> bool:
        """Connect to signal-cli daemon and start SSE listener."""
        logger.debug(
            f"[Signal] connect() called, http_url={self.http_url}, account={self._redact_phone(self.account)}"
        )

        if not self.http_url or not self.account:
            logger.warning("[Signal] HTTP URL or account not configured")
            logger.debug(f"[Signal] http_url={self.http_url}, account={self._redact_phone(self.account)}")
            return False

        try:
            # Health check
            health_url = f"{self.http_url}/api/v1/check"
            response = await self.client.get(health_url)
            if response.status_code != 200:
                logger.warning(f"[Signal] Health check failed: {response.status_code}")
                return False

            logger.info(
                f"[Signal] Connected to {self.http_url} for account {self._redact_phone(self.account)}"
            )

            # Start SSE listener
            self._running = True
            self._sse_task = asyncio.create_task(self._sse_listener())

            # Start health monitor
            self._health_monitor_task = asyncio.create_task(self._health_monitor())

            return True

        except Exception as e:
            logger.error(f"[Signal] Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Stop SSE listener and cleanup."""
        self._running = False

        # Cancel SSE task
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        # Cancel health monitor task
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

        # Close HTTP client
        await self.client.aclose()

        logger.info("[Signal] Disconnected")

    async def _sse_listener(self):
        """
        Continuous SSE stream with exponential backoff reconnection.

        Connects to /api/v1/events?account=<phone> and processes incoming envelopes.
        On failure, reconnects with exponential backoff (2s → 60s cap).
        """
        backoff = SSE_RETRY_DELAY_INITIAL
        url = f"{self.http_url}/api/v1/events?account={self.account}"

        logger.debug(f"[Signal] SSE listener starting for account: {self.account}")
        logger.debug(f"[Signal] SSE endpoint: {url}")

        while self._running:
            try:
                logger.info(f"[Signal] Connecting to SSE stream: {url}")
                logger.debug(f"[Signal] Current backoff: {backoff}s")

                # Use longer timeout for SSE streaming
                async with self.client.stream(
                    "GET", url, headers={"Accept": "text/event-stream"}, timeout=None
                ) as response:
                    self._sse_response = response
                    response.raise_for_status()
                    logger.info(
                        f"[Signal] SSE connection established (status: {response.status_code})"
                    )
                    backoff = SSE_RETRY_DELAY_INITIAL  # Reset on successful connection
                    self._last_sse_activity = time.time()  # Reset activity tracker

                    # Buffer for incomplete lines
                    buffer = ""
                    async for chunk in response.aiter_text():
                        # Track activity on every chunk
                        self._last_sse_activity = time.time()
                        if not self._running:
                            logger.debug(
                                "[Signal] SSE listener stopping (_running=False)"
                            )
                            break

                        # Add chunk to buffer
                        buffer += chunk

                        # Process complete lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue

                            # Handle both "data:" and "data: " formats (signal-cli uses no space)
                            if not line.startswith("data:"):
                                continue

                            try:
                                # Extract JSON after "data:" (handle both "data:" and "data: ")
                                if line.startswith("data: "):
                                    json_str = line[6:]  # Remove "data: "
                                else:
                                    json_str = line[
                                        5:
                                    ].strip()  # Remove "data:" and strip
                                data = json.loads(json_str)
                                await self._handle_envelope(data)
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"[Signal] Invalid JSON in SSE: {line[:100]} - {e}"
                                )
                            except Exception as e:
                                logger.error(f"[Signal] Error processing envelope: {type(e).__name__}: {e}")

            except httpx.HTTPError as e:
                logger.warning(f"[Signal] SSE HTTP error: {e}")
            except Exception as e:
                logger.warning(f"[Signal] SSE error: {type(e).__name__}: {e}")
                import traceback

                logger.debug(f"[Signal] SSE error traceback:\n{traceback.format_exc()}")

            if self._running:
                logger.info(
                    f"[Signal] Reconnecting in {backoff}s (exponential backoff)"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)  # Exponential backoff

        logger.debug("[Signal] SSE listener stopped")

    async def _health_monitor(self):
        """
        Background health monitor that checks SSE connection health.

        If no SSE data received for HEALTH_CHECK_STALE_THRESHOLD seconds,
        verifies daemon health and forces reconnect if unhealthy.
        Runs periodically while adapter is connected.
        """
        logger.debug("[Signal] Health monitor started")

        while self._running:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

                if not self._running:
                    break

                # Check if SSE data is stale
                time_since_activity = time.time() - self._last_sse_activity

                if time_since_activity > HEALTH_CHECK_STALE_THRESHOLD:
                    logger.warning(
                        f"[Signal] No SSE activity for {time_since_activity:.0f}s, "
                        f"checking daemon health..."
                    )

                    # Check daemon health via /api/v1/check
                    try:
                        health_url = f"{self.http_url}/api/v1/check"
                        response = await self.client.get(health_url, timeout=10.0)

                        if response.status_code != 200:
                            logger.warning(
                                f"[Signal] Daemon health check failed: {response.status_code}, "
                                f"forcing reconnect"
                            )
                            await self._force_reconnect()
                        else:
                            logger.debug(
                                "[Signal] Daemon health check passed, "
                                "connection may just be idle"
                            )
                            # Update activity timestamp to avoid repeated checks
                            self._last_sse_activity = time.time()

                    except Exception as e:
                        logger.warning(
                            f"[Signal] Daemon health check error: {e}, "
                            f"forcing reconnect"
                        )
                        await self._force_reconnect()

            except asyncio.CancelledError:
                logger.debug("[Signal] Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"[Signal] Health monitor error: {e}")

        logger.debug("[Signal] Health monitor stopped")

    async def _force_reconnect(self):
        """Force SSE reconnection by closing current response."""
        logger.info("[Signal] Forcing SSE reconnect")

        # Close current SSE response if available
        if self._sse_response and not self._sse_response.is_closed:
            try:
                await self._sse_response.aclose()
                logger.debug("[Signal] Closed stale SSE response")
            except Exception as e:
                logger.debug(f"[Signal] Error closing SSE response: {e}")

    async def _handle_envelope(self, envelope: dict):
        """
        Process incoming Signal envelope.

        Handles:
        - Story filtering
        - Allowlist/pairing checks
        - Attachment fetching
        - Message event creation
        """

        # signal-cli wraps the envelope in an "envelope" key
        # Extract the actual envelope data
        if "envelope" in envelope:
            envelope_data = envelope["envelope"]
        else:
            envelope_data = envelope

        # Extract sender info (signal-cli uses camelCase: source, sourceNumber, sourceName, sourceUuid)
        sender = (
            envelope_data.get("sourceNumber")
            or envelope_data.get("sourceUuid")
            or envelope_data.get("source")
        )
        sender_name = envelope_data.get("sourceName") or sender
        sender_uuid = envelope_data.get("sourceUuid")

        if not sender:
            logger.warning(f"[Signal] Rejecting message with no sender identifier. Keys: {list(envelope_data.keys())}")
            return

        # Check for story (filter if configured)
        if envelope_data.get("storyMessage"):
            if self.ignore_stories:
                return

        # Check if this is a group message
        data_message = envelope_data.get("dataMessage", {})

        # Skip non-message envelopes (receipts, typing indicators, etc.)
        if not data_message:
            return

        group_info = data_message.get("groupInfo", {})
        is_group = bool(group_info.get("groupId"))
        group_id = group_info.get("groupId", "") if is_group else ""

        # Check allowlist / pairing
        if not self._is_user_allowed(sender, is_group):

            # Handle unauthorized based on context
            if not is_group and self.dm_policy == "pairing":
                # Send pairing code for unauthorized DM
                logger.info(
                    f"[Signal] Sending pairing code to {self._redact_phone(sender)} (DM policy=pairing)"
                )
                try:
                    code = self.pairing_store.generate_code(
                        "signal", sender, sender_name or ""
                    )
                    if code == "__RATE_LIMITED__":
                        # User has existing pending code, inform them
                        await self.send(
                            sender,
                            "Your pairing request is still pending approval. Please wait for the owner to approve it.",
                        )
                        logger.info(
                            f"[Signal] Sent rate limit notice to {self._redact_phone(sender)} (pending approval)"
                        )
                    elif code:
                        await self.send(
                            sender,
                            f"Pairing requested. Owner: approve with `hermes pairing approve signal {code}`",
                        )
                        logger.info(f"[Signal] Sent pairing code {code} to {self._redact_phone(sender)}")
                    else:
                        logger.warning(
                            f"[Signal] Failed to generate pairing code (locked out or max pending reached)"
                        )
                except Exception as e:
                    logger.error(f"[Signal] Failed to send pairing code: {e}")
            else:
                # Unauthorized user in group or non-pairing DM - silently ignore
                logger.debug(
                    f"[Signal] Ignoring message from unauthorized user {self._redact_phone(sender)}"
                )
            return

    # Build chat info
        chat_id = sender
        chat_name = sender_name
        chat_type = "group" if is_group else "dm"

        if is_group:
            group_id = group_info.get("groupId", "")
            chat_id = f"group:{group_id}"
            chat_name = group_info.get("groupName", "Group")
            logger.debug(f"[Signal] Group message: id={group_id}, name={chat_name}")

        # Extract message text
        text = data_message.get("message", "")
        logger.debug(f"[Signal] Message text length: {len(text)} chars")
        if text:
            logger.debug(f"[Signal] Message preview: {text[:100]}...")

        # Handle attachments
        attachment_paths = []
        attachments = data_message.get("attachments", [])

        logger.debug(
            f"[Signal] Attachments count: {len(attachments)}, ignore_attachments={self.ignore_attachments}"
        )

        if attachments and not self.ignore_attachments:
            for att in attachments:
                att_id = att.get("id")
                content_type = att.get("contentType", "unknown")
                logger.debug(
                    f"[Signal] Processing attachment: id={att_id}, type={content_type}"
                )
                if att_id:
                    try:
                        path = await self._fetch_attachment(att_id)
                        if path:
                            attachment_paths.append(path)
                            logger.debug(f"[Signal] Attachment saved to: {path}")
                        else:
                            logger.warning(
                                f"[Signal] Attachment {att_id} returned None path"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[Signal] Failed to fetch attachment {att_id}: {e}"
                        )
        elif attachments and self.ignore_attachments:
            pass  # Skip logging when ignoring attachments

        # Build SessionSource
        # For Signal: 
        # - user_id is the E164 number (e.g., +15551234567)
        # - user_id_alt is the UUID (if available)
        # - chat_id_alt is the raw group ID (without "group:" prefix)
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender_name,
            user_id_alt=sender_uuid if sender_uuid and sender_uuid != sender else None,
            chat_id_alt=group_id if is_group else None,
        )

        # Determine message type
        message_type = MessageType.TEXT
        if attachment_paths:
            # Check if it's an image, audio, etc.
            first_path = attachment_paths[0] if attachment_paths else None
            if first_path:
                if first_path.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".gif", ".webp")
                ):
                    message_type = MessageType.PHOTO
                elif first_path.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
                    message_type = MessageType.AUDIO
                elif first_path.lower().endswith((".mp4", ".avi", ".mkv")):
                    message_type = MessageType.VIDEO

        # Build timestamp
        timestamp = envelope.get("timestamp", 0)
        if timestamp:
            timestamp = datetime.fromtimestamp(timestamp / 1000)
        else:
            timestamp = datetime.now()

        # Create and handle message event
        event = MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            media_urls=attachment_paths,
            timestamp=timestamp,
        )

        # Minimal message receipt log (matches other platforms)
        logger.debug(f"signal: received from {self._redact_phone(sender)} ({len(text)} chars)")

        # Note: Base class handles typing indicator in _process_message_background
        await self.handle_message(event)

    def _is_user_allowed(self, user_id: str, is_group: bool = False) -> bool:
        """
        Check if user is allowed to interact.

        Checks:
        1. Pairing store (approved users)
        2. Allowlist (SIGNAL_ALLOWED_USERS)
        3. Group allowlist (SIGNAL_GROUP_ALLOWED_USERS, if group message)
        """
        logger.debug(
            f"[Signal] Checking authorization for {self._redact_phone(user_id)} (group={is_group})"
        )
        logger.debug(f"[Signal] Allowed users: {self._redact_phone_list(self.allowed_users)}")
        logger.debug(f"[Signal] Group policy: {self.group_policy}")
        logger.debug(f"[Signal] Group allowlist: {self._redact_phone_list(self.group_allow_from)}")

        # Check pairing store first
        if self.pairing_store.is_approved("signal", user_id):
            logger.debug(f"[Signal] User {self._redact_phone(user_id)} approved via pairing store")
            return True

        # Check allowlist
        if user_id in self.allowed_users:
            logger.debug(f"[Signal] User {self._redact_phone(user_id)} in allowed_users")
            return True

        # Check for wildcard in allowlist (must be exact match)
        if self.allowed_users == ["*"]:
            logger.debug(f"[Signal] Wildcard match for {self._redact_phone(user_id)}")
            return True

        # Group-specific checks (only if not already authorized above)
        if is_group:
            if self.group_policy == "disabled":
                logger.debug(f"[Signal] Group policy is disabled, rejecting {self._redact_phone(user_id)}")
                return False
            if self.group_policy == "open":
                logger.debug(f"[Signal] Group policy is open, allowing {self._redact_phone(user_id)}")
                return True
            if self.group_policy == "allowlist":
                # Check group allowlist (must be exact wildcard match)
                if self.group_allow_from == ["*"]:
                    logger.debug(f"[Signal] Group wildcard match for {self._redact_phone(user_id)}")
                    return True
                if user_id in self.group_allow_from:
                    logger.debug(f"[Signal] User {self._redact_phone(user_id)} in group_allow_from")
                    return True
                # User not in group allowlist
                logger.debug(f"[Signal] User {self._redact_phone(user_id)} not in group allowlist")
                return False
            # Unknown group policy - reject for safety
            logger.debug(f"[Signal] Unknown group policy, rejecting {self._redact_phone(user_id)}")
            return False

        # Not a group message and not authorized above
        logger.debug(f"[Signal] User {self._redact_phone(user_id)} not authorized for DM")
        return False

    async def _fetch_attachment(self, attachment_id: str) -> Optional[str]:
        """Fetch attachment from signal-cli and cache locally."""
        try:
            response = await self.client.post(
                f"{self.http_url}/api/v1/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "getAttachment",
                    "params": {"account": self.account, "attachmentId": attachment_id},
                    "id": f"att_{attachment_id}",
                },
            )

            result = response.json()
            if "result" in result:
                # Base64 decode
                data = base64.b64decode(result["result"])

                # Guess extension from magic bytes
                ext = self._guess_extension(data)

                # Save to appropriate cache
                if self._is_image_ext(ext):
                    path = cache_image_from_bytes(data, ext)
                elif self._is_audio_ext(ext):
                    path = cache_audio_from_bytes(data, ext)
                else:
                    path = cache_document_from_bytes(data, f"attachment{ext}")

                return path
        except Exception as e:
            logger.warning(f"[Signal] Failed to fetch attachment: {e}")

        return None

    def _guess_extension(self, data: bytes) -> str:
        """Guess file extension from magic bytes."""
        if len(data) < 8:
            return ".bin"

        # Check magic bytes
        if data[:4] == b"\x89PNG":
            return ".png"
        elif data[:2] == b"\xff\xd8":
            return ".jpg"
        elif data[:4] == b"GIF8":
            return ".gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        elif data[:4] == b"%PDF":
            return ".pdf"
        elif data[:2] == b"PK":
            return ".docx"  # Could be other Office formats

        # Check for audio/video
        if len(data) >= 12 and data[4:8] == b"ftyp":
            # Check for MP4 brands: mp42, isom, avc1, M4V, etc.
            brand = data[8:12]
            if brand in (b"mp42", b"isom", b"avc1", b"M4V ", b"M4A "):
                return ".mp4"
        elif data[:4] == b"OggS":
            return ".ogg"
        elif len(data) >= 3 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
            # MP3 frame sync: 11 bits of 1s (0xFFE0 mask on byte 2)
            return ".mp3"

        return ".bin"

    def _is_image_ext(self, ext: str) -> bool:
        """Check if extension is an image."""
        return ext.lower() in [".jpg", ".jpeg", ".png", ".gif", ".webp"]

    def _is_audio_ext(self, ext: str) -> bool:
        """Check if extension is audio."""
        return ext.lower() in [".mp3", ".wav", ".ogg", ".m4a", ".aac"]

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send message via JSON-RPC send method."""
        if not self._running:
            return SendResult(success=False, error="Not connected")

        try:
            # Stop typing indicator if active
            await self._stop_typing_indicator(chat_id)

            chat_id_display = chat_id[6:] if chat_id.startswith("group:") else self._redact_phone(chat_id)
            logger.debug(f"[Signal] Sending message to {chat_id_display}: {content[:50]}...")
            logger.debug(f"[Signal] Account: {self.account}")

            # Prepare JSON-RPC params based on recipient type
            params = {
                "account": self.account,
                "message": content,
            }

            # Handle group vs DM recipients
            if chat_id.startswith("group:"):
                # Group message: use groupId parameter
                group_id = chat_id[6:]  # Remove "group:" prefix
                params["groupId"] = group_id
                logger.debug(f"[Signal] Sending to group: {group_id}")
            else:
                # DM: use recipient parameter
                params["recipient"] = chat_id
                logger.debug(f"[Signal] Sending to recipient: {self._redact_phone(chat_id)}")

            # Send via JSON-RPC
            payload = {
                "jsonrpc": "2.0",
                "method": "send",
                "params": params,
                "id": f"send_{int(time.time() * 1000)}",
            }

            response = await self.client.post(
                f"{self.http_url}/api/v1/rpc", json=payload
            )

            result = response.json()

            if "error" in result:
                return SendResult(success=False, error=result["error"])

            return SendResult(success=True, raw_response=result)

        except Exception as e:
            logger.error(f"[Signal] Send failed: {e}")
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator via JSON-RPC sendTyping."""
        if not self._running:
            return

        try:
            # Prepare params based on recipient type
            params = {"account": self.account}
            if chat_id.startswith("group:"):
                group_id = chat_id[6:]
                params["groupId"] = group_id
            else:
                params["recipient"] = chat_id

            response = await self.client.post(
                f"{self.http_url}/api/v1/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "sendTyping",
                    "params": params,
                    "id": "typing",
                },
            )
        except Exception as e:
            logger.debug(f"[Signal] Typing indicator failed: {e}")

    async def _start_typing_indicator(self, chat_id: str):
        """Start background typing indicator task."""
        chat_id_display = chat_id[6:] if chat_id.startswith("group:") else self._redact_phone(chat_id)
        if chat_id in self._typing_tasks:
            logger.debug(f"[Signal] Typing indicator already running for {chat_id_display}")
            return

        logger.debug(f"[Signal] Starting typing indicator for {chat_id_display}")

        async def typing_loop():
            iteration = 0
            while chat_id in self._typing_tasks and self._running:
                iteration += 1
                try:
                    logger.debug(f"[Signal] Sending typing indicator #{iteration} to {chat_id_display}")
                    await self.send_typing(chat_id)
                    await asyncio.sleep(TYPING_INTERVAL)
                except asyncio.CancelledError:
                    logger.debug(f"[Signal] Typing loop cancelled for {chat_id_display} after {iteration} iterations")
                    break
                except Exception as e:
                    logger.debug(f"[Signal] Typing loop error for {chat_id_display}: {e}")
                    break
            logger.debug(f"[Signal] Typing loop ended for {chat_id_display} after {iteration} iterations")

        task = asyncio.create_task(typing_loop())
        self._typing_tasks[chat_id] = task

    async def _stop_typing_indicator(self, chat_id: str):
        """Stop typing indicator task."""
        chat_id_display = chat_id[6:] if chat_id.startswith("group:") else self._redact_phone(chat_id)
        if chat_id in self._typing_tasks:
            logger.debug(f"[Signal] Stopping typing indicator for {chat_id_display}")
            self._typing_tasks[chat_id].cancel()
            try:
                await self._typing_tasks[chat_id]
            except asyncio.CancelledError:
                pass
            del self._typing_tasks[chat_id]
            logger.debug(f"[Signal] Typing indicator stopped for {chat_id_display}")
        else:
            logger.debug(f"[Signal] No typing indicator to stop for {chat_id_display}")

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send image as attachment using file paths (not base64)."""
        if not self._running:
            return SendResult(success=False, error="Not connected")

        try:
            # Get the file path to send
            file_path = None

            if image_url.startswith("file://"):
                # Read local file and copy to cache for persistence
                import urllib.parse

                source_path = image_url[7:]  # Remove "file://" prefix
                source_path = urllib.parse.unquote(source_path)  # URL decode

                path_obj = Path(source_path)
                if not path_obj.exists():
                    return SendResult(
                        success=False, error=f"Local file not found: {source_path}"
                    )

                file_ext = path_obj.suffix or ".png"
                image_data = path_obj.read_bytes()
                file_path = cache_image_from_bytes(image_data, file_ext)
                logger.debug(f"[Signal] Copied file:// image to cache: {file_path}")

            else:
                # Download image to cache directory
                try:
                    file_path = await cache_image_from_url(image_url)
                    logger.debug(f"[Signal] Downloaded image to cache: {file_path}")
                except Exception as e:
                    return SendResult(
                        success=False, error=f"Failed to download image: {e}"
                    )

            # Validate file path and size
            if not file_path:
                return SendResult(
                    success=False,
                    error="Failed to get file path for image"
                )

            path_obj = Path(file_path)
            file_size = path_obj.stat().st_size
            if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                return SendResult(
                    success=False,
                    error=f"Image too large: {file_size} bytes (max: {SIGNAL_MAX_ATTACHMENT_SIZE})",
                )

            # Prepare params based on recipient type
            params = {
                "account": self.account,
                "message": caption or "",
                "attachments": [file_path],
            }
            if chat_id.startswith("group:"):
                group_id = chat_id[6:]
                params["groupId"] = group_id
            else:
                params["recipient"] = chat_id

            # Send via JSON-RPC with file path in attachments array
            response = await self.client.post(
                f"{self.http_url}/api/v1/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "send",
                    "params": params,
                    "id": f"send_img_{int(time.time() * 1000)}",
                },
            )

            result = response.json()

            if "error" in result:
                return SendResult(success=False, error=result["error"])

            return SendResult(success=True, raw_response=result)

        except Exception as e:
            logger.error(f"[Signal] Send image failed: {e}")
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get contact/group info via JSON-RPC."""
        try:
            # Try to get contact info
            response = await self.client.post(
                f"{self.http_url}/api/v1/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "getContact",
                    "params": {"account": self.account, "contactAddress": chat_id},
                    "id": f"contact_{int(time.time() * 1000)}",
                },
            )

            result = response.json()

            if "result" in result:
                contact = result["result"]
                return {
                    "name": contact.get("name", chat_id),
                    "type": "dm",
                    "chat_id": chat_id,
                }

            return {"name": chat_id, "type": "dm", "chat_id": chat_id}

        except Exception as e:
            logger.warning(f"[Signal] Failed to get chat info: {e}")
            return {"name": chat_id, "type": "unknown", "chat_id": chat_id}

    # Note: Typing indicator is started in _handle_envelope after authorization check
