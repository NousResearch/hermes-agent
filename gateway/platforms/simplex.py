"""SimpleX Chat platform adapter.

Connects to a simplex-chat terminal app running with WebSocket API enabled.
Inbound messages arrive via WebSocket JSON events.
Outbound messages and actions use WebSocket JSON commands.

Requires:
  - simplex-chat installed and running: simplex-chat -p 5225
  - SIMPLEX_WS_URL environment variable set (e.g., ws://127.0.0.1:5225)
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
from typing import Dict, List, Optional, Any

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 8000  # SimpleX message size limit
WS_RETRY_DELAY_INITIAL = 2.0
WS_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0  # seconds between health checks
HEALTH_CHECK_STALE_THRESHOLD = 300.0  # seconds without WS activity before concern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
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
    return ext.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp")


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in (".mp3", ".wav", ".ogg", ".m4a", ".aac")


def check_simplex_requirements() -> bool:
    """Check if SimpleX is configured (has WebSocket URL)."""
    return bool(os.getenv("SIMPLEX_WS_URL"))


# ---------------------------------------------------------------------------
# SimpleX Adapter
# ---------------------------------------------------------------------------

class SimplexAdapter(BasePlatformAdapter):
    """SimpleX Chat adapter using simplex-chat WebSocket API."""

    platform = Platform.SIMPLEX
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SIMPLEX)

        extra = config.extra or {}
        self.ws_url = extra.get("ws_url", "ws://127.0.0.1:5225").rstrip("/")
        self.auto_accept = extra.get("auto_accept", True)

        # Parse group allowlist
        group_allowed_str = os.getenv("SIMPLEX_GROUP_ALLOWED", "")
        self.group_allow_from = set(_parse_comma_list(group_allowed_str))

        # WebSocket connection
        self._ws = None

        # Background tasks
        self._ws_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_ws_activity = 0.0

        # Track sent message IDs to prevent echo loops
        self._recent_sent_ids: set = set()
        self._max_recent_ids = 50

        # Correlation tracking for send commands
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._corr_counter = 0

        logger.info(
            "SimpleX adapter initialized: ws_url=%s auto_accept=%s groups=%s",
            self.ws_url,
            self.auto_accept,
            "enabled" if self.group_allow_from else "disabled",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to simplex-chat WebSocket and start listener."""
        if not self.ws_url:
            logger.error("SimpleX: SIMPLEX_WS_URL is required")
            return False

        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.error(
                "SimpleX: 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return False

        # Test connectivity
        try:
            import websockets
            async with websockets.connect(self.ws_url, open_timeout=10) as ws:
                logger.debug("SimpleX: WebSocket test connection succeeded")
        except Exception as e:
            logger.error("SimpleX: cannot reach simplex-chat at %s: %s", self.ws_url, e)
            return False

        self._running = True
        self._last_ws_activity = time.time()
        self._ws_task = asyncio.create_task(self._ws_listener())
        self._health_monitor_task = asyncio.create_task(self._health_monitor())

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

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Cancel pending futures
        for fut in self._pending_responses.values():
            if not fut.done():
                fut.cancel()
        self._pending_responses.clear()

        self._mark_disconnected()
        logger.info("SimpleX: disconnected")

    # ------------------------------------------------------------------
    # WebSocket Listener (inbound messages)
    # ------------------------------------------------------------------

    async def _ws_listener(self) -> None:
        """Listen for WebSocket events from simplex-chat."""
        import websockets
        from websockets.exceptions import ConnectionClosed

        backoff = WS_RETRY_DELAY_INITIAL

        while self._running:
            try:
                logger.debug("SimpleX WS: connecting to %s", self.ws_url)
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    backoff = WS_RETRY_DELAY_INITIAL  # Reset on successful connection
                    self._last_ws_activity = time.time()
                    logger.info("SimpleX WS: connected")

                    async for raw_message in ws:
                        if not self._running:
                            break
                        self._last_ws_activity = time.time()

                        try:
                            data = json.loads(raw_message)
                            await self._handle_event(data)
                        except json.JSONDecodeError:
                            logger.debug(
                                "SimpleX WS: invalid JSON: %s",
                                str(raw_message)[:100],
                            )
                        except Exception:
                            logger.exception("SimpleX WS: error handling event")

            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: connection closed: %s (reconnecting in %.0fs)",
                        e,
                        backoff,
                    )
            except Exception as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: error: %s (reconnecting in %.0fs)",
                        e,
                        backoff,
                    )

            self._ws = None
            if self._running:
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, WS_RETRY_DELAY_MAX)

    # ------------------------------------------------------------------
    # Health Monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Monitor WebSocket connection health."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break

            elapsed = time.time() - self._last_ws_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.debug("SimpleX: WS idle for %.0fs (no user messages, connection healthy via ping/pong)", elapsed)

    def _force_reconnect(self) -> None:
        """Force WebSocket reconnection by closing the current connection."""
        ws = self._ws
        if ws:
            try:
                task = asyncio.create_task(ws.close())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                pass
            self._ws = None

    # ------------------------------------------------------------------
    # Event Handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        """Process an incoming simplex-chat WebSocket event."""
        # simplex-chat WebSocket API sends JSON with a "resp" key containing
        # the event type, and a "corrId" for correlated request-response.
        resp = event.get("resp", {})
        corr_id = event.get("corrId")

        # Handle correlated responses (replies to our commands)
        if corr_id and corr_id in self._pending_responses:
            fut = self._pending_responses.pop(corr_id)
            if not fut.done():
                fut.set_result(resp)
            return

        resp_type = resp.get("type", "")

        # Auto-accept contact requests
        if resp_type == "contactRequest" and self.auto_accept:
            contact_req = resp.get("contactRequest", {})
            contact_req_id = contact_req.get("contactRequestId")
            if contact_req_id is not None:
                logger.info(
                    "SimpleX: auto-accepting contact request %s",
                    _redact_id(str(contact_req_id)),
                )
                await self._send_command(
                    f"/accept {contact_req_id}"
                )
            return

        # Early file-descriptor ready: simplex fires this before newChatItems for
        # some file types (especially large files and voice messages transferred via
        # XFTP).  We must send /freceive immediately so the download starts; the
        # corresponding chat item will arrive in a subsequent newChatItems event and
        # populate _pending_file_transfers as normal.
        if resp_type == "rcvFileDescrReady":
            rcv_file = resp.get("rcvFileTransfer", {})
            file_id = rcv_file.get("fileId") if isinstance(rcv_file, dict) else None
            if file_id is not None:
                logger.debug(
                    "SimpleX: rcvFileDescrReady for fileId=%s — sending /freceive",
                    file_id,
                )
                await self._send_fire_and_forget(f"/freceive {file_id}")
            return

        # Handle new messages (simplex-chat sends "newChatItems" with an array)
        if resp_type == "newChatItems":
            chat_items = resp.get("chatItems", [])
            if not isinstance(chat_items, list):
                chat_items = [chat_items]
            for item in chat_items:
                try:
                    await self._handle_chat_item(item)
                except Exception:
                    logger.exception("SimpleX: error processing chat item")
            return

        # Handle file transfer completion — deliver pending voice messages
        if resp_type == "rcvFileComplete":
            chat_item = resp.get("chatItem", {})
            chat_item_data = chat_item.get("chatItem", {})
            file_info = chat_item_data.get("file", {})
            file_id = file_info.get("fileId") if isinstance(file_info, dict) else None
            if file_id is not None and file_id in self._pending_file_transfers:
                pending = self._pending_file_transfers.pop(file_id)
                file_source = file_info.get("fileSource", {})
                file_path = (
                    file_source.get("filePath")
                    if isinstance(file_source, dict)
                    else None
                )
                if file_path:
                    pending_item_data = pending.get("chatItem", {})
                    pending_item_data.setdefault("file", {})["fileSource"] = {
                        "filePath": file_path
                    }
                    pending["chatItem"] = pending_item_data
                    try:
                        await self._handle_chat_item(pending)
                    except Exception:
                        logger.exception(
                            "SimpleX: error processing deferred voice message"
                        )
            return

        # Log other events at debug level
        if resp_type:
            logger.debug("SimpleX: unhandled event type: %s", resp_type)

    async def _handle_chat_item(self, chat_item: dict) -> None:
        """Process a single chat item from a newChatItems event."""
        chat_info = chat_item.get("chatInfo", {})
        chat_item_data = chat_item.get("chatItem", {})

        # Determine chat type and IDs
        chat_type = chat_info.get("type", "")

        # Extract message content
        meta = chat_item_data.get("meta", {})
        content = chat_item_data.get("content", {})
        msg_content = content.get("msgContent", {})

        # Filter out our own messages (sent items)
        item_direction = chat_item_data.get("chatDir", {})
        direction_type = item_direction.get("type", "") if isinstance(item_direction, dict) else ""
        if direction_type in ("directSnd", "groupSnd"):
            return  # Skip messages we sent

        # Only process received message content
        content_type = content.get("type", "") if isinstance(content, dict) else ""
        if content_type != "rcvMsgContent":
            return  # Not a received message (e.g. sndMsgContent, rcvDeleted, etc.)

        # Get text content
        text = ""
        msg_type_str = msg_content.get("type", "") if isinstance(msg_content, dict) else ""
        if msg_type_str in ("text", "file", "image", "voice", "link", "video"):
            text = msg_content.get("text", "")

        if not text and msg_type_str not in ("image", "file", "voice"):
            return  # No text content and not a media message

        # Extract sender info based on chat type
        sender_id = ""
        sender_name = ""
        chat_id = ""
        is_group = False

        if chat_type == "direct":
            contact = chat_info.get("contact", {})
            sender_id = str(contact.get("contactId", ""))
            sender_name = contact.get("localDisplayName", "") or contact.get("profile", {}).get("displayName", "")
            chat_id = sender_id
        elif chat_type == "group":
            group_info = chat_info.get("groupInfo", {})
            group_id = str(group_info.get("groupId", ""))
            chat_id = f"group:{group_id}"
            is_group = True

            # Extract sender from chatDir
            member = item_direction.get("groupMember", {})
            sender_id = str(member.get("memberId", ""))
            sender_name = member.get("localDisplayName", "") or member.get("memberProfile", {}).get("displayName", "")

            # Group filtering
            if self.group_allow_from:
                if "*" not in self.group_allow_from and group_id not in self.group_allow_from:
                    logger.debug(
                        "SimpleX: group %s not in allowlist",
                        _redact_id(group_id),
                    )
                    return
            else:
                # No group allowlist → groups disabled by default
                logger.debug("SimpleX: ignoring group message (no SIMPLEX_GROUP_ALLOWED)")
                return
        else:
            logger.debug("SimpleX: unhandled chat type: %s", chat_type)
            return

        if not sender_id:
            logger.debug("SimpleX: ignoring message with no sender")
            return

        # Process file/image attachments
        # File info is at chatItem.chatItem.file (sibling of meta, content, chatDir)
        media_urls = []
        media_types = []
        file_info = chat_item_data.get("file")

        if file_info and isinstance(file_info, dict):
            file_source = file_info.get("fileSource", {})
            file_path = file_source.get("filePath") if isinstance(file_source, dict) else None
            file_name = file_info.get("fileName", "")
            file_id = file_info.get("fileId")

            # Determine extension from path or filename
            ext = ""
            if file_path:
                ext = Path(file_path).suffix.lower()
            if not ext and file_name:
                ext = Path(file_name).suffix.lower()

            if not file_path and _is_audio_ext(ext) and file_id is not None:
                # File transfer not yet complete — accept and wait for rcvFileComplete.
                # Use fire-and-forget because simplex-chat never sends a corr-id reply
                # for /freceive; waiting for one would block the event loop for 30 s.
                logger.info(
                    "SimpleX: voice message file %d not yet received, accepting transfer",
                    file_id,
                )
                self._pending_file_transfers[file_id] = chat_item
                await self._send_fire_and_forget(f"/freceive {file_id}")
                return

            if file_path:
                ext = Path(file_path).suffix.lower() or (Path(file_name).suffix.lower() if file_name else "")
                if _is_image_ext(ext):
                    media_urls.append(file_path)
                    media_types.append(f"image/{ext.lstrip('.')}")
                elif _is_audio_ext(ext):
                    media_urls.append(file_path)
                    media_types.append(f"audio/{ext.lstrip('.')}")
                else:
                    media_urls.append(file_path)
                    media_types.append("application/octet-stream")

        # Build session source
        chat_name = sender_name
        if is_group:
            group_info = chat_info.get("groupInfo", {})
            chat_name = group_info.get("localDisplayName", "") or group_info.get("groupProfile", {}).get("displayName", chat_id)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type="group" if is_group else "dm",
            user_id=sender_id,
            user_name=sender_name or sender_id,
        )

        # Determine message type
        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO

        # Parse timestamp
        item_ts = meta.get("itemTs") or meta.get("createdAt", "")
        try:
            if item_ts:
                timestamp = datetime.fromisoformat(item_ts.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(tz=timezone.utc)
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        # Build and dispatch event
        msg_event = MessageEvent(
            source=source,
            text=text or "",
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
        )

        logger.debug(
            "SimpleX: message from %s in %s: %s",
            _redact_id(sender_id),
            chat_id[:20],
            (text or "")[:50],
        )

        await self.handle_message(msg_event)

    # ------------------------------------------------------------------
    # Command Interface
    # ------------------------------------------------------------------

    async def _send_command(self, command: str, timeout: float = 30.0) -> Optional[dict]:
        """Send a command to simplex-chat via WebSocket and optionally wait for response."""
        ws = self._ws
        if not ws:
            logger.warning("SimpleX: command sent but WebSocket not connected")
            return None

        self._corr_counter += 1
        corr_id = f"hermes_{self._corr_counter}_{int(time.time() * 1000)}"

        payload = json.dumps({
            "corrId": corr_id,
            "cmd": command,
        })

        # Create a future to receive the correlated response
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
        """Send a command to simplex-chat without waiting for a correlated response.

        Use this for commands that simplex-chat never sends a corrId reply for,
        such as /freceive.  Waiting for a corr-id response on these commands
        would block the event loop for the full timeout period.
        """
        ws = self._ws
        if not ws:
            logger.warning(
                "SimpleX: fire-and-forget command sent but WebSocket not connected"
            )
            return

        self._corr_counter += 1
        corr_id = f"hermes_{self._corr_counter}_{int(time.time() * 1000)}"
        payload = json.dumps({"corrId": corr_id, "cmd": command})
        try:
            await ws.send(payload)
        except Exception as e:
            logger.warning(
                "SimpleX: fire-and-forget send failed: %s — %s", command[:50], e
            )

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
        """Send a text message.

        If *content* contains ``MEDIA:<path>`` tags (embedded by TTS/audio
        tools to signal file attachments), they are stripped from the text
        and sent as native voice notes via :meth:`send_voice`.
        """
        # Extract and strip MEDIA:<path> tags before sending as text.
        _voice_exts = {".ogg", ".mp3", ".wav", ".m4a", ".opus"}
        media_paths = re.findall(r'MEDIA:(\S+)', content)
        if media_paths:
            content = re.sub(r'MEDIA:\S+', '', content).strip()

        text_result = SendResult(success=True)
        if content:
            if chat_id.startswith("group:"):
                group_id = chat_id[6:]
                # Use the /_send structured API with numeric group ID (#<id>) to
                # avoid ambiguity when multiple groups share the same display name.
                # The plain '#<name>' chat command looks up by display name and
                # fails if the name is not unique or matches the wrong group.
                # Use json.dumps for the full payload to correctly escape newlines,
                # backslashes, and other special characters in the message text.
                composed = json.dumps([{"msgContent": {"type": "text", "text": content}}])
                command = f"/_send #{group_id} json {composed}"
            else:
                command = f"@{chat_id} {content}"

            # SimpleX CLI uses @ prefix for DMs and # prefix for groups:
            #   @<contactId> <message>  or  #<groupId> <message>
            # The structured API form is:
            #   /_send @<contactId> json [{"msgContent":{"type":"text","text":"..."}}]
            # The simpler chat command form also works for plain text.

            result = await self._send_command(command)

            if result is not None:
                result_type = result.get("type", "")
                if result_type in ("newChatItems", "newChatItem"):
                    text_result = SendResult(success=True)
                elif "error" in str(result_type).lower():
                    text_result = SendResult(
                        success=False,
                        error=f"SimpleX error: {result.get('chatError', result)}",
                    )
                # else: unrecognised response type — treat as success
            elif not self._ws:
                # No response and no WebSocket — connection is gone
                text_result = SendResult(success=False, error="WebSocket not connected")

        if not text_result.success:
            return text_result

        # Send any MEDIA attachments extracted at the top of this method.
        for path in media_paths:
            is_voice = os.path.splitext(path)[1].lower() in _voice_exts
            if is_voice:
                media_result = await self.send_voice(chat_id, path)
            else:
                media_result = await self.send_document(chat_id, path)
            if not media_result.success:
                return media_result

        return text_result

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image. Supports file:// URLs and http(s):// URLs."""
        from urllib.parse import unquote

        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            # Download remote image to cache
            try:
                from gateway.platforms.base import cache_image_from_url
                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("SimpleX: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))

        if not file_path or not Path(file_path).exists():
            return SendResult(success=False, error="Image file not found")

        # Send file via simplex-chat command
        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/f #{group_id} {file_path}"
        else:
            command = f"/f @{chat_id} {file_path}"

        result = await self._send_command(command)

        # Send caption as a separate message if provided
        if caption and result is not None:
            await self.send(chat_id, caption)

        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Failed to send image")

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

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/f #{group_id} {file_path}"
        else:
            command = f"/f @{chat_id} {file_path}"

        result = await self._send_command(command)

        if caption and result is not None:
            await self.send(chat_id, caption)

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
        """Send an audio file as a SimpleX voice message (plays inline).

        SimpleX differentiates between a generic file attachment (type:"file")
        and an inline voice note (type:"voice").  Using the plain /f command
        sends a downloadable file.  To get the voice-note player the client
        must receive a /_send command whose msgContent.type is "voice".
        """
        if not Path(audio_path).exists():
            return SendResult(success=False, error="Voice file not found")

        caption_text = caption or ""
        composed = json.dumps(
            [
                {
                    "msgContent": {
                        "type": "voice",
                        "text": caption_text,
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

        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
        }
