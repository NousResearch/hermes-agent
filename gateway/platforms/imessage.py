"""
iMessage platform adapter for Hermes gateway.

Uses the `imsg` CLI (https://github.com/steipete/imsg) to send and receive
iMessages via the local Messages.app database and AppleScript.

Requirements:
  - macOS 14+
  - `imsg` CLI installed (brew install steipete/tap/imsg)
  - Full Disk Access for the Hermes process
  - Messages.app signed into iCloud
"""

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_document_from_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 8000
WATCH_RESTART_DELAY_INITIAL = 2.0
WATCH_RESTART_DELAY_MAX = 60.0
TYPING_INTERVAL = 5.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".heic", ".webp", ".tiff", ".bmp"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".aac", ".ogg", ".wav", ".caf"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class iMessageAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for iMessage via the imsg CLI."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.IMESSAGE)
        self._imsg_path: str = config.extra.get("imsg_path", "imsg")
        self._allowed_chats: Optional[List[str]] = _parse_comma_list(
            os.getenv("IMESSAGE_ALLOWED_CHATS", "")
        )
        self._watch_proc: Optional[asyncio.subprocess.Process] = None
        self._watch_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        self._last_sent_guids: Dict[str, float] = {}  # guid → timestamp for echo filtering
        self._last_watch_activity: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._imsg_path, "--help",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                logger.error("imsg CLI not working (exit %d)", proc.returncode)
                self._set_fatal_error("imsg_not_found", "imsg CLI not working", retryable=False)
                return False
        except FileNotFoundError:
            logger.error("imsg CLI not found at '%s'", self._imsg_path)
            self._set_fatal_error("imsg_not_found", f"imsg not found at {self._imsg_path}", retryable=False)
            return False

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info("[iMessage] Connected — watching for incoming messages")
        return True

    async def disconnect(self) -> None:
        self._running = False

        if self._watch_proc and self._watch_proc.returncode is None:
            self._watch_proc.terminate()
            try:
                await asyncio.wait_for(self._watch_proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._watch_proc.kill()

        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        logger.info("[iMessage] Disconnected")

    # ------------------------------------------------------------------
    # Watch loop — stream incoming messages
    # ------------------------------------------------------------------

    async def _watch_loop(self) -> None:
        """Spawn `imsg watch --json --attachments` and parse output lines."""
        delay = WATCH_RESTART_DELAY_INITIAL

        while self._running:
            try:
                self._watch_proc = await asyncio.create_subprocess_exec(
                    self._imsg_path, "watch", "--json", "--attachments",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._last_watch_activity = time.monotonic()
                delay = WATCH_RESTART_DELAY_INITIAL  # reset on successful start

                logger.info("[iMessage] Watch process started (PID %d)", self._watch_proc.pid)

                # Check for immediate failure (permissions, missing DB, etc.)
                # by reading the first line with a short timeout
                first_line_error = None
                try:
                    first_line = await asyncio.wait_for(
                        self._watch_proc.stdout.readline(), timeout=2.0,
                    )
                    if first_line:
                        decoded = first_line.decode("utf-8", errors="replace").strip()
                        try:
                            msg = json.loads(decoded)
                            # Valid JSON message — process it normally
                            await self._handle_message(msg)
                        except json.JSONDecodeError:
                            # Not JSON — likely an error message
                            if self._watch_proc.returncode is not None:
                                first_line_error = decoded
                except asyncio.TimeoutError:
                    pass  # No output yet — that's normal, watch is waiting

                if first_line_error:
                    logger.error("[iMessage] Watch failed on start: %s", first_line_error)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, WATCH_RESTART_DELAY_MAX)
                    continue

                async for line in self._watch_proc.stdout:
                    if not self._running:
                        break

                    self._last_watch_activity = time.monotonic()
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        continue

                    try:
                        msg = json.loads(decoded)
                    except json.JSONDecodeError:
                        logger.debug("[iMessage] Non-JSON line from watch: %s", decoded[:200])
                        continue

                    await self._handle_message(msg)

                # Process exited — drain stderr for diagnostics
                rc = await self._watch_proc.wait()
                stderr = ""
                if self._watch_proc.stderr:
                    try:
                        raw = await asyncio.wait_for(self._watch_proc.stderr.read(), timeout=2.0)
                        stderr = raw.decode("utf-8", errors="replace").strip()
                    except (asyncio.TimeoutError, Exception):
                        pass
                if rc != 0:
                    logger.warning("[iMessage] Watch exited with code %d: %s", rc, stderr[:500] if stderr else "(no stderr)")

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[iMessage] Watch loop error: %s", exc)

            if not self._running:
                break

            logger.info("[iMessage] Reconnecting watch in %.1fs", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, WATCH_RESTART_DELAY_MAX)

    # ------------------------------------------------------------------
    # Inbound message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict) -> None:
        """Process a single JSON message from imsg watch."""
        # Skip our own messages
        if msg.get("is_from_me"):
            return

        text = msg.get("text", "")
        sender = msg.get("sender", "")
        chat_id = str(msg.get("chat_id", ""))
        message_id = msg.get("guid", "")
        created_at = msg.get("created_at", "")

        if not chat_id or not sender:
            return

        # Allowlist filtering
        if self._allowed_chats and sender not in self._allowed_chats and chat_id not in self._allowed_chats:
            return

        # Handle attachments
        media_urls: List[str] = []
        media_types: List[str] = []
        attachments = msg.get("attachments", [])
        for att in attachments:
            filepath = att.get("path") or att.get("filename", "")
            mime = att.get("mime_type", "")
            if filepath and Path(filepath).exists():
                media_urls.append(filepath)
                media_types.append(mime or _guess_mime(filepath))

        # Determine message type
        message_type = MessageType.TEXT
        if media_urls:
            ext = Path(media_urls[0]).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                message_type = MessageType.PHOTO
            elif ext in AUDIO_EXTENSIONS:
                message_type = MessageType.VOICE
            elif ext in VIDEO_EXTENSIONS:
                message_type = MessageType.VIDEO
            else:
                message_type = MessageType.DOCUMENT

        # Parse timestamp
        timestamp = datetime.now(timezone.utc)
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Build source
        chat_type = "group" if _is_group_chat(msg) else "dm"
        source = self.build_source(
            chat_id=chat_id,
            chat_name=msg.get("chat_name") or sender,
            chat_type=chat_type,
            user_id=sender,
            user_name=msg.get("sender_name") or sender,
        )

        event = MessageEvent(
            text=text or "",
            message_type=message_type,
            source=source,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
        )

        logger.info(
            "[iMessage] Received from %s in chat %s: %s",
            _redact_id(sender), chat_id, (text or "<media>")[:80],
        )

        await self.handle_message(event)

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
        await self._stop_typing_indicator(chat_id)

        if not content.strip():
            return SendResult(success=True)

        # Truncate if needed
        if len(content) > MAX_MESSAGE_LENGTH:
            content = content[:MAX_MESSAGE_LENGTH] + "\n… (truncated)"

        try:
            args = [self._imsg_path, "send", "--chat-id", str(chat_id), "--text", content, "--json"]
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            if proc.returncode != 0:
                error = stderr.decode("utf-8", errors="replace").strip()
                logger.error("[iMessage] Send failed (exit %d): %s", proc.returncode, error[:200])
                return SendResult(success=False, error=error, retryable=True)

            # Parse response for message ID
            msg_id = None
            try:
                result = json.loads(stdout.decode("utf-8", errors="replace"))
                msg_id = result.get("guid") or result.get("id")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            return SendResult(success=True, message_id=str(msg_id) if msg_id else None)

        except asyncio.TimeoutError:
            logger.error("[iMessage] Send timed out for chat %s", chat_id)
            return SendResult(success=False, error="Send timed out", retryable=True)
        except Exception as exc:
            logger.error("[iMessage] Send error: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict] = None) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._imsg_path, "typing", "--chat-id", str(chat_id),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except Exception:
            pass  # Typing indicators are best-effort

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        # Resolve to local file
        local_path = image_url
        if image_url.startswith(("http://", "https://")):
            from gateway.platforms.base import cache_image_from_url
            local_path = await cache_image_from_url(image_url)

        if not local_path or not Path(local_path).exists():
            # Fall back to sending URL as text
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id, text)

        return await self._send_attachment(chat_id, local_path, caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        if not Path(file_path).exists():
            return SendResult(success=False, error=f"File not found: {file_path}")
        return await self._send_attachment(chat_id, file_path, caption)

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        try:
            args = [self._imsg_path, "send", "--chat-id", str(chat_id), "--file", file_path, "--json"]
            if caption:
                args.extend(["--text", caption])

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)

            if proc.returncode != 0:
                error = stderr.decode("utf-8", errors="replace").strip()
                logger.error("[iMessage] Attachment send failed: %s", error[:200])
                return SendResult(success=False, error=error, retryable=True)

            return SendResult(success=True)

        except asyncio.TimeoutError:
            return SendResult(success=False, error="Attachment send timed out", retryable=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Try to get chat details from imsg
        try:
            proc = await asyncio.create_subprocess_exec(
                self._imsg_path, "chats", "--json", "--limit", "50",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            for line in stdout.decode("utf-8", errors="replace").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    chat = json.loads(line)
                    if str(chat.get("id")) == str(chat_id):
                        return {
                            "name": chat.get("name") or chat.get("identifier", str(chat_id)),
                            "type": "group" if "," in chat.get("identifier", "") else "dm",
                            "chat_id": str(chat_id),
                        }
                except json.JSONDecodeError:
                    continue

        except Exception:
            pass

        return {"name": str(chat_id), "type": "dm", "chat_id": str(chat_id)}

    # ------------------------------------------------------------------
    # Typing indicator management
    # ------------------------------------------------------------------

    async def _start_typing_indicator(self, chat_id: str) -> None:
        async def _loop():
            while True:
                await self.send_typing(chat_id)
                await asyncio.sleep(TYPING_INTERVAL)

        if chat_id in self._typing_tasks:
            self._typing_tasks[chat_id].cancel()

        self._typing_tasks[chat_id] = asyncio.create_task(_loop())

    async def _stop_typing_indicator(self, chat_id: str) -> None:
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()

        # Send stop signal
        try:
            proc = await asyncio.create_subprocess_exec(
                self._imsg_path, "typing", "--chat-id", str(chat_id), "--stop", "true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_group_chat(msg: dict) -> bool:
    """Heuristic: group chats typically have a chat_identifier with 'chat' in it."""
    identifier = msg.get("chat_identifier", "")
    return ";+;" in identifier or "chat" in identifier.lower()


def _redact_id(identifier: str) -> str:
    """Mask phone numbers / emails for logging."""
    if "@" in identifier:
        parts = identifier.split("@")
        return f"{parts[0][:2]}***@{parts[1]}"
    if len(identifier) > 6:
        return f"{identifier[:3]}***{identifier[-2:]}"
    return "***"


def _guess_mime(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".heic": "image/heic", ".webp": "image/webp",
        ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
        ".pdf": "application/pdf", ".zip": "application/zip",
    }
    return mime_map.get(ext, "application/octet-stream")


def _parse_comma_list(value: str) -> Optional[List[str]]:
    if not value.strip():
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


def check_imessage_requirements() -> bool:
    """Verify imsg CLI is available."""
    enabled = os.getenv("IMESSAGE_ENABLED", "").lower() in ("true", "1", "yes")
    if not enabled:
        return False

    if not shutil.which("imsg"):
        logger.warning("iMessage: imsg CLI not found in PATH")
        return False

    return True
