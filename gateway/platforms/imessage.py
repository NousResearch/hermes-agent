"""iMessage platform adapter.

Uses the `imsg` CLI tool (installed at /opt/homebrew/bin/imsg) to send and
receive iMessages.  Inbound messages arrive via `imsg watch --json --attachments`
which streams one JSON object per line.  Outbound messages use `imsg send`.

No token or API key is required -- the CLI interfaces directly with the local
macOS Messages database.

Requires:
  - imsg CLI installed at /opt/homebrew/bin/imsg
  - IMESSAGE_ALLOWED_USERS env var (comma-separated phone numbers / emails)
  - Optionally IMESSAGE_HOME_CHANNEL (a chat_id rowid number)
"""

import asyncio
import json
import logging
import os
import shutil
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
MAX_MESSAGE_LENGTH = 10000
IMSG_BIN = "/opt/homebrew/bin/imsg"
WATCH_RESTART_DELAY_INITIAL = 2.0
WATCH_RESTART_DELAY_MAX = 60.0

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".tiff"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".caf"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}


def _parse_comma_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def check_imessage_requirements() -> bool:
    """Check if the imsg CLI is available."""
    return shutil.which("imsg") is not None or Path(IMSG_BIN).exists()


def _imsg_path() -> str:
    if Path(IMSG_BIN).exists():
        return IMSG_BIN
    found = shutil.which("imsg")
    return found or IMSG_BIN


class IMessageAdapter(BasePlatformAdapter):
    """iMessage adapter using the imsg CLI tool."""

    platform = Platform.IMESSAGE

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.IMESSAGE)
        self._watch_proc: Optional[asyncio.subprocess.Process] = None
        self._watch_popen = None
        self._watch_task: Optional[asyncio.Task] = None
        self._running = False
        allowed_str = os.getenv("IMESSAGE_ALLOWED_USERS", "")
        self.allowed_users: set = set(_parse_comma_list(allowed_str))
        logger.info("iMessage adapter initialized: allowed_users=%d", len(self.allowed_users))

    async def connect(self) -> bool:
        imsg = _imsg_path()
        if not Path(imsg).exists() and not shutil.which("imsg"):
            logger.error("iMessage: imsg CLI not found at %s", imsg)
            return False
        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        self._watch_task.add_done_callback(self._watch_task_done)
        logger.info("iMessage: connected (watching for messages)")
        return True

    def _watch_task_done(self, task: asyncio.Task) -> None:
        """Log if the watch task exits unexpectedly."""
        try:
            exc = task.exception()
            if exc:
                logger.error("iMessage: watch task failed with exception: %s", exc, exc_info=exc)
        except asyncio.CancelledError:
            logger.debug("iMessage: watch task cancelled")
        except Exception as e:
            logger.error("iMessage: watch task done callback error: %s", e)

    async def disconnect(self) -> None:
        self._running = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        await self._kill_watch_proc()
        logger.info("iMessage: disconnected")

    async def _kill_watch_proc(self) -> None:
        # Handle asyncio subprocess (legacy)
        proc = self._watch_proc
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        self._watch_proc = None
        # Handle Popen subprocess
        popen = getattr(self, "_watch_popen", None)
        if popen and popen.poll() is None:
            try:
                popen.terminate()
                popen.wait(timeout=5)
            except Exception:
                try:
                    popen.kill()
                except Exception:
                    pass
        self._watch_popen = None

    async def _watch_loop(self) -> None:
        """Poll for new iMessages using `imsg history`.

        Uses a polling approach instead of `imsg watch` because the
        watch command requires filesystem event access (Full Disk Access)
        which may not be available to launchd-managed processes.
        """
        import subprocess as _sp

        POLL_INTERVAL = 3.0  # seconds between polls
        imsg = _imsg_path()

        # Get the latest message rowid to start from
        last_seen_id = await self._get_latest_rowid(imsg)
        logger.info("iMessage: polling started, last_seen_id=%s", last_seen_id)

        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                if not self._running:
                    break

                # Get all chats to check for new messages
                new_messages = await self._poll_new_messages(imsg, last_seen_id)

                for msg in new_messages:
                    msg_id = msg.get("id", 0)
                    if msg_id > last_seen_id:
                        last_seen_id = msg_id
                    await self._handle_message_json(msg)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("iMessage: poll error")
                await asyncio.sleep(5.0)

        logger.info("iMessage: polling stopped")

    async def _get_latest_rowid(self, imsg: str) -> int:
        """Get the latest message rowid across all chats."""
        import subprocess as _sp
        try:
            proc = _sp.Popen(
                [imsg, "chats", "--json", "--limit", "20"],
                stdout=_sp.PIPE, stderr=_sp.PIPE, stdin=_sp.DEVNULL,
            )
            stdout, _ = proc.communicate(timeout=15)
            if proc.returncode != 0:
                return 0

            max_id = 0
            for line in stdout.decode("utf-8", errors="replace").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    chat = json.loads(line)
                    chat_id = str(chat.get("id", ""))
                    # Get latest message from this chat
                    hp = _sp.Popen(
                        [imsg, "history", "--chat-id", chat_id, "--json", "--limit", "1"],
                        stdout=_sp.PIPE, stderr=_sp.PIPE, stdin=_sp.DEVNULL,
                    )
                    h_out, _ = hp.communicate(timeout=10)
                    if hp.returncode == 0:
                        for hline in h_out.decode("utf-8", errors="replace").strip().split("\n"):
                            hline = hline.strip()
                            if hline:
                                try:
                                    m = json.loads(hline)
                                    mid = m.get("id", 0)
                                    if mid > max_id:
                                        max_id = mid
                                except json.JSONDecodeError:
                                    pass
                except (json.JSONDecodeError, Exception):
                    continue
            return max_id
        except Exception as e:
            logger.warning("iMessage: failed to get latest rowid: %s", e)
            return 0

    async def _poll_new_messages(self, imsg: str, last_seen_id: int) -> list:
        """Poll all chats for messages newer than last_seen_id."""
        import subprocess as _sp
        new_msgs = []
        try:
            # Get recent chats
            proc = _sp.Popen(
                [imsg, "chats", "--json", "--limit", "20"],
                stdout=_sp.PIPE, stderr=_sp.PIPE, stdin=_sp.DEVNULL,
            )
            stdout, _ = proc.communicate(timeout=15)
            if proc.returncode != 0:
                return []

            for line in stdout.decode("utf-8", errors="replace").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    chat = json.loads(line)
                    chat_id = str(chat.get("id", ""))
                except (json.JSONDecodeError, Exception):
                    continue

                # Get recent messages from this chat
                hp = _sp.Popen(
                    [imsg, "history", "--chat-id", chat_id, "--json",
                     "--attachments", "--limit", "10"],
                    stdout=_sp.PIPE, stderr=_sp.PIPE, stdin=_sp.DEVNULL,
                )
                h_out, _ = hp.communicate(timeout=10)
                if hp.returncode != 0:
                    continue

                for hline in h_out.decode("utf-8", errors="replace").strip().split("\n"):
                    hline = hline.strip()
                    if not hline:
                        continue
                    try:
                        msg = json.loads(hline)
                        msg_id = msg.get("id", 0)
                        if msg_id > last_seen_id:
                            new_msgs.append(msg)
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.warning("iMessage: poll error: %s", e)

        # Sort by id to process in order
        new_msgs.sort(key=lambda m: m.get("id", 0))
        return new_msgs

    async def _handle_message_json(self, data: dict) -> None:
        if data.get("is_from_me"):
            return
        sender = data.get("sender", "")
        text = data.get("text", "") or ""
        chat_id = str(data.get("chat_id", ""))
        msg_id = str(data.get("id", ""))

        if not sender or not chat_id:
            logger.debug("iMessage: ignoring message with no sender/chat_id")
            return

        attachments = data.get("attachments", [])
        media_urls: List[str] = []
        media_types: List[str] = []
        for att in attachments:
            if isinstance(att, str):
                att_path = att
            elif isinstance(att, dict):
                att_path = att.get("path") or att.get("file") or ""
            else:
                continue
            if att_path and Path(att_path).exists():
                media_urls.append(att_path)
                ext = Path(att_path).suffix.lower()
                if ext in _IMAGE_EXTS:
                    media_types.append(f"image/{ext.lstrip('.')}")
                elif ext in _AUDIO_EXTS:
                    media_types.append(f"audio/{ext.lstrip('.')}")
                elif ext in _VIDEO_EXTS:
                    media_types.append(f"video/{ext.lstrip('.')}")
                else:
                    media_types.append("application/octet-stream")

        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO

        created_at = data.get("created_at", "")
        try:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=sender,
            chat_type="dm",
            user_id=sender,
            user_name=sender,
        )

        event = MessageEvent(
            source=source,
            text=text,
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            message_id=msg_id,
            timestamp=timestamp,
        )

        logger.debug("iMessage: message from %s in chat %s: %s", sender, chat_id, text[:50])
        await self.handle_message(event)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        imsg = _imsg_path()
        if len(content) > MAX_MESSAGE_LENGTH:
            chunks = self.truncate_message(content, MAX_MESSAGE_LENGTH)
            last_result = SendResult(success=False, error="no chunks")
            for chunk in chunks:
                last_result = await self._send_single(imsg, chat_id, chunk)
                if not last_result.success:
                    return last_result
            return last_result
        return await self._send_single(imsg, chat_id, content)

    async def _send_single(self, imsg: str, chat_id: str, text: str) -> SendResult:
        try:
            cmd = [imsg, "send", "--chat-id", str(chat_id), "--text", text]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            if proc.returncode == 0:
                return SendResult(success=True)
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            logger.warning("iMessage: send failed (rc=%d): %s", proc.returncode, error_msg)
            return SendResult(success=False, error=error_msg or "imsg send failed")
        except asyncio.TimeoutError:
            return SendResult(success=False, error="imsg send timed out")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, **kwargs) -> SendResult:
        if image_url.startswith(("http://", "https://")):
            try:
                from gateway.platforms.base import cache_image_from_url
                image_url = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("iMessage: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))
        return await self._send_file(chat_id, image_url, caption)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None,
                            **kwargs) -> SendResult:
        return await self._send_file(chat_id, file_path, caption)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_file(chat_id, audio_path, caption)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_file(chat_id, image_path, caption)

    async def _send_file(self, chat_id: str, file_path: str,
                         caption: Optional[str] = None) -> SendResult:
        imsg = _imsg_path()
        if not Path(file_path).exists():
            return SendResult(success=False, error=f"File not found: {file_path}")
        try:
            cmd = [imsg, "send", "--chat-id", str(chat_id), "--file", file_path]
            if caption:
                cmd.extend(["--text", caption])
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            if proc.returncode == 0:
                return SendResult(success=True)
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            return SendResult(success=False, error=error_msg or "imsg send --file failed")
        except asyncio.TimeoutError:
            return SendResult(success=False, error="imsg send --file timed out")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        imsg = _imsg_path()
        try:
            proc = await asyncio.create_subprocess_exec(
                imsg, "chats", "--json", "--limit", "50",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            if proc.returncode == 0:
                for line in stdout.decode("utf-8", errors="replace").strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chat = json.loads(line)
                        if str(chat.get("id")) == str(chat_id):
                            return {
                                "name": chat.get("name") or chat.get("identifier", chat_id),
                                "type": "dm",
                                "chat_id": str(chat.get("id")),
                            }
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug("iMessage: get_chat_info failed: %s", e)
        return {"name": f"iMessage chat {chat_id}", "type": "dm", "chat_id": chat_id}
