"""iMessage platform adapter.

Listens for incoming iMessages via `imsg watch` subprocess and sends
replies via `imsg send`. Requires macOS and the `imsg` CLI tool
(brew install steipete/tap/imsg).

Architecture: subprocess-based listener, similar to the Signal adapter's
SSE streaming pattern with exponential backoff restart.
"""

import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
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
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 8000
WATCH_RETRY_DELAY_INITIAL = 2.0
WATCH_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0
HEALTH_CHECK_STALE_THRESHOLD = 120.0
CACHE_REFRESH_INTERVAL = 300.0  # 5 minutes
DEDUP_TTL = 300.0  # 5 minutes
DEDUP_MAX_SIZE = 2000
POLL_INTERVAL_DEFAULT = 3.0  # seconds between DB polls
POLL_COLLECT_TIMEOUT = 2.0  # seconds to wait for imsg watch --since-rowid output
AUTO_FALLBACK_THRESHOLD = 30.0  # seconds: switch from fsevents to poll if no output

# Phone number pattern for redaction
_PHONE_RE = re.compile(r"\+[1-9]\d{6,14}")
# Apple ID (email) pattern for redaction
_APPLEID_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_imessage_requirements() -> bool:
    """Check if iMessage adapter can run: macOS + imsg CLI available."""
    if sys.platform != "darwin":
        logger.debug("iMessage: not macOS (platform=%s)", sys.platform)
        return False
    if not shutil.which("imsg"):
        logger.debug("iMessage: imsg CLI not found in PATH")
        return False
    return True


def _redact_imessage_id(identifier: str) -> str:
    """Redact phone numbers and Apple IDs for logging."""
    if not identifier:
        return "<none>"
    # Phone number: +15551234567 -> +155****4567
    if identifier.startswith("+"):
        if len(identifier) <= 8:
            return identifier[:2] + "****" + identifier[-2:] if len(identifier) > 4 else "****"
        return identifier[:4] + "****" + identifier[-4:]
    # Apple ID email: user@example.com -> us****@example.com
    if "@" in identifier:
        local, domain = identifier.split("@", 1)
        if len(local) <= 2:
            return "****@" + domain
        return local[:2] + "****@" + domain
    return identifier


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class IMessageAdapter(BasePlatformAdapter):
    """iMessage gateway adapter using the imsg CLI."""

    platform = Platform.IMESSAGE
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.IMESSAGE)

        # Config
        self._watch_chat_ids: List[str] = config.extra.get("watch_chat_ids", [])

        # Watch mode: "fsevents" | "poll" | "auto"
        # auto = try fsevents, fall back to poll if no output detected
        self._watch_mode: str = config.extra.get("watch_mode", "auto")
        self._poll_interval: float = config.extra.get(
            "poll_interval", POLL_INTERVAL_DEFAULT
        )

        # Background tasks
        self._watch_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cache_refresh_task: Optional[asyncio.Task] = None

        # Subprocess
        self._watch_process: Optional[asyncio.subprocess.Process] = None
        self._running = False

        # Chat cache: maps numeric chat_id -> {identifier, name, service}
        self._chat_cache: Dict[int, dict] = {}

        # Dedup: guid -> timestamp
        self._seen_messages: Dict[str, float] = {}

        # Track own sends to filter echo-backs
        self._recent_sent_ids: set = set()

        # Health tracking
        self._last_message_time: float = time.monotonic()

        # Poll mode state: highest message rowid seen
        self._last_rowid: int = 0

    async def connect(self) -> bool:
        """Verify imsg available, load chat cache, start background tasks."""
        if not check_imessage_requirements():
            self._set_fatal_error(
                "IMESSAGE_REQUIREMENTS",
                "iMessage requires macOS and the imsg CLI. "
                "Install: brew install steipete/tap/imsg",
                retryable=False,
            )
            return False

        # Health check: verify imsg can run and has permissions
        try:
            proc = await asyncio.create_subprocess_exec(
                "imsg", "chats", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode != 0:
                err = (stderr.decode("utf-8", errors="replace").strip()
                       or stdout.decode("utf-8", errors="replace").strip())
                msg = (
                    f"imsg chats failed (exit {proc.returncode}): {err}. "
                    "Ensure Full Disk Access is granted to your terminal."
                )
                self._set_fatal_error("IMESSAGE_PERMISSION", msg, retryable=True)
                logger.error("iMessage: %s", msg)
                return False
            self._parse_chat_list(stdout.decode("utf-8", errors="replace"))
        except asyncio.TimeoutError:
            msg = "imsg chats timed out — check Full Disk Access permissions"
            self._set_fatal_error("IMESSAGE_TIMEOUT", msg, retryable=True)
            logger.error("iMessage: %s", msg)
            return False
        except Exception as e:
            msg = f"imsg chats failed: {e}"
            self._set_fatal_error("IMESSAGE_ERROR", msg, retryable=True)
            logger.error("iMessage: %s", msg)
            return False

        self._running = True

        # Seed last_rowid so poll mode only sees new messages
        await self._seed_last_rowid()

        # Choose listener based on watch_mode
        if self._watch_mode == "poll":
            self._watch_task = asyncio.ensure_future(self._poll_listener())
            mode_label = "poll"
        elif self._watch_mode == "fsevents":
            self._watch_task = asyncio.ensure_future(self._watch_listener())
            mode_label = "fsevents"
        else:  # auto
            self._watch_task = asyncio.ensure_future(self._auto_watch_listener())
            mode_label = "auto"

        self._health_monitor_task = asyncio.ensure_future(self._health_monitor())
        self._cache_refresh_task = asyncio.ensure_future(self._cache_refresh_loop())

        logger.info(
            "iMessage: connected — watching %s chats (%s mode), cache has %d entries",
            "all" if not self._watch_chat_ids else len(self._watch_chat_ids),
            mode_label,
            len(self._chat_cache),
        )
        return True

    async def disconnect(self) -> None:
        """Stop all background tasks and terminate subprocess."""
        self._running = False

        # Cancel tasks
        for task in (self._watch_task, self._health_monitor_task, self._cache_refresh_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Terminate subprocess
        if self._watch_process and self._watch_process.returncode is None:
            try:
                self._watch_process.terminate()
                try:
                    await asyncio.wait_for(self._watch_process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._watch_process.kill()
                    await self._watch_process.wait()
            except ProcessLookupError:
                pass

        self._watch_task = None
        self._health_monitor_task = None
        self._cache_refresh_task = None
        self._watch_process = None
        logger.info("iMessage: disconnected")

    # -----------------------------------------------------------------------
    # Inbound: watch listener
    # -----------------------------------------------------------------------

    async def _watch_listener(self) -> None:
        """Core inbound loop: run `imsg watch` and process JSON lines."""
        retry_delay = WATCH_RETRY_DELAY_INITIAL

        while self._running:
            try:
                cmd = ["imsg", "watch", "--json", "--attachments"]
                self._watch_process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                logger.info("iMessage: started imsg watch (pid=%s)", self._watch_process.pid)
                retry_delay = WATCH_RETRY_DELAY_INITIAL  # Reset on successful start

                while self._running:
                    try:
                        line = await asyncio.wait_for(
                            self._watch_process.stdout.readline(),
                            timeout=HEALTH_CHECK_STALE_THRESHOLD,
                        )
                    except asyncio.TimeoutError:
                        # No output for a while — process may be stuck
                        logger.debug("iMessage: no output for %.0fs, continuing", HEALTH_CHECK_STALE_THRESHOLD)
                        continue

                    if not line:
                        # EOF — process exited
                        break

                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue

                    try:
                        data = json.loads(line_str)
                        self._last_message_time = time.monotonic()
                        await self._handle_message_data(data)
                    except json.JSONDecodeError:
                        logger.debug("iMessage: non-JSON line: %s", line_str[:200])
                    except Exception as e:
                        logger.warning("iMessage: error handling message: %s", e)

                # Process exited
                rc = self._watch_process.returncode
                if rc is None:
                    try:
                        rc = await self._watch_process.wait()
                    except Exception:
                        rc = -1
                logger.warning("iMessage: imsg watch exited (code=%s)", rc)

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("iMessage: watch listener error: %s", e)

            if not self._running:
                return

            # Exponential backoff with jitter
            jitter = retry_delay * 0.2 * random.random()
            wait = retry_delay + jitter
            logger.info("iMessage: reconnecting in %.1fs", wait)
            await asyncio.sleep(wait)
            retry_delay = min(retry_delay * 2, WATCH_RETRY_DELAY_MAX)

    # -----------------------------------------------------------------------
    # Inbound: poll listener (fallback for broken FSEvents)
    # -----------------------------------------------------------------------

    async def _seed_last_rowid(self) -> None:
        """Get the current highest message rowid so poll mode starts from now.

        Uses the most recent chat's latest message as the high-water mark.
        imsg history requires --chat-id, so we first find the most recent chat.
        """
        try:
            # Find most recent chat
            if not self._chat_cache:
                return
            # Pick the first chat (we don't know which is newest without sorting,
            # but any chat gives us a reasonable high-water mark)
            chat_id = next(iter(self._chat_cache))

            proc = await asyncio.create_subprocess_exec(
                "imsg", "history", "--json", "--limit", "1",
                "--chat-id", str(chat_id),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                line = stdout.decode("utf-8", errors="replace").strip().split("\n")[0]
                if line:
                    data = json.loads(line)
                    self._last_rowid = data.get("id", 0)
                    logger.debug("iMessage: seeded last_rowid=%d", self._last_rowid)
        except Exception as e:
            logger.debug("iMessage: could not seed last_rowid: %s", e)

    async def _poll_once(self) -> int:
        """Run a short-lived `imsg watch --since-rowid` to fetch new messages.

        Returns the number of messages processed.
        """
        count = 0
        try:
            cmd = ["imsg", "watch", "--json", "--attachments",
                   "--since-rowid", str(self._last_rowid)]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Read the catchup dump (messages since last_rowid), then kill
            try:
                lines: List[str] = []
                while True:
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=POLL_COLLECT_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        break  # No more catchup output — done
                    if not line:
                        break  # EOF
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if line_str:
                        lines.append(line_str)
            finally:
                # Always kill the subprocess — we only want the catchup dump
                try:
                    proc.kill()
                    await proc.wait()
                except (ProcessLookupError, OSError):
                    pass

            # Process collected messages
            for line_str in lines:
                try:
                    data = json.loads(line_str)
                    rowid = data.get("id", 0)
                    if rowid > self._last_rowid:
                        self._last_rowid = rowid
                    self._last_message_time = time.monotonic()
                    await self._handle_message_data(data)
                    count += 1
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.warning("iMessage: poll error handling message: %s", e)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("iMessage: poll error: %s", e)

        return count

    async def _poll_listener(self) -> None:
        """Poll-based inbound loop: periodic short-lived imsg watch calls."""
        logger.info(
            "iMessage: poll mode — checking every %.1fs (since rowid %d)",
            self._poll_interval, self._last_rowid,
        )
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._running:
                    return
                await self._poll_once()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("iMessage: poll listener error: %s", e)

    async def _auto_watch_listener(self) -> None:
        """Auto mode: start with FSEvents, fall back to poll if no output."""
        logger.info("iMessage: auto mode — trying FSEvents first")

        # Start FSEvents watch
        cmd = ["imsg", "watch", "--json", "--attachments"]
        self._watch_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("iMessage: started imsg watch (pid=%s)", self._watch_process.pid)

        # Wait for first output or timeout
        got_output = False
        deadline = time.monotonic() + AUTO_FALLBACK_THRESHOLD

        while self._running and time.monotonic() < deadline:
            try:
                line = await asyncio.wait_for(
                    self._watch_process.stdout.readline(),
                    timeout=min(5.0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            if not line:
                break  # EOF — process exited

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
                self._last_message_time = time.monotonic()
                await self._handle_message_data(data)
                got_output = True
                break  # FSEvents works — switch to full watch mode
            except json.JSONDecodeError:
                pass

        if got_output:
            # FSEvents works — continue with the normal watch listener loop
            logger.info("iMessage: FSEvents working — staying in watch mode")
            # Kill this process and hand off to the full watch listener
            try:
                self._watch_process.kill()
                await self._watch_process.wait()
            except (ProcessLookupError, OSError):
                pass
            await self._watch_listener()
        else:
            # No output — fall back to poll mode
            logger.warning(
                "iMessage: no FSEvents output after %.0fs — switching to poll mode "
                "(every %.1fs). Set IMESSAGE_WATCH_MODE=poll to skip this probe.",
                AUTO_FALLBACK_THRESHOLD, self._poll_interval,
            )
            try:
                self._watch_process.kill()
                await self._watch_process.wait()
            except (ProcessLookupError, OSError):
                pass
            self._watch_process = None
            await self._poll_listener()

    # -----------------------------------------------------------------------
    # Inbound: message handling
    # -----------------------------------------------------------------------

    async def _handle_message_data(self, data: dict) -> None:
        """Process a single JSON message from imsg watch."""
        # Filter self-messages
        if data.get("is_from_me"):
            return

        # Dedup by guid
        guid = data.get("guid", "")
        if guid and self._is_duplicate(guid):
            return

        # Filter by configured watch_chat_ids
        chat_id = data.get("chat_id")
        if self._watch_chat_ids and str(chat_id) not in self._watch_chat_ids:
            return

        # Resolve chat metadata
        chat_meta = self._chat_cache.get(int(chat_id)) if chat_id is not None else None
        if chat_meta is None and chat_id is not None:
            # Cache miss — refresh and retry
            await self._refresh_chat_cache()
            chat_meta = self._chat_cache.get(int(chat_id))

        # Build source info
        sender = data.get("sender", "")
        chat_identifier = ""
        chat_name = ""
        service = data.get("service", "iMessage")

        if chat_meta:
            chat_identifier = chat_meta.get("identifier", "")
            chat_name = chat_meta.get("name", "") or chat_identifier
        else:
            chat_identifier = str(chat_id) if chat_id is not None else ""
            chat_name = chat_identifier

        # Use sender as user_id; chat_identifier as chat_id for routing
        effective_chat_id = chat_identifier or str(chat_id or "")
        user_id = sender or effective_chat_id

        # Determine chat type
        is_group = data.get("is_group", False)
        chat_type = "group" if is_group else "dm"

        source = self.build_source(
            chat_id=effective_chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=user_id,
            user_name=sender,
            user_id_alt=sender,
            chat_id_alt=str(chat_id) if chat_id is not None else None,
        )

        # Extract text
        text = data.get("text", "") or ""

        # Process attachments
        media_urls = []
        media_types = []
        attachments = data.get("attachments", [])
        for att in attachments:
            path = att.get("path", "")
            if path and os.path.exists(path):
                mime = att.get("mime_type", "")
                media_urls.append(path)
                media_types.append(mime or "application/octet-stream")

        # Determine message type
        if media_urls and not text:
            msg_type = MessageType.PHOTO if any(
                mt.startswith("image/") for mt in media_types
            ) else MessageType.DOCUMENT
        else:
            msg_type = MessageType.TEXT

        # Build timestamp
        timestamp = None
        ts = data.get("date")
        if ts:
            try:
                timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                pass

        event = MessageEvent(
            source=source,
            text=text,
            message_type=msg_type,
            media_urls=media_urls if media_urls else None,
            media_types=media_types if media_types else None,
            timestamp=timestamp,
        )

        logger.info(
            "iMessage: received from %s in %s: %s",
            _redact_imessage_id(sender),
            _redact_imessage_id(chat_name),
            text[:80] if text else "[media]",
        )
        await self.handle_message(event)

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------

    def _is_duplicate(self, guid: str) -> bool:
        """TTL-based dedup (matches Slack/WeCom pattern)."""
        now = time.monotonic()

        # Check existing
        if guid in self._seen_messages:
            return True

        # Prune expired entries
        if len(self._seen_messages) >= DEDUP_MAX_SIZE:
            expired = [k for k, v in self._seen_messages.items() if now - v > DEDUP_TTL]
            for k in expired:
                del self._seen_messages[k]
            # If still over limit, remove oldest
            if len(self._seen_messages) >= DEDUP_MAX_SIZE:
                oldest = min(self._seen_messages, key=self._seen_messages.get)
                del self._seen_messages[oldest]

        self._seen_messages[guid] = now
        return False

    # -----------------------------------------------------------------------
    # Chat cache
    # -----------------------------------------------------------------------

    def _parse_chat_list(self, output: str) -> None:
        """Parse `imsg chats --json` output into _chat_cache.

        imsg outputs one JSON object per line (NDJSON), not a JSON array.
        We also handle the JSON-array format for forward compatibility.
        """
        # Try JSON array first
        try:
            data = json.loads(output)
            if isinstance(data, list):
                for chat in data:
                    self._cache_chat_entry(chat)
                return
        except (json.JSONDecodeError, ValueError):
            pass

        # NDJSON: one JSON object per line
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chat = json.loads(line)
                self._cache_chat_entry(chat)
            except (json.JSONDecodeError, ValueError):
                continue

    def _cache_chat_entry(self, chat: dict) -> None:
        """Store a single chat entry in the cache."""
        cid = chat.get("id") or chat.get("chat_id")
        if cid is not None:
            self._chat_cache[int(cid)] = {
                "identifier": chat.get("identifier", ""),
                "name": chat.get("name", "") or chat.get("display_name", ""),
                "service": chat.get("service", "iMessage"),
            }

    async def _refresh_chat_cache(self) -> None:
        """Run `imsg chats --json` and refresh the cache."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "imsg", "chats", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode == 0:
                self._parse_chat_list(stdout.decode("utf-8", errors="replace"))
                logger.debug("iMessage: chat cache refreshed (%d entries)", len(self._chat_cache))
        except Exception as e:
            logger.warning("iMessage: chat cache refresh failed: %s", e)

    async def _cache_refresh_loop(self) -> None:
        """Periodically refresh the chat cache."""
        while self._running:
            try:
                await asyncio.sleep(CACHE_REFRESH_INTERVAL)
                if self._running:
                    await self._refresh_chat_cache()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("iMessage: cache refresh loop error: %s", e)

    # -----------------------------------------------------------------------
    # Health monitor
    # -----------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Check subprocess liveness periodically."""
        while self._running:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                if not self._running:
                    return

                # Check subprocess alive
                if self._watch_process and self._watch_process.returncode is not None:
                    logger.warning("iMessage: watch process not running (code=%s)", self._watch_process.returncode)

                # Check for idle
                idle_secs = time.monotonic() - self._last_message_time
                if idle_secs > HEALTH_CHECK_STALE_THRESHOLD:
                    logger.debug("iMessage: no messages for %.0fs (normal if no one is texting)", idle_secs)

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("iMessage: health monitor error: %s", e)

    # -----------------------------------------------------------------------
    # Outbound: send
    # -----------------------------------------------------------------------

    def _resolve_recipient(self, chat_id: str) -> str:
        """Resolve chat_id to a recipient for imsg send.

        If chat_id looks like a phone/email, use directly.
        Otherwise look up in chat cache. Fallback to chat_id as-is.
        """
        if not chat_id:
            return ""

        # Phone number or email — use directly
        if chat_id.startswith("+") or "@" in chat_id:
            return chat_id

        # Try numeric lookup in cache
        try:
            cid = int(chat_id)
            meta = self._chat_cache.get(cid)
            if meta and meta.get("identifier"):
                return meta["identifier"]
        except (ValueError, TypeError):
            pass

        # Fallback: return as-is
        return chat_id

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict] = None) -> SendResult:
        """Send a text message via imsg send."""
        recipient = self._resolve_recipient(chat_id)
        if not recipient:
            return SendResult(success=False, error="Could not resolve recipient")

        try:
            cmd = ["imsg", "send", "--to", recipient, "--text", content]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                return SendResult(success=False, error=f"imsg send failed (exit {proc.returncode}): {err}")

            logger.info("iMessage: sent to %s", _redact_imessage_id(recipient))
            return SendResult(success=True)

        except asyncio.TimeoutError:
            return SendResult(success=False, error="imsg send timed out")
        except Exception as e:
            return SendResult(success=False, error=f"imsg send error: {e}")

    async def send_typing(self, chat_id: str, metadata: Optional[dict] = None) -> None:
        """iMessage doesn't support typing indicators via imsg."""
        pass

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, **kwargs) -> SendResult:
        """Send an image (by URL — download first, then send as file)."""
        # For URLs, we'd need to download first. Delegate to send_image_file for local paths.
        if os.path.exists(image_url):
            return await self.send_image_file(chat_id, image_url, caption=caption, **kwargs)
        return SendResult(success=False, error="iMessage send_image only supports local files")

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None, **kwargs) -> SendResult:
        """Send an image file via imsg send --file."""
        return await self._send_file(chat_id, image_path, caption=caption)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None, **kwargs) -> SendResult:
        """Send a document file."""
        return await self._send_file(chat_id, file_path, caption=caption)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None, **kwargs) -> SendResult:
        """Send an audio file."""
        return await self._send_file(chat_id, audio_path, caption=caption)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None, **kwargs) -> SendResult:
        """Send a video file."""
        return await self._send_file(chat_id, video_path, caption=caption)

    async def _send_file(self, chat_id: str, file_path: str, caption: Optional[str] = None) -> SendResult:
        """Send a file via imsg send --file with optional caption."""
        recipient = self._resolve_recipient(chat_id)
        if not recipient:
            return SendResult(success=False, error="Could not resolve recipient")

        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        try:
            cmd = ["imsg", "send", "--to", recipient]
            if caption:
                cmd.extend(["--text", caption])
            cmd.extend(["--file", file_path])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                return SendResult(success=False, error=f"imsg send file failed (exit {proc.returncode}): {err}")

            logger.info("iMessage: sent file to %s: %s", _redact_imessage_id(recipient), Path(file_path).name)
            return SendResult(success=True)

        except asyncio.TimeoutError:
            return SendResult(success=False, error="imsg send file timed out")
        except Exception as e:
            return SendResult(success=False, error=f"imsg send file error: {e}")

    async def get_chat_info(self, chat_id: str) -> dict:
        """Return chat metadata."""
        try:
            cid = int(chat_id)
            meta = self._chat_cache.get(cid)
            if meta:
                return {
                    "name": meta.get("name", ""),
                    "type": "dm",
                    "chat_id": chat_id,
                    "identifier": meta.get("identifier", ""),
                    "service": meta.get("service", "iMessage"),
                }
        except (ValueError, TypeError):
            pass

        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
        }
