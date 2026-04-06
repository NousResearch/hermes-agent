"""iMessage platform adapter.

Uses the `imsg` CLI (https://github.com/steipete/imsg) to send and receive
iMessages via macOS Messages.app.

Requirements:
  - macOS with Messages.app signed in to an Apple ID
  - `imsg` installed: brew install steipete/tap/imsg
  - Full Disk Access granted to the process running hermes (System Settings
    -> Privacy -> Full Disk Access)

Optional environment variables:
  IMESSAGE_ALLOWED_USERS   Comma-separated Apple IDs or E.164 phone numbers
                           allowed to interact. Leave unset to allow all.
  IMESSAGE_HOME_CHANNEL    Chat rowid (integer) to use as default cron
                           delivery target.
  IMESSAGE_POLL_INTERVAL   Seconds between polls (default: 30).
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Set, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 8000
DEFAULT_POLL_INTERVAL = 30.0

# Maximum seen message rowids to track per chat (prevents unbounded growth)
_MAX_SEEN_IDS_PER_CHAT = 200

# Seconds within which a (chat, text-hash) combo is considered a duplicate
_DEDUP_WINDOW_SECONDS = 60

# Streaming cursor characters appended by GatewayStreamConsumer during
# intermediate (non-final) sends.  iMessage has no edit-message capability,
# so we detect these and suppress the send entirely rather than letting a
# partial-response fragment land in the conversation.
_STREAMING_CURSORS = (" ▉", "▉", "▌", "█", "…")

# Full path to imsg binary — gateway may run with a stripped PATH
_IMSG_BIN: Optional[str] = None
_IMSG_CANDIDATES = [
    "/opt/homebrew/bin/imsg",
    "/usr/local/bin/imsg",
    "/opt/local/bin/imsg",
]

# Subprocess environment with homebrew bin in PATH
_SUBPROCESS_ENV = os.environ.copy()
if "/opt/homebrew/bin" not in _SUBPROCESS_ENV.get("PATH", ""):
    _SUBPROCESS_ENV["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + _SUBPROCESS_ENV.get("PATH", "")


def _find_imsg() -> Optional[str]:
    """Locate the imsg binary, checking known paths then PATH."""
    global _IMSG_BIN
    if _IMSG_BIN:
        return _IMSG_BIN
    for candidate in _IMSG_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            _IMSG_BIN = candidate
            return _IMSG_BIN
    import shutil
    found = shutil.which("imsg")
    if found:
        _IMSG_BIN = found
    return _IMSG_BIN


def _redact(identifier: str) -> str:
    """Redact an Apple ID or phone for safe logging."""
    if not identifier:
        return "<none>"
    if "@" in identifier:
        local, domain = identifier.split("@", 1)
        return local[:3] + "***@" + domain if len(local) > 3 else "***@" + domain
    if identifier.startswith("+"):
        return identifier[:5] + "***" + identifier[-3:]
    return identifier[:4] + "***"


def _parse_comma_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def check_imessage_requirements() -> bool:
    """Return True if imsg CLI is available."""
    imsg = _find_imsg()
    if not imsg:
        return False
    try:
        result = subprocess.run(
            [imsg, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_SUBPROCESS_ENV,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


async def _run_imsg(*args: str) -> Optional[str]:
    """Run an imsg command async and return stdout, or None on failure."""
    imsg = _find_imsg()
    if not imsg:
        logger.error("iMessage: imsg binary not found")
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            imsg, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=_SUBPROCESS_ENV,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        decoded = stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            count = getattr(_run_imsg, "_perm_error_count", 0) + 1
            _run_imsg._perm_error_count = count
            if "permissionDenied" in decoded or "authorization denied" in decoded:
                if count == 1 or count % 20 == 0:
                    logger.warning(
                        "iMessage: FDA not granted — grant Full Disk Access to the "
                        "Python interpreter running this process in "
                        "System Settings > Privacy > Full Disk Access (error %d)", count
                    )
            else:
                logger.warning("iMessage: imsg %s exit=%d stderr=%s stdout=%s",
                               args[0], proc.returncode, err[:200], decoded[:200])
            return None
        if not decoded.strip():
            logger.debug("iMessage: imsg %s returned empty output", args[0])
        _run_imsg._perm_error_count = 0
        return decoded
    except asyncio.TimeoutError:
        logger.error("iMessage: imsg %s timed out", args[0] if args else "?")
        return None
    except Exception as exc:
        logger.error("iMessage: imsg %s error: %s", args[0] if args else "?", exc)
        return None


class IMessageAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.IMESSAGE)
        self._poll_interval: float = float(
            os.getenv("IMESSAGE_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))
        )
        allowed_raw = os.getenv("IMESSAGE_ALLOWED_USERS", "+14806893441")
        self._allowed_users: List[str] = (
            _parse_comma_list(allowed_raw) if allowed_raw else []
        )
        self._allow_all = not self._allowed_users
        self._chat_identifiers: Dict[int, str] = {}
        self._last_seen: Dict[int, str] = {}

        # Per-message dedup: track seen rowids per chat to avoid re-dispatching
        # messages whose timestamp predates (or equals) a reply we just sent.
        # Capped at _MAX_SEEN_IDS_PER_CHAT entries per chat.
        self._seen_message_ids: Dict[int, Set[int]] = {}

        # Per-chat processing lock: skip a chat that is already being handled
        # so concurrent polls don't pile up duplicate dispatches.
        self._chat_locks: Dict[int, asyncio.Lock] = {}

        # Recent-dispatch dedup: (chat_rowid, text_hash) -> monotonic timestamp.
        # Drops a message that was already dispatched within _DEDUP_WINDOW_SECONDS.
        self._recent_dispatches: Dict[Tuple[int, str], float] = {}

        self._poll_task: Optional[asyncio.Task] = None
        self._stream_chats: set = set()
        self._last_sent_content: dict = {}
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        if not check_imessage_requirements():
            logger.error("iMessage: imsg not found. Install: brew install steipete/tap/imsg")
            return False
        await self._refresh_chat_map()
        if not self._chat_identifiers:
            logger.warning("iMessage: no chats found — will keep polling until Messages.app has history")
        else:
            logger.info("iMessage: seeded %d chat(s): %s",
                len(self._chat_identifiers),
                [_redact(v) for v in self._chat_identifiers.values()])
        await self._seed_last_seen()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("iMessage: connected (poll every %.1fs)", self._poll_interval)
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("iMessage: disconnected")

    # ── Send ─────────────────────────────────────────────────────────────────

    async def send(self, chat_id, content, *, reply_to=None, metadata=None, **kwargs) -> SendResult:
        # iMessage cannot edit messages so streaming is handled by sending
        # delta content as separate bubbles.
        #
        # Cursor-bearing sends (intermediate streaming): strip cursor, send
        # the new delta since last send, return a fake message_id so the
        # stream consumer tracks a session (already_sent=True prevents
        # the normal response path from double-sending).
        #
        # Non-cursor sends while in a stream session: this is the got_done
        # final send -- also send as delta, also return fake message_id.
        #
        # Non-cursor sends outside any stream session: normal send.
        has_cursor = any(content.endswith(c) for c in _STREAMING_CURSORS)
        if has_cursor:
            for cursor in _STREAMING_CURSORS:
                if content.endswith(cursor):
                    content = content[:-len(cursor)]
            content = content.rstrip()

        content = content.lstrip("\n").rstrip()
        if not content:
            if has_cursor:
                return SendResult(success=True, message_id=f"imsg-{chat_id}")
            return SendResult(success=True)

        # Delta: only send text that is new since the last send for this chat
        last = self._last_sent_content.get(str(chat_id), "")
        if last and content.startswith(last):
            delta = content[len(last):].lstrip("\n").strip()
        else:
            delta = content

        if not delta:
            if has_cursor or str(chat_id) in self._stream_chats:
                return SendResult(success=True, message_id=f"imsg-{chat_id}")
            return SendResult(success=True)

        if has_cursor or str(chat_id) in self._stream_chats:
            self._stream_chats.add(str(chat_id))

        chunks = self._split_message(delta)
        last_result = SendResult(success=False, error="no chunks")
        for chunk in chunks:
            last_result = await self._send_chunk(chat_id, chunk)
            if not last_result.success:
                break

        if last_result.success:
            self._last_sent_content[str(chat_id)] = content
            if str(chat_id) in self._stream_chats:
                return SendResult(success=True, message_id=f"imsg-{chat_id}")

        return last_result

    async def edit_message(self, chat_id, message_id, content, metadata=None, **kwargs) -> SendResult:
        """Stream consumer edits -- deliver as delta send."""
        return await self.send(chat_id, content, metadata=metadata)

    async def _send_chunk(self, chat_id: str, text: str) -> SendResult:
        if chat_id.isdigit():
            identifier = self._chat_identifiers.get(int(chat_id))
            if not identifier:
                return SendResult(success=False, error="no identifier for chat")
        else:
            identifier = chat_id
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'tell application "Messages"\n'
            '  set targetService to 1st service whose service type = iMessage\n'
            f'  send "{escaped}" to buddy "{identifier}" of targetService\n'
            'end tell'
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=_SUBPROCESS_ENV,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
            if proc.returncode == 0:
                return SendResult(success=True)
            err = (stderr or stdout or b"unknown").decode(errors="replace").strip()
            return SendResult(success=False, error=err)
        except asyncio.TimeoutError:
            return SendResult(success=False, error="timeout")
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Start iMessage typing indicator via imsg."""
        imsg = _find_imsg()
        if not imsg:
            return
        try:
            await asyncio.create_subprocess_exec(
                imsg, "typing", "--chat-id", str(chat_id), "--duration", "45s",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                env=_SUBPROCESS_ENV,
            )
        except Exception:
            pass

    async def stop_typing(self, chat_id: str) -> None:
        """Stop iMessage typing indicator."""
        imsg = _find_imsg()
        if not imsg:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                imsg, "typing", "--chat-id", str(chat_id), "--stop", "true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                env=_SUBPROCESS_ENV,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:
            pass

    async def get_chat_info(self, chat_id: str) -> dict:
        identifier = self._chat_identifiers.get(int(chat_id) if chat_id.isdigit() else -1, chat_id)
        return {"name": identifier, "type": "private", "chat_id": chat_id}

    # ── Processing lifecycle hooks ────────────────────────────────────────────

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Start typing indicator when processing begins."""
        chat_id = event.source.chat_id if event.source else None
        if chat_id:
            await self.send_typing(chat_id)

    async def on_processing_complete(self, event: MessageEvent, success: bool) -> None:
        """Stop typing indicator when processing finishes."""
        chat_id = event.source.chat_id if event.source else None
        if chat_id:
            await self.stop_typing(chat_id)

    async def _run_applescript(self, script: str) -> None:
        """Fire-and-forget AppleScript execution."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=_SUBPROCESS_ENV,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10)
        except Exception as exc:
            logger.debug("iMessage: AppleScript error: %s", exc)

    # ── Poll loop ─────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._check_new_messages()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("iMessage: poll error: %s", exc)
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break

    async def _check_new_messages(self) -> None:
        chats = await self._get_chats()
        if not chats:
            return

        # Expire old recent-dispatch entries to keep the dict bounded.
        _now = time.monotonic()
        expired = [k for k, ts in self._recent_dispatches.items()
                   if _now - ts > _DEDUP_WINDOW_SECONDS * 2]
        for k in expired:
            del self._recent_dispatches[k]

        for chat in chats:
            rowid = chat.get("id")
            identifier = chat.get("identifier", "")
            last_msg_at = chat.get("last_message_at")
            if rowid is None or not last_msg_at:
                continue
            if identifier:
                self._chat_identifiers[rowid] = identifier
            if not self._allow_all and identifier not in self._allowed_users:
                continue

            # Skip chats that are already being processed to avoid concurrent
            # duplicate dispatches when agent response time exceeds poll interval.
            lock = self._chat_locks.get(rowid)
            if lock is None:
                lock = asyncio.Lock()
                self._chat_locks[rowid] = lock
            if lock.locked():
                logger.debug("iMessage: chat %d already processing, skipping poll", rowid)
                continue

            prev_seen = self._last_seen.get(rowid)
            if prev_seen is None:
                self._last_seen[rowid] = last_msg_at
                continue
            if last_msg_at <= prev_seen:
                continue

            messages = await self._get_history(rowid, limit=20)
            new_messages = [
                m for m in messages
                if m.get("created_at", "") > prev_seen
                and not m.get("is_from_me", True)
            ]

            # Secondary rowid-based filter: only messages we haven't seen before.
            seen_ids = self._seen_message_ids.setdefault(rowid, set())
            truly_new = []
            for m in new_messages:
                msg_rowid = m.get("rowid") or m.get("id")
                if msg_rowid is None:
                    # No rowid available — fall back to timestamp filter only
                    truly_new.append(m)
                elif msg_rowid not in seen_ids:
                    truly_new.append(m)

            if not truly_new:
                # Update timestamp even if nothing new to dispatch so we
                # don't keep re-fetching history on every poll.
                self._last_seen[rowid] = last_msg_at
                continue

            # Mark all fetched rowids as seen *before* dispatching so that
            # if a concurrent poll fires while we're awaiting the agent the
            # rowids are already recorded.
            for m in truly_new:
                msg_rowid = m.get("rowid") or m.get("id")
                if msg_rowid is not None:
                    seen_ids.add(msg_rowid)
            # Cap set size to avoid unbounded memory growth.
            if len(seen_ids) > _MAX_SEEN_IDS_PER_CHAT:
                overflow = len(seen_ids) - _MAX_SEEN_IDS_PER_CHAT
                # Discard the smallest (oldest) rowids.
                for old_id in sorted(seen_ids)[:overflow]:
                    seen_ids.discard(old_id)

            self._last_seen[rowid] = last_msg_at

            async with lock:
                for msg in truly_new:
                    await self._dispatch_message(rowid, identifier, msg)

    async def _dispatch_message(self, chat_rowid: int, identifier: str, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            return

        # Recent-dispatch dedup: drop if the same (chat, text) was dispatched
        # within the dedup window (guards against edge-case race conditions not
        # covered by the rowid filter alone).
        text_hash = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        dedup_key = (chat_rowid, text_hash)
        _now = time.monotonic()
        last_dispatch = self._recent_dispatches.get(dedup_key)
        if last_dispatch is not None and (_now - last_dispatch) < _DEDUP_WINDOW_SECONDS:
            logger.warning(
                "iMessage: dropping duplicate dispatch for chat %d (text hash %s, "
                "%.1fs since last dispatch)",
                chat_rowid, text_hash, _now - last_dispatch,
            )
            return
        self._recent_dispatches[dedup_key] = _now

        chat_id_str = str(chat_rowid)
        source = self.build_source(
            chat_id=chat_id_str,
            user_id=identifier or chat_id_str,
            chat_name=identifier or chat_id_str,
            user_name=identifier or chat_id_str,
        )
        event = MessageEvent(
            message_type=MessageType.TEXT,
            text=text,
            source=source,
            raw_message=msg,
        )
        await self.handle_message(event)

    async def _get_chats(self) -> List[dict]:
        output = await _run_imsg("chats", "--json")
        if not output:
            return []
        chats = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chats.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return chats

    async def _get_history(self, chat_rowid: int, limit: int = 20) -> List[dict]:
        output = await _run_imsg("history", "--chat-id", str(chat_rowid), "--limit", str(limit), "--json")
        if not output:
            return []
        messages = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return messages

    async def _refresh_chat_map(self) -> None:
        chats = await self._get_chats()
        for chat in chats:
            rowid = chat.get("id")
            identifier = chat.get("identifier", "")
            if rowid is not None and identifier:
                self._chat_identifiers[rowid] = identifier

    async def _seed_last_seen(self) -> None:
        """Seed last-seen timestamps and pre-populate seen rowids so existing
        messages are not re-dispatched on first connect."""
        chats = await self._get_chats()
        for chat in chats:
            rowid = chat.get("id")
            last_msg_at = chat.get("last_message_at")
            if rowid is None or not last_msg_at:
                continue
            self._last_seen[rowid] = last_msg_at
            # Pre-populate seen rowids from the most recent messages so we
            # don't dispatch historical messages after a restart.
            messages = await self._get_history(rowid, limit=_MAX_SEEN_IDS_PER_CHAT)
            seen_ids = self._seen_message_ids.setdefault(rowid, set())
            for m in messages:
                msg_rowid = m.get("rowid") or m.get("id")
                if msg_rowid is not None:
                    seen_ids.add(msg_rowid)

    def _split_message(self, text: str) -> List[str]:
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return [text]
        chunks = []
        while text:
            chunks.append(text[: self.MAX_MESSAGE_LENGTH])
            text = text[self.MAX_MESSAGE_LENGTH :]
        return chunks
