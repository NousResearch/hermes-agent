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

# Seconds to wait before restarting a crashed watch process
_WATCH_RESTART_DELAY = 5.0

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
        allowed_raw = os.getenv("IMESSAGE_ALLOWED_USERS", "+148****3441")
        self._allowed_users: List[str] = (
            _parse_comma_list(allowed_raw) if allowed_raw else []
        )
        self._allow_all = not self._allowed_users
        self._chat_identifiers: Dict[int, str] = {}

        # Per-message dedup: track seen rowids per chat to avoid re-dispatching
        # messages whose timestamp predates (or equals) a reply we just sent.
        # Capped at _MAX_SEEN_IDS_PER_CHAT entries per chat.
        self._seen_message_ids: Dict[int, Set[int]] = {}

        # Per-chat processing lock: skip a chat that is already being handled
        # so concurrent watch events don't pile up duplicate dispatches.
        self._chat_locks: Dict[int, asyncio.Lock] = {}

        # Recent-dispatch dedup: (chat_rowid, text_hash) -> monotonic timestamp.
        # Drops a message that was already dispatched within _DEDUP_WINDOW_SECONDS.
        self._recent_dispatches: Dict[Tuple[int, str], float] = {}

        # Per-chat watch tasks, keyed by chat rowid
        self._watch_tasks: Dict[int, asyncio.Task] = {}
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        if not check_imessage_requirements():
            logger.error("iMessage: imsg not found. Install: brew install steipete/tap/imsg")
            return False
        await self._refresh_chat_map()
        if not self._chat_identifiers:
            logger.warning("iMessage: no chats found — will not watch any chats until restart")
        else:
            logger.info("iMessage: seeded %d chat(s): %s",
                len(self._chat_identifiers),
                [_redact(v) for v in self._chat_identifiers.values()])

        self._running = True

        # For each allowed chat, find the max message rowid then spawn a watch task
        chats_to_watch = []
        for rowid, identifier in self._chat_identifiers.items():
            if not self._allow_all and identifier not in self._allowed_users:
                continue
            # Get current max rowid to avoid replaying history on connect
            since_rowid = await self._get_max_rowid(rowid)
            chats_to_watch.append((rowid, identifier, since_rowid))

        for rowid, identifier, since_rowid in chats_to_watch:
            task = asyncio.create_task(
                self._watch_chat(rowid, identifier, since_rowid),
                name=f"imsg-watch-{rowid}",
            )
            self._watch_tasks[rowid] = task
            logger.info(
                "iMessage: watching chat %d (%s) since rowid %d",
                rowid, _redact(identifier), since_rowid,
            )

        logger.info(
            "iMessage: connected — watching %d chat(s) via imsg watch",
            len(self._watch_tasks),
        )
        return True

    async def disconnect(self) -> None:
        self._running = False
        for rowid, task in list(self._watch_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._watch_tasks.clear()
        logger.info("iMessage: disconnected")

    # ── Send ─────────────────────────────────────────────────────────────────

    async def send(self, chat_id, content, *, reply_to=None, metadata=None, **kwargs) -> SendResult:
        # Detect intermediate streaming updates: GatewayStreamConsumer appends
        # a cursor character while the agent is still generating.  iMessage has
        # no edit-message API so we suppress partial fragments.
        # Returning a fake message_id lets the stream consumer establish an
        # "edit session" -- subsequent streaming deltas go via edit_message()
        # which we also suppress, and the final cursor-free edit is the only
        # real send that reaches AppleScript.
        for cursor in _STREAMING_CURSORS:
            if content.endswith(cursor):
                return SendResult(success=True, message_id=f"imsg-stream-{chat_id}")

        # Strip leading newlines (streaming artifacts) and trailing cursor chars.
        content = content.lstrip(chr(10))
        for cursor in _STREAMING_CURSORS:
            if content.endswith(cursor):
                content = content[: -len(cursor)].rstrip()

        if not content.strip():
            return SendResult(success=True)

        chunks = self._split_message(content)
        last_result = SendResult(success=False, error="no chunks")
        for chunk in chunks:
            last_result = await self._send_chunk(chat_id, chunk)
            if not last_result.success:
                break
        return last_result

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        """Handle streaming edit calls from GatewayStreamConsumer.

        Intermediate edits (content ends with cursor) are suppressed.
        The final cursor-free edit is the one real send we deliver.
        """
        # Suppress all intermediate streaming edits (have cursor appended).
        for cursor in _STREAMING_CURSORS:
            if content.endswith(cursor):
                return SendResult(success=True)

        # Final edit: strip leading newlines and send the clean text.
        content = content.lstrip(chr(10)).rstrip()
        if not content:
            return SendResult(success=True)

        chunks = self._split_message(content)
        last_result = SendResult(success=False, error="no chunks")
        for chunk in chunks:
            last_result = await self._send_chunk(chat_id, chunk)
            if not last_result.success:
                break
        return last_result

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

    # ── Watch loop ────────────────────────────────────────────────────────────

    async def _watch_chat(self, chat_rowid: int, identifier: str, since_rowid: int) -> None:
        """Persistent per-chat watch task using `imsg watch --since-rowid`."""
        imsg = _find_imsg()
        if not imsg:
            logger.error("iMessage: imsg binary not found; cannot watch chat %d", chat_rowid)
            return

        current_since = since_rowid

        while self._running:
            logger.debug(
                "iMessage: spawning watch for chat %d (%s) since rowid %d",
                chat_rowid, _redact(identifier), current_since,
            )
            try:
                proc = await asyncio.create_subprocess_exec(
                    imsg, "watch",
                    "--chat-id", str(chat_rowid),
                    "--since-rowid", str(current_since),
                    "--json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=_SUBPROCESS_ENV,
                )
            except Exception as exc:
                logger.error(
                    "iMessage: failed to spawn watch for chat %d: %s; retrying in %.1fs",
                    chat_rowid, exc, _WATCH_RESTART_DELAY,
                )
                await asyncio.sleep(_WATCH_RESTART_DELAY)
                continue

            try:
                await self._read_watch_stream(proc, chat_rowid, identifier, current_since)
            except asyncio.CancelledError:
                # Disconnect requested — kill the process and bail out
                try:
                    proc.kill()
                except Exception:
                    pass
                raise
            except Exception as exc:
                logger.error(
                    "iMessage: watch stream error for chat %d: %s",
                    chat_rowid, exc,
                )

            # Process exited unexpectedly (or errored) — grab the last seen rowid
            # from our seen_ids so we don't re-replay on restart
            seen_ids = self._seen_message_ids.get(chat_rowid)
            if seen_ids:
                current_since = max(seen_ids)

            if self._running:
                logger.warning(
                    "iMessage: watch process for chat %d exited; restarting in %.1fs",
                    chat_rowid, _WATCH_RESTART_DELAY,
                )
                await asyncio.sleep(_WATCH_RESTART_DELAY)

    async def _read_watch_stream(
        self,
        proc: asyncio.subprocess.Process,
        chat_rowid: int,
        identifier: str,
        since_rowid: int,
    ) -> None:
        """Read JSON lines from a running `imsg watch` process until it exits."""
        assert proc.stdout is not None

        # Expire old recent-dispatch entries to keep the dict bounded
        _now = time.monotonic()
        expired = [k for k, ts in self._recent_dispatches.items()
                   if _now - ts > _DEDUP_WINDOW_SECONDS * 2]
        for k in expired:
            del self._recent_dispatches[k]

        while True:
            try:
                line_bytes = await proc.stdout.readline()
            except Exception as exc:
                logger.debug("iMessage: readline error for chat %d: %s", chat_rowid, exc)
                break

            if not line_bytes:
                # EOF — process exited
                break

            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("iMessage: non-JSON line from watch (chat %d): %s", chat_rowid, line[:120])
                continue

            # Filter: skip outbound messages
            if msg.get("is_from_me", False):
                continue

            msg_rowid = msg.get("id") or msg.get("rowid")

            # Rowid-based dedup (safety net)
            seen_ids = self._seen_message_ids.setdefault(chat_rowid, set())
            if msg_rowid is not None:
                if msg_rowid in seen_ids:
                    logger.debug(
                        "iMessage: skipping already-seen rowid %d for chat %d",
                        msg_rowid, chat_rowid,
                    )
                    continue
                seen_ids.add(msg_rowid)
                # Cap set size to avoid unbounded memory growth
                if len(seen_ids) > _MAX_SEEN_IDS_PER_CHAT:
                    overflow = len(seen_ids) - _MAX_SEEN_IDS_PER_CHAT
                    for old_id in sorted(seen_ids)[:overflow]:
                        seen_ids.discard(old_id)

            # Per-chat lock: prevent concurrent dispatches for the same chat
            lock = self._chat_locks.get(chat_rowid)
            if lock is None:
                lock = asyncio.Lock()
                self._chat_locks[chat_rowid] = lock

            async with lock:
                await self._dispatch_message(chat_rowid, identifier, msg)

        # Wait for process to fully exit
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass

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

    # ── Helpers ───────────────────────────────────────────────────────────────

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

    async def _get_max_rowid(self, chat_rowid: int) -> int:
        """Return the highest message rowid currently in the chat history.

        Used at startup so `imsg watch --since-rowid` doesn't replay existing
        messages. Falls back to 0 if history is empty or unavailable.
        """
        messages = await self._get_history(chat_rowid, limit=_MAX_SEEN_IDS_PER_CHAT)
        if not messages:
            return 0
        # Pre-populate seen_ids as a safety net
        seen_ids = self._seen_message_ids.setdefault(chat_rowid, set())
        max_id = 0
        for m in messages:
            msg_rowid = m.get("id") or m.get("rowid")
            if msg_rowid is not None:
                seen_ids.add(msg_rowid)
                if msg_rowid > max_id:
                    max_id = msg_rowid
        return max_id

    async def _refresh_chat_map(self) -> None:
        chats = await self._get_chats()
        for chat in chats:
            rowid = chat.get("id")
            identifier = chat.get("identifier", "")
            if rowid is not None and identifier:
                self._chat_identifiers[rowid] = identifier

    def _split_message(self, text: str) -> List[str]:
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return [text]
        chunks = []
        while text:
            chunks.append(text[: self.MAX_MESSAGE_LENGTH])
            text = text[self.MAX_MESSAGE_LENGTH :]
        return chunks
