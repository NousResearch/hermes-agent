"""Shared helper classes for gateway platform adapters.

Extracts common patterns that were duplicated across 5-7 adapters:
message deduplication, text batch aggregation, markdown stripping,
and thread participation tracking.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from utils import atomic_json_write

if TYPE_CHECKING:
    from gateway.platforms.base import MessageEvent

logger = logging.getLogger(__name__)


# ─── Message Deduplication ────────────────────────────────────────────────────


class MessageDeduplicator:
    """TTL-based message deduplication cache.

    Replaces the identical ``_seen_messages`` / ``_is_duplicate()`` pattern
    previously duplicated in discord, slack, dingtalk, wecom, weixin,
    mattermost, and feishu adapters.

    Usage::

        self._dedup = MessageDeduplicator()

        # In message handler:
        if self._dedup.is_duplicate(msg_id):
            return
    """

    def __init__(self, max_size: int = 2000, ttl_seconds: float = 300):
        self._seen: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def is_duplicate(self, msg_id: str) -> bool:
        """Return True if *msg_id* was already seen within the TTL window."""
        if not msg_id:
            return False
        now = time.time()
        if msg_id in self._seen:
            if now - self._seen[msg_id] < self._ttl:
                return True
            # Entry has expired — remove it and treat as new
            del self._seen[msg_id]
        self._seen[msg_id] = now
        if len(self._seen) > self._max_size:
            cutoff = now - self._ttl
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
            if len(self._seen) > self._max_size:
                # TTL pruning alone does not cap the cache when every entry is
                # still fresh. Keep the newest entries so the helper's
                # max_size bound is enforced under sustained traffic.
                newest = sorted(
                    self._seen.items(),
                    key=lambda item: item[1],
                )[-self._max_size:]
                self._seen = dict(newest)
        return False

    def clear(self):
        """Clear all tracked messages."""
        self._seen.clear()


# ─── Text Batch Aggregation ──────────────────────────────────────────────────


class TextBatchAggregator:
    """Aggregates rapid-fire text events into single messages.

    Replaces the ``_enqueue_text_event`` / ``_flush_text_batch`` pattern
    previously duplicated in telegram, discord, matrix, wecom, and feishu.

    Usage::

        self._text_batcher = TextBatchAggregator(
            handler=self._message_handler,
            batch_delay=0.6,
            split_threshold=1900,
        )

        # In message dispatch:
        if msg_type == MessageType.TEXT and self._text_batcher.is_enabled():
            self._text_batcher.enqueue(event, session_key)
            return
    """

    def __init__(
        self,
        handler,
        *,
        batch_delay: float = 0.6,
        split_delay: float = 2.0,
        split_threshold: int = 4000,
    ):
        self._handler = handler
        self._batch_delay = batch_delay
        self._split_delay = split_delay
        self._split_threshold = split_threshold
        self._pending: Dict[str, "MessageEvent"] = {}
        self._pending_tasks: Dict[str, asyncio.Task] = {}

    def is_enabled(self) -> bool:
        """Return True if batching is active (delay > 0)."""
        return self._batch_delay > 0

    def enqueue(self, event: "MessageEvent", key: str) -> None:
        """Add *event* to the pending batch for *key*."""
        chunk_len = len(event.text or "")
        existing = self._pending.get(key)
        if not existing:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending[key] = event
        else:
            existing.text = f"{existing.text}\n{event.text}"
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]

        # Cancel prior flush timer, start a new one
        prior = self._pending_tasks.get(key)
        if prior and not prior.done():
            prior.cancel()
        self._pending_tasks[key] = asyncio.create_task(self._flush(key))

    async def _flush(self, key: str) -> None:
        """Wait then dispatch the batched event for *key*."""
        current_task = self._pending_tasks.get(key)
        pending = self._pending.get(key)
        last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0

        # Use longer delay when the last chunk looks like a split message
        delay = self._split_delay if last_len >= self._split_threshold else self._batch_delay
        await asyncio.sleep(delay)

        event = self._pending.pop(key, None)
        if event:
            try:
                await self._handler(event)
            except Exception:
                logger.exception("[TextBatchAggregator] Error dispatching batched event for %s", key)

        if self._pending_tasks.get(key) is current_task:
            self._pending_tasks.pop(key, None)

    def cancel_all(self) -> None:
        """Cancel all pending flush tasks."""
        for task in self._pending_tasks.values():
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()
        self._pending.clear()


# ─── Markdown Stripping ──────────────────────────────────────────────────────

# Pre-compiled regexes for performance
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_RE_ITALIC_STAR = re.compile(r"\*(.+?)\*", re.DOTALL)
_RE_BOLD_UNDER = re.compile(r"\b__(?![\s_])(.+?)(?<![\s_])__\b", re.DOTALL)
_RE_ITALIC_UNDER = re.compile(r"\b_(?![\s_])(.+?)(?<![\s_])_\b", re.DOTALL)
_RE_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\n?")
_RE_INLINE_CODE = re.compile(r"`(.+?)`")
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain-text platforms (SMS, iMessage, etc.).

    Replaces the identical ``_strip_markdown()`` functions previously
    duplicated in sms.py, bluebubbles.py, and feishu.py.
    """
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_ITALIC_STAR.sub(r"\1", text)
    text = _RE_BOLD_UNDER.sub(r"\1", text)
    text = _RE_ITALIC_UNDER.sub(r"\1", text)
    text = _RE_CODE_BLOCK.sub("", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_HEADING.sub("", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ─── Thread Participation Tracking ───────────────────────────────────────────


class ThreadParticipationTracker:
    """Persistent tracking of threads the bot has participated in.

    Replaces the identical ``_load/_save_participated_threads`` +
    ``_mark_thread_participated`` pattern previously duplicated in
    discord.py and matrix.py.

    Usage::

        self._threads = ThreadParticipationTracker("discord")

        # Check membership:
        if thread_id in self._threads:
            ...

        # Mark participation:
        self._threads.mark(thread_id)
    """

    _MAX_TRACKED = 500

    def __init__(self, platform_name: str, max_tracked: int = 500):
        self._platform = platform_name
        self._max_tracked = max_tracked
        self._threads: dict[str, None] = {
            str(thread_id): None for thread_id in self._load()
        }

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / f"{self._platform}_threads.json"

    def _load(self) -> list[str]:
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [str(thread_id) for thread_id in data]
            except Exception:
                pass
        return []

    def _save(self) -> None:
        path = self._state_path()
        thread_list = list(self._threads)
        if len(thread_list) > self._max_tracked:
            thread_list = thread_list[-self._max_tracked:]
            self._threads = dict.fromkeys(thread_list)
        atomic_json_write(path, thread_list, indent=None)

    def mark(self, thread_id: str) -> None:
        """Mark *thread_id* as participated and persist."""
        if thread_id not in self._threads:
            self._threads[thread_id] = None
            self._save()

    def __contains__(self, thread_id: str) -> bool:
        return thread_id in self._threads

    def clear(self) -> None:
        self._threads.clear()


# ─── Bot Thread Routing State ────────────────────────────────────────────────


class BotThreadMembershipTracker:
    """Persistent, TTL-bound authorization for bot-to-bot thread followups.

    This is intentionally separate from ``ThreadParticipationTracker``.  Generic
    thread participation means "this bot may answer humans without repeated
    mentions."  Bot-thread membership is authorization state for a specific
    sender bot talking to this receiver bot in a specific Discord thread.
    """

    def __init__(self, platform_name: str, ttl_seconds: float = 24 * 60 * 60):
        self._platform = platform_name
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._records: Dict[str, Dict[str, Any]] = self._load()
        self._evict_expired(save=True)

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / f"{self._platform}_bot_threads.json"

    @staticmethod
    def _key(receiver_bot_id: str, sender_bot_id: str, thread_id: str) -> str:
        return f"{receiver_bot_id}:{sender_bot_id}:{thread_id}"

    def _load(self) -> Dict[str, Dict[str, Any]]:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            logger.debug("Failed to load %s bot-thread state", self._platform, exc_info=True)
        return {}

    def _save(self) -> None:
        atomic_json_write(self._state_path(), self._records, indent=0)

    def _evict_expired(self, *, save: bool = False) -> None:
        now = time.time()
        before = len(self._records)
        self._records = {
            key: rec
            for key, rec in self._records.items()
            if float(rec.get("expires_at", 0) or 0) > now
        }
        if save and len(self._records) != before:
            self._save()

    def mark(
        self,
        *,
        receiver_bot_id: str,
        sender_bot_id: str,
        thread_id: str,
        parent_channel_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
    ) -> None:
        now = time.time()
        key = self._key(str(receiver_bot_id), str(sender_bot_id), str(thread_id))
        self._records[key] = {
            "receiver_bot_id": str(receiver_bot_id),
            "sender_bot_id": str(sender_bot_id),
            "thread_id": str(thread_id),
            "parent_channel_id": str(parent_channel_id) if parent_channel_id else None,
            "source_message_id": str(source_message_id) if source_message_id else None,
            "last_seen": now,
            "expires_at": now + self._ttl_seconds,
        }
        self._evict_expired()
        self._save()

    def refresh(self, *, receiver_bot_id: str, sender_bot_id: str, thread_id: str) -> None:
        key = self._key(str(receiver_bot_id), str(sender_bot_id), str(thread_id))
        rec = self._records.get(key)
        if not rec:
            return
        now = time.time()
        rec["last_seen"] = now
        rec["expires_at"] = now + self._ttl_seconds
        self._save()

    def contains(self, *, receiver_bot_id: str, sender_bot_id: str, thread_id: str) -> bool:
        self._evict_expired(save=True)
        key = self._key(str(receiver_bot_id), str(sender_bot_id), str(thread_id))
        return key in self._records


class BotLoopFuse:
    """Persistent rate fuse for accepted bot-authored Discord messages."""

    def __init__(
        self,
        platform_name: str,
        *,
        window_seconds: Optional[float] = None,
        max_messages: Optional[int] = None,
        suppress_seconds: Optional[float] = None,
    ):
        # Defaults are intentionally conservative but less brittle than the
        # previous 3/60s/600s setting, which dropped legitimate PM→Galt
        # escalation bursts. Worst-case accepted loop throughput is roughly
        # max_messages * 3600 / (window_seconds + suppress_seconds):
        # 5 * 3600 / 300 ≈ 60 accepted messages/hour per sender/receiver/thread.
        self._platform = platform_name
        self._window_seconds = max(
            1.0,
            float(
                self._env_float(
                    "DISCORD_BOT_LOOP_FUSE_WINDOW_SECONDS",
                    60.0 if window_seconds is None else window_seconds,
                )
            ),
        )
        self._max_messages = max(
            1,
            int(
                self._env_int(
                    "DISCORD_BOT_LOOP_FUSE_MAX_MESSAGES",
                    5 if max_messages is None else max_messages,
                )
            ),
        )
        self._suppress_seconds = max(
            1.0,
            float(
                self._env_float(
                    "DISCORD_BOT_LOOP_FUSE_SUPPRESS_SECONDS",
                    4 * 60 if suppress_seconds is None else suppress_seconds,
                )
            ),
        )
        self._state: Dict[str, Dict[str, Any]] = self._load()
        self._evict_old(save=True)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
            return float(default)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return int(default)
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
            return int(default)

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / f"{self._platform}_bot_loop_fuse.json"

    @staticmethod
    def _key(receiver_bot_id: str, sender_bot_id: str, thread_id: Optional[str]) -> str:
        return f"{receiver_bot_id}:{sender_bot_id}:{thread_id or 'channel'}"

    def _load(self) -> Dict[str, Dict[str, Any]]:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            logger.debug("Failed to load %s bot-loop fuse state", self._platform, exc_info=True)
        return {}

    def _save(self) -> None:
        atomic_json_write(self._state_path(), self._state, indent=0)

    def _evict_old(self, *, save: bool = False) -> None:
        now = time.time()
        before = len(self._state)
        cutoff = now - self._window_seconds
        new_state: Dict[str, Dict[str, Any]] = {}
        for key, rec in self._state.items():
            suppressed_until = float(rec.get("suppressed_until", 0) or 0)
            events = [float(ts) for ts in rec.get("events", []) if float(ts) >= cutoff]
            if suppressed_until > now or events:
                new_state[key] = {**rec, "events": events}
        self._state = new_state
        if save and len(self._state) != before:
            self._save()

    def is_suppressed(self, *, receiver_bot_id: str, sender_bot_id: str, thread_id: Optional[str]) -> bool:
        self._evict_old(save=True)
        key = self._key(str(receiver_bot_id), str(sender_bot_id), str(thread_id) if thread_id else None)
        return float(self._state.get(key, {}).get("suppressed_until", 0) or 0) > time.time()

    def record_and_check(
        self,
        *,
        receiver_bot_id: str,
        sender_bot_id: str,
        thread_id: Optional[str],
    ) -> bool:
        """Record an accepted bot message. Return True if this message trips suppression."""
        self._evict_old()
        now = time.time()
        key = self._key(str(receiver_bot_id), str(sender_bot_id), str(thread_id) if thread_id else None)
        rec = self._state.setdefault(key, {"events": [], "suppressed_until": 0})
        cutoff = now - self._window_seconds
        events = [float(ts) for ts in rec.get("events", []) if float(ts) >= cutoff]
        events.append(now)
        tripped = len(events) > self._max_messages
        rec["events"] = events
        if tripped:
            rec["suppressed_until"] = now + self._suppress_seconds
        self._save()
        return tripped


# ─── Phone Number Redaction ──────────────────────────────────────────────────


def redact_phone(phone: str) -> str:
    """Redact a phone number for logging, preserving country code and last 4.

    Replaces the identical ``_redact_phone()`` functions in signal.py,
    sms.py, and bluebubbles.py.
    """
    if not phone:
        return "<none>"
    if len(phone) <= 8:
        return phone[:2] + "****" + phone[-2:] if len(phone) > 4 else "****"
    return phone[:4] + "****" + phone[-4:]


# ─── GFM Markdown Table → Bullet Conversion ─────────────────────────────────
# Shared by Discord and Telegram adapters.  Discord calls
# convert_table_to_bullets() directly; Telegram imports the primitives
# but keeps its own MarkdownV2-aware renderer.


# Matches a GFM table delimiter row: optional outer pipes, cells of dashes
# (with optional alignment colons) separated by '|'.
# Requires at least one internal '|' so lone '---' rules are NOT matched.
TABLE_SEPARATOR_RE = re.compile(
    r'^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$'
)


def is_table_row(line: str) -> bool:
    """Return True if *line* could plausibly be a table data row."""
    stripped = line.strip()
    return bool(stripped) and '|' in stripped


def split_markdown_table_row(line: str) -> list[str]:
    """Split a GFM table row into stripped cell values."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _render_table_block(table_block: list[str]) -> str:
    """Render a detected GFM table as bold-heading + bullet groups.

    Uses the same alignment logic as Telegram's renderer: for non-row-label
    tables, ``data_cells = cells`` (the full row) and the bullet whose value
    duplicates the heading is skipped.  This keeps header→value alignment
    correct.
    """
    if len(table_block) < 3:
        return "\n".join(table_block)

    headers = split_markdown_table_row(table_block[0])
    if len(headers) < 2:
        return "\n".join(table_block)

    first_data_row = (
        split_markdown_table_row(table_block[2])
        if len(table_block) > 2
        else []
    )
    has_row_label_col = len(first_data_row) == len(headers) + 1

    rendered_groups: list[str] = []
    for index, row in enumerate(table_block[2:], start=1):
        cells = split_markdown_table_row(row)
        if has_row_label_col:
            heading = cells[0] if cells and cells[0] else f"Row {index}"
            data_cells = cells[1:]
        else:
            heading = next((cell for cell in cells if cell), f"Row {index}")
            data_cells = cells

        if len(data_cells) < len(headers):
            data_cells.extend([""] * (len(headers) - len(data_cells)))
        elif len(data_cells) > len(headers):
            data_cells = data_cells[: len(headers)]

        bullets: list[str] = []
        for header, value in zip(headers, data_cells):
            if not has_row_label_col and value == heading:
                continue
            bullets.append(f"• {header}: {value}")

        group_lines = [f"**{heading}**", *bullets]
        rendered_groups.append("\n".join(group_lines))

    return "\n\n".join(rendered_groups)


def convert_table_to_bullets(text: str) -> str:
    """Rewrite GFM pipe tables into bold-heading + bullet groups.

    Tables inside fenced code blocks are left alone.
    """
    if '|' not in text or '-' not in text:
        return text

    lines = text.split('\n')
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith('```'):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        if (
            '|' in line
            and i + 1 < len(lines)
            and TABLE_SEPARATOR_RE.match(lines[i + 1])
        ):
            table_block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and is_table_row(lines[j]):
                table_block.append(lines[j])
                j += 1
            out.append(_render_table_block(table_block))
            i = j
            continue

        out.append(line)
        i += 1

    return '\n'.join(out)
