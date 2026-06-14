"""Shared helper classes for gateway platform adapters.

Extracts common patterns that were duplicated across 5-7 adapters:
message deduplication, text batch aggregation, markdown stripping,
and thread participation tracking.
"""

import asyncio
import json
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Dict

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
_RE_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_RE_LINK_WITH_URL = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_AUTOLINK = re.compile(r"<((?:https?|mailto):[^>\s]+)>")
_RE_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_RE_BULLET = re.compile(r"^[ \t]*[-*+]\s+", re.MULTILINE)
_RE_ORDERED_LIST = re.compile(r"^[ \t]*\d+[.)]\s+", re.MULTILINE)


class _MarkdownPlainTextExtractor(HTMLParser):
    """Extract readable plain text from Markdown-generated HTML."""

    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "dl",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol",
        "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead",
        "tr", "ul",
    }
    _PARAGRAPH_TAGS = {"p", "pre"}

    def __init__(self, *, preserve_urls: bool = False):
        super().__init__(convert_charrefs=True)
        self.preserve_urls = preserve_urls
        self._parts: list[str] = []
        self._skip_depth = 0
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        attrs_dict = {str(k).lower(): str(v) for k, v in attrs if v is not None}
        if tag in self._BLOCK_TAGS:
            self._newline()
        if tag == "li":
            self._append("• ")
        elif tag == "a":
            self._anchor_href = attrs_dict.get("href")
            self._anchor_text = []
        elif tag == "img":
            src = attrs_dict.get("src", "")
            alt = attrs_dict.get("alt", "")
            if self.preserve_urls and src:
                self._append(src)
            elif alt:
                self._append(alt)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return

        if tag == "a":
            href = self._anchor_href
            label = "".join(self._anchor_text).strip()
            if (
                self.preserve_urls
                and href
                and label
                and href.strip() != label
            ):
                self._append(f" ({href.strip()})")
            self._anchor_href = None
            self._anchor_text = []
        if tag in self._PARAGRAPH_TAGS:
            self._newline(blank=True)
        elif tag in self._BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if not data.strip() and "\n" in data:
            return
        self._append(data)
        if self._anchor_href is not None:
            self._anchor_text.append(data)

    def text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def _append(self, text: str) -> None:
        if text:
            self._parts.append(text)

    def _newline(self, *, blank: bool = False) -> None:
        if not self._parts:
            return
        current = "".join(self._parts)
        if blank:
            if current.endswith("\n\n"):
                return
            if current.endswith("\n"):
                self._parts.append("\n")
            else:
                self._parts.append("\n\n")
            return
        if current.endswith("\n"):
            return
        self._parts.append("\n")


def _regex_strip_markdown(text: str, *, preserve_urls: bool = False) -> str:
    """Regex fallback for environments where Python-Markdown is unavailable."""
    text = re.sub(
        r"```[a-zA-Z0-9_+-]*\n?([\s\S]*?)```",
        lambda m: m.group(1).rstrip("\n"),
        text,
    )
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_AUTOLINK.sub(r"\1", text)
    if preserve_urls:
        text = _RE_IMAGE.sub(lambda m: m.group(2), text)
        text = _RE_LINK_WITH_URL.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    else:
        text = _RE_IMAGE.sub(lambda m: m.group(1) or m.group(2), text)
        text = _RE_LINK_WITH_URL.sub(r"\1", text)
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_ITALIC_STAR.sub(r"\1", text)
    text = _RE_BOLD_UNDER.sub(r"\1", text)
    text = _RE_ITALIC_UNDER.sub(r"\1", text)
    text = _RE_STRIKE.sub(r"\1", text)
    text = _RE_HEADING.sub("", text)
    text = _RE_BULLET.sub("• ", text)
    text = _RE_ORDERED_LIST.sub("", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain-text platforms (SMS, iMessage, etc.).

    Replaces the identical ``_strip_markdown()`` functions previously
    duplicated in sms.py, bluebubbles.py, and feishu.py.
    """
    return strip_markdown_preserving_urls(text, preserve_urls=False)


def strip_markdown_preserving_urls(text: str, *, preserve_urls: bool = True) -> str:
    """Convert Markdown to readable plain text.

    Uses Python-Markdown plus a small HTML text extractor when available, so
    fenced code blocks, tables, inline HTML, nested emphasis, images, and
    links are handled by a real Markdown parser.  The regex path is only a
    fallback for unusual runtime environments.
    """
    if not text:
        return text

    try:
        import markdown as _markdown

        text = _RE_STRIKE.sub(r"\1", text)
        html = _markdown.markdown(
            text,
            extensions=["fenced_code", "sane_lists", "tables"],
            output_format="html",
        )
        parser = _MarkdownPlainTextExtractor(preserve_urls=preserve_urls)
        parser.feed(html)
        parser.close()
        return parser.text()
    except Exception:
        logger.debug("Markdown parser unavailable; using regex fallback", exc_info=True)
        return _regex_strip_markdown(text, preserve_urls=preserve_urls)


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
