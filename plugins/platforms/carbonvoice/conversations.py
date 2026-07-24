"""Per-conversation state for the Carbon Voice adapter.

`ConversationTracker` consolidates the mutable per-thread state that
otherwise leaks across `adapter.py`. PR 1 introduces only the pieces
that replace existing adapter state:

- **Reply anchors** — outbound threading targets, *keyed by thread_id*
  (fixes the latent bug from `DEVELOPMENT.md §7.6`: the old
  `adapter._last_inbound_msg` was keyed by `channel_id`, so two
  concurrent threads in the same channel trampled each other).
- **Parent text cache** — small LRU around `get_message()` so the
  adapter does not re-fetch the same parent transcript every time a
  star-shaped thread receives a new reply.

Later PRs grow this module without changing the public surface PR 1
introduces:

- PR 2 wires `thread_id_of(msg)` into `SessionSource.thread_id` to
  enable shared sessions in groups.
- PR 3 adds `mark_engaged` / `is_engaged` / `record_outbound` /
  `is_bot_message` for the thread-memory mention gate shortcuts.

Thread root resolution is a synchronous one-liner (`thread_id_of`)
because Carbon Voice enforces flat replies — the Flutter client's
`Message.getTopLevelGuid()` returns `parent_message_id` or self
without walking, the send queue redirects any reply targeting a
non-top-level message back to its parent, and the backend rejects
depth-2+ replies with HTTP 400. See `DEVELOPMENT.md §4` and §7.4 for
the full chain of evidence.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .api import CarbonVoiceAPI
from .parse import extract_transcript, first_str

logger = logging.getLogger(__name__)

# Defaults sized for typical per-process working sets:
#  - ~1000 reply anchors covers very active workspaces without
#    bounded growth (one entry per active thread per process).
#  - ~128 parent texts is enough for star-shaped threads where many
#    replies share the same root; smaller because each entry carries
#    a full transcript string.
DEFAULT_MAX_REPLY_ANCHORS = 1000
DEFAULT_MAX_PARENT_TEXT = 128
# Thread-context cache (PR 4): per-thread formatted prefix that the adapter
# prepends on the first @mention so the agent has the prior thread history.
# 30-minute TTL is the working balance — long enough that a quick follow-up
# reuses the cached context (avoiding a second index + by-ids fetch), short
# enough that a long-quiet thread refetches before injecting stale state.
DEFAULT_MAX_THREAD_CONTEXT = 200
DEFAULT_THREAD_CONTEXT_TTL_S = 1800


@dataclass
class _ThreadContextEntry:
    """One cached thread-context fetch.

    ``content`` is the formatted prefix string the adapter will prepend to
    the inbound message text; ``fetched_at`` is monotonic seconds so the
    TTL check is robust against wall-clock jumps.
    """

    content: str
    fetched_at: float


class _LRUDict:
    """Tiny in-process LRU around ``collections.OrderedDict``.

    Insertion / access bump the entry to the most-recently-used slot;
    when ``max_size`` is exceeded the oldest entry is evicted. Used by
    `ConversationTracker` for bounded reply-anchor and parent-text
    caches.
    """

    __slots__ = ("_max_size", "_data")

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._data: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Any:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


class ConversationTracker:
    """Per-process per-conversation memory.

    The adapter delegates two state axes to this tracker in PR 1:

    1. **Reply anchors** — the message id we thread our outbound reply
       under, looked up by ``thread_id``. Without rekeying away from
       ``channel_id``, two parallel threads in the same channel
       overwrite each other's anchor.
    2. **Parent text cache** — `get_parent_text` mirrors the old
       ``adapter._resolve_parent_text`` but caches each fetched
       parent transcript in a small LRU.

    All sizes are tunable via constructor kwargs so tests can exercise
    eviction without thousands of inserts.
    """

    def __init__(
        self,
        api: Optional[CarbonVoiceAPI] = None,
        *,
        max_reply_anchors: int = DEFAULT_MAX_REPLY_ANCHORS,
        max_parent_text: int = DEFAULT_MAX_PARENT_TEXT,
        max_thread_context: int = DEFAULT_MAX_THREAD_CONTEXT,
        thread_context_ttl_s: int = DEFAULT_THREAD_CONTEXT_TTL_S,
    ):
        self._api = api
        self._reply_anchors = _LRUDict(max_reply_anchors)
        self._parent_text = _LRUDict(max_parent_text)
        self._thread_context = _LRUDict(max_thread_context)
        self._thread_context_ttl_s = thread_context_ttl_s

    # ── Thread resolution ───────────────────────────────────────────

    @staticmethod
    def thread_id_of(msg: Dict[str, Any]) -> Optional[str]:
        """Return the canonical thread root id for *msg*.

        For top-level messages this is the message's own id. For replies
        it is ``parent_message_id`` (which CV guarantees is the true
        root — see module docstring). Returns ``None`` only when the
        payload is malformed enough that neither field is present.
        """
        parent = first_str(
            msg.get("parent_message_id"),
            msg.get("parent_message_guid"),
        )
        if parent:
            return parent
        return first_str(msg.get("message_id"), msg.get("_id"))

    # ── Reply anchors (outbound threading) ──────────────────────────

    def get_reply_anchor(self, thread_id: Optional[str]) -> Optional[str]:
        """Return the message id we should thread the next reply under."""
        if not thread_id:
            return None
        return self._reply_anchors.get(thread_id)

    def set_reply_anchor(self, thread_id: str, message_id: str) -> None:
        """Record *message_id* as the next reply-target for *thread_id*."""
        if not thread_id or not message_id:
            return
        self._reply_anchors.set(thread_id, message_id)

    def clear_reply_anchor(self, thread_id: str) -> None:
        """Drop the anchor for *thread_id* (used on stale-anchor recovery)."""
        if not thread_id:
            return
        self._reply_anchors._data.pop(thread_id, None)

    # ── Parent transcript cache ─────────────────────────────────────

    # ── Thread context cache (PR 4) ─────────────────────────────────

    def get_cached_thread_context(
        self, thread_id: Optional[str]
    ) -> Optional[str]:
        """Return the cached thread-context prefix for *thread_id*.

        Returns ``None`` if not cached, or if the entry has aged past
        ``thread_context_ttl_s``. The TTL check uses monotonic time so
        wall-clock jumps don't make entries spuriously valid/invalid.
        """
        if not thread_id:
            return None
        entry = self._thread_context.get(thread_id)
        if entry is None:
            return None
        if (time.monotonic() - entry.fetched_at) > self._thread_context_ttl_s:
            # Expired — drop the entry so the LRU slot frees up next eviction
            self._thread_context._data.pop(thread_id, None)
            return None
        return entry.content

    def set_cached_thread_context(
        self, thread_id: str, content: str
    ) -> None:
        """Store the formatted thread-context prefix for *thread_id*."""
        if not thread_id:
            return
        self._thread_context.set(
            thread_id,
            _ThreadContextEntry(content=content, fetched_at=time.monotonic()),
        )

    def clear_cached_thread_context(self, thread_id: str) -> None:
        """Drop the cached entry for *thread_id* (forces a refetch next time)."""
        if not thread_id:
            return
        self._thread_context._data.pop(thread_id, None)

    # ── Parent transcript cache ─────────────────────────────────────

    async def get_parent_text(self, parent_id: Optional[str]) -> Optional[str]:
        """Return the cached or freshly fetched transcript of *parent_id*.

        Mirrors the prior ``adapter._resolve_parent_text`` behavior
        (failures degrade to ``None`` so threading still works without
        injecting parent context). Adds an LRU cache so star-shaped
        threads — many replies sharing a single parent — pay one fetch
        instead of N.
        """
        if not parent_id or self._api is None:
            return None
        cached = self._parent_text.get(parent_id)
        if cached is not None:
            # Cache hit. ``cached`` is "" when an earlier fetch found
            # the parent but its transcript was empty; we treat that as
            # "no useful context" and return None to the caller, but
            # keep the empty value cached so we don't re-fetch.
            return cached or None
        try:
            parent_msg = await self._api.get_message(parent_id)
        except Exception as exc:
            logger.debug(
                "carbonvoice: get_parent_text(%s) failed: %s", parent_id, exc
            )
            return None
        if not parent_msg:
            self._parent_text.set(parent_id, "")
            return None
        text = extract_transcript(parent_msg) or ""
        self._parent_text.set(parent_id, text)
        return text or None
