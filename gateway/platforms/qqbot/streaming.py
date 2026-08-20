"""
QQ Bot C2C streaming state management.

Backs :class:`QQAdapter`'s streaming-reply implementation.  The QQ Official
Bot API exposes ``POST /v2/users/{openid}/stream_messages`` for private
(C2C) chats: the caller sends the first chunk without ``stream_msg_id``,
the response body carries the platform-assigned ``id`` (aka
``stream_msg_id``), and every subsequent chunk must reuse that id together
with a monotonically increasing ``index``.  When the response is done the
caller sends a final chunk with ``input_state=10`` (``DONE``).

Hermes' stream consumer expresses this lifecycle through the standard
adapter surface — a single ``send()`` for the first chunk followed by
``edit_message()`` for each subsequent update (``finalize=True`` for the
last).  Consumers reference the streaming reply solely via
``SendResult.message_id``.

To bridge those two protocols the adapter keeps an in-memory
:class:`StreamSession` per outgoing streaming reply.  The consumer sees an
opaque adapter-generated ``logical_id`` as ``SendResult.message_id`` and
uses it for every ``edit_message`` call.  Internally the adapter uses
:class:`StreamManager` to look up the session and translate the edit into a
``stream_messages`` call with the correct ``stream_msg_id`` / ``msg_seq`` /
``index`` triple.

The manager also enforces a bounded lifetime — sessions are cleaned up on
``finalize=True``, on an explicit ``drop``, on TTL expiry, or by an LRU
cap.  When the consumer hands back a ``logical_id`` for which no session
exists (e.g. after adapter restart or QQ's ~5 min passive-message window
expired), the adapter returns ``SendResult(success=False)`` and the
consumer falls back to a fresh ``send()`` naturally.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# QQ Bot stream_messages content_raw hard limit — the platform accepts
# larger payloads on regular POST /messages, but streaming replies are
# capped tighter.  Value chosen to align with openclaw's practical
# ceiling; keep well below the doc-stated maximum.
MAX_STREAM_CONTENT_LEN = 4096

# Default session TTL — QQ's passive-message window is ~5 min, so any
# session older than that will fail server-side anyway.  We give a little
# extra headroom for slow generation runs.
DEFAULT_SESSION_TTL_SECONDS = 600.0

# Bounded session table to prevent runaway growth if adapters somehow
# leak sessions (e.g. consumer crashes mid-stream without finalize).
DEFAULT_MAX_SESSIONS = 1000


@dataclass
class StreamSession:
    """In-memory state for a single QQ C2C streaming reply.

    A session is created on the adapter's first ``send()`` for a
    streaming-capable outbound reply and destroyed on finalize / TTL /
    LRU eviction.  All fields are managed by :class:`StreamManager` and
    the adapter — callers should treat instances as opaque.
    """

    # Adapter-generated id exposed to the stream consumer as
    # ``SendResult.message_id``.  Consumers echo this back on every
    # ``edit_message`` call so the adapter can look the session up.
    logical_id: str

    # QQ target user openid (C2C chats only — group / guild are not
    # supported by the stream_messages endpoint).
    openid: str

    # Inbound message id used as the ``msg_id`` (passive-reply id) AND
    # ``event_id`` for every stream_messages call in this session.
    # openclaw uses the inbound message id for both fields (see
    # ``outbound-dispatch.ts`` where ``eventId: event.messageId``), so
    # hermes matches that behaviour and avoids a separate event_id cache.
    passive_msg_id: str

    # Session-wide ``msg_seq`` — QQ requires the same value across all
    # chunks of one stream (openclaw ``streaming-c2c.ts`` reuses a single
    # ``this.msgSeq`` after the first chunk).  Generated lazily on first
    # send inside the adapter.
    msg_seq: int

    # QQ-assigned stream identifier.  ``None`` until the first
    # stream_messages POST returns; every subsequent call must include
    # this value.
    stream_msg_id: Optional[str] = None

    # Next ``index`` to send.  Starts at 0 for the first chunk and
    # increments after each successful send.
    next_index: int = 0

    # Highest ``index`` that has been successfully sent to QQ.  Used by
    # the adapter to defensively drop out-of-order edits (per the "if id
    # arrives out of order, ignore" simplification agreed for this
    # release).  ``-1`` = nothing sent yet.
    last_sent_index: int = -1

    # Sticky flag: True once we've sent ``input_state=10 (DONE)`` so
    # the adapter can defend against duplicate finalize calls.
    finalized: bool = False

    # The exact ``content_raw`` string QQ last accepted for this stream.
    # QQ enforces a "each chunk's full text MUST start with the previous
    # chunk's full text" invariant for ``input_mode=replace`` — any
    # violation is rejected as ``系统繁忙`` (HTTP 500) and permanently
    # breaks the session.  We keep the last-accepted text here so the
    # adapter can enforce that invariant even when the upstream stream
    # consumer's per-frame content isn't strictly monotonic (e.g. a
    # trailing typewriter cursor character that disappears on the next
    # frame, or an occasional short retry frame).  Empty string until
    # the first successful send.
    last_sent_content: str = ""

    # Monotonic timestamp at creation — used by the LRU/TTL sweeper.
    created_at: float = field(default_factory=time.monotonic)


class StreamManager:
    """Bounded LRU + TTL table of active :class:`StreamSession`s.

    Not thread-safe: the QQ adapter is asyncio-driven and every mutation
    happens on the single event loop.  Concurrent ``edit_message`` calls
    against the same ``logical_id`` are already serialised by the stream
    consumer (it awaits each edit before issuing the next), so no lock
    is required here.
    """

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        self._sessions: "OrderedDict[str, StreamSession]" = OrderedDict()
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Public API — used by QQAdapter
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        openid: str,
        passive_msg_id: str,
        msg_seq: int,
    ) -> StreamSession:
        """Create and register a new streaming session.

        Sweeps expired sessions first, then enforces the LRU cap by
        evicting the least-recently-used entry when full.  Returns the
        freshly registered session; caller mutates ``stream_msg_id`` /
        ``next_index`` / ``last_sent_index`` after each successful API
        call.
        """
        self._sweep_expired()
        while len(self._sessions) >= self._max_sessions:
            # popitem(last=False) drops the LRU entry.
            self._sessions.popitem(last=False)

        session = StreamSession(
            logical_id=uuid.uuid4().hex,
            openid=openid,
            passive_msg_id=passive_msg_id,
            msg_seq=msg_seq,
        )
        self._sessions[session.logical_id] = session
        return session

    def get(self, logical_id: str) -> Optional[StreamSession]:
        """Fetch a session and mark it as most-recently-used.

        Returns ``None`` when the session is unknown or has expired.
        Expired entries are removed as a side effect so callers don't
        keep hitting them.
        """
        session = self._sessions.get(logical_id)
        if session is None:
            return None
        if self._is_expired(session):
            self._sessions.pop(logical_id, None)
            return None
        # Refresh LRU ordering — recently-accessed sessions stay warm.
        self._sessions.move_to_end(logical_id)
        return session

    def drop(self, logical_id: str) -> None:
        """Remove a session from the table (idempotent)."""
        self._sessions.pop(logical_id, None)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_expired(self, session: StreamSession) -> bool:
        return (time.monotonic() - session.created_at) > self._ttl_seconds

    def _sweep_expired(self) -> None:
        # Snapshot the keys since we mutate during iteration.  Sessions
        # are stored in insertion order, so the first non-expired entry
        # marks the boundary — everything before it can go.  We still
        # iterate defensively in case the TTL clock is non-monotonic in
        # a mocked test environment.
        expired = [
            key for key, session in self._sessions.items() if self._is_expired(session)
        ]
        for key in expired:
            self._sessions.pop(key, None)
