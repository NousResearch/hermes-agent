"""WeCom stream-protocol data types — StreamFrameResult, StreamTurn, error.

Extracted from ``adapter.py`` for bounded ownership.  These are pure data
classes / enums with no dependency on the adapter's runtime state; they can
be imported independently by tests and by ``stream_consumer.py``.

Public symbols (re-exported by ``adapter.py`` for backward compat):
- ``StreamFrameResult``
- ``WeComStreamExpiredError``
- ``StreamTurn``
- ``STREAM_EXPIRED_ERRCODE``
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ── Error-code constants used by WeComStreamExpiredError and the adapter ──────
STREAM_EXPIRED_ERRCODE = 846608  # 10 min from first frame — stream is dead


class StreamFrameResult(Enum):
    """Tri-state return from ``send_stream_frame`` / ``_send_stream_frame_inner``.

    Replaces the bare ``bool`` so the consumer can distinguish a confirmed
    delivery from an *indeterminate* one (frame sent, ACK not received).

    * ``DELIVERED`` — confirmed success (was ``True``).
    * ``INDETERMINATE`` — frame was sent but delivery could not be confirmed
      (ACK channel poisoned / fence timed-out).  The consumer should mark
      ``_final_response_sent`` (don't retry/duplicate) but must NOT mark
      ``_final_content_delivered`` (delivery unconfirmed).
    * ``FAILED`` — definitive dispatch failure (was ``False``); consumer
      should roll back and fall through to the send() fallback.
    """
    DELIVERED = "delivered"
    INDETERMINATE = "indeterminate"
    FAILED = "failed"

    # Allow truthiness checks for backward compat: DELIVERED and INDETERMINATE
    # are truthy (frame was sent — don't duplicate), FAILED is falsy.
    def __bool__(self) -> bool:
        return self is not StreamFrameResult.FAILED


@dataclass
class StreamSendOutcome:
    """Rich return of ``send_stream_frame`` / ``_send_stream_frame_inner``.

    Wraps the tri-state ``StreamFrameResult`` and signals a Layer 2 rotation
    back to the gateway so it can render the NEW bubble with only the
    incremental (post-split) text instead of the full cumulative buffer —
    which would repeat what the sealed old bubble already showed.

    Backward compatibility is preserved so the gateway's existing checks need
    no change:
    * ``__bool__`` proxies ``result`` so ``if ok:`` still works (FAILED falsy).
    * ``value`` proxies ``result.value`` so the ``_is_indeterminate`` check
      (``getattr(ok, "value", None) == "indeterminate"``) still works.

    ``rotated`` is True when a rotation sealed the old bubble on (or before,
    for the deferred active-timer case) this call.  The adapter does NOT report
    a split *length*: the sealed content is the gateway's cumulative text at
    seal time, but the adapter only sees the composed frame (which may carry a
    tool-progress overlay), so its length is not a clean ``_accumulated``
    coordinate.  The gateway instead advances its split offset to its OWN
    clean seal point — the ``_accumulated`` length as of the PREVIOUS committed
    frame (``_native_committed_len``), i.e. what the old bubble actually showed.
    (Using ``len(_accumulated)`` at observe-time would be the current length,
    which already includes this frame's own increment and would drop it — the
    off-by-one that silently lost a segment.)
    """
    result: StreamFrameResult
    rotated: bool = False

    def __bool__(self) -> bool:
        return bool(self.result)

    @property
    def value(self) -> str:
        return self.result.value


class WeComStreamExpiredError(RuntimeError):
    """Raised when WeCom returns errcode 846608 or 846604 (stream/req expired).

    WeCom's stream protocol caps a stream session at 10 minutes from the
    first ``finish=false`` frame. This deadline is ABSOLUTE — keep-alive
    (finish=false) frames do NOT refresh it. After the window the server
    refuses further updates with 846608 (stream update window) or 846604
    (req_id reply-request window) — including a late ``finish=true`` — and
    the reply flow is dead. Callers must either rotate to a fresh stream
    BEFORE the deadline (see the adapter's stream rotation) or fall back to a
    proactive ``aibot_send_msg`` to deliver the remaining content.
    """

    def __init__(self, errcode: int = STREAM_EXPIRED_ERRCODE, errmsg: str = ""):
        super().__init__(
            f"WeCom stream expired (errcode={errcode}): {errmsg or 'no detail'}"
        )
        self.errcode = errcode
        self.errmsg = errmsg


class StreamTurn:
    """Per-turn stream state to avoid global state conflicts.

    Each inbound message creates its own StreamTurn, ensuring concurrent
    messages don't interfere with each other's stream state.
    """
    def __init__(self, chat_id: str, req_id: str):
        self.chat_id = chat_id
        self.req_id = req_id
        self.stream_id = f"stream_{uuid.uuid4().hex[:12]}"
        self.accumulated_text = ""
        self.finalized = False
        self.seeded = False  # True after seed frame sent (prevents double seed)
        self.start_time = time.monotonic()
        self.expired = False
        # Track the last content that was ACTUALLY sent to WeCom (not skipped).
        # Used by finalize to detect duplicate content and avoid silent ack drops.
        self.last_sent_content: str = ""
        # Per-turn intermediate-frame counter (count-based cap at
        # MAX_INTERMEDIATE_FRAMES to leave room for the finalize frame).
        self._last_frame_sent_at: float = 0.0
        self._intermediate_frames_sent: int = 0
        # Idle flush handle — retained for _cancel_idle_flush() compatibility
        # (called in finalize/boundary paths; always None in fire-and-forget).
        self.idle_flush_handle: Optional[asyncio.TimerHandle] = None
        # Keep-alive handle (Layer 1) — set when the stream-level keep-alive
        # timer is armed.  Structurally identical to idle_flush_handle: a
        # per-turn asyncio TimerHandle that MUST be cancelled on every turn
        # exit path (finalize / expired / error / cleanup) to avoid a leaked
        # timer firing on a dead turn.  None when keep-alive is disabled or
        # the turn has no armed timer.
        self.keepalive_handle: Optional[asyncio.TimerHandle] = None
        # Active rotation check handle (Layer 2 active timer) — a periodic
        # asyncio TimerHandle that checks stream age independently of frame
        # pushes.  Ensures rotation fires even during long tool executions
        # where no text deltas produce frames.  MUST be cancelled on every
        # turn exit path (same as keepalive_handle / idle_flush_handle).
        self.rotation_check_handle: Optional[asyncio.TimerHandle] = None
        # Pending rotation signal awaiting delivery to the gateway.  Set True
        # when an ACTIVE-timer rotation seals the old bubble while NO frame-send
        # is in flight (pure tool-call stretch), so the gateway cannot learn
        # about the rotation from a synchronous return value.  The next
        # _send_stream_frame_inner call drains this into its returned
        # StreamSendOutcome.rotated (deferred by one frame).  The gateway then
        # takes its own len(_accumulated) as the split offset.
        self.pending_rotation_signal: bool = False
        # Per-turn rotation lock (Layer 2 concurrency guard).  The active
        # rotation timer runs _rotate_stream as an independent coroutine
        # concurrently with the frame-send path; both read/mutate stream_id,
        # seeded and last_sent_content across await points.  This lock makes
        # each side's "check stream state -> act on it" critical section
        # atomic so a rotation cannot interleave a frame-send (and vice
        # versa), preventing the old-bubble-unsealed + new-bubble-uncreated
        # double-loss.  Created lazily via rotation_lock() because
        # __init__ may run outside a running event loop.
        self._rotation_lock: Optional[asyncio.Lock] = None

    def rotation_lock(self) -> asyncio.Lock:
        """Return the per-turn rotation lock, creating it lazily.

        asyncio.Lock() binds to the running loop on first use; StreamTurn is
        sometimes constructed outside a loop, so we defer creation until the
        first async caller needs it (all callers run inside the adapter's
        event loop).
        """
        if self._rotation_lock is None:
            self._rotation_lock = asyncio.Lock()
        return self._rotation_lock

    def rotate(self) -> None:
        """Swap in a fresh stream_id and reset the age clock for rotation.

        WeCom binds an ABSOLUTE 10-minute deadline to a stream (from its first
        ``finish=false`` frame) that keep-alive cannot refresh.  When the turn
        nears that deadline the adapter finishes the current stream and calls
        this to open a *new* bubble on the SAME req_id for the remaining
        content: a new ``stream_id`` (WeCom keys a bubble by stream id, so a
        new id yields a new bubble), the age clock reset to now, and the seed
        flag cleared so the next frame re-seeds.

        ``req_id`` and ``accumulated_text`` are preserved — the reply channel
        is unchanged and the new bubble continues from the accumulated content.
        ``last_sent_content`` is cleared so the first frame on the new stream is
        never dedup-skipped against what the old bubble already showed, and the
        intermediate-frame counter is reset so the new bubble gets its own
        MAX_INTERMEDIATE_FRAMES budget.

        NOTE: ``start_time`` is NOT reset here.  The age clock for the new
        bubble is anchored by the caller just before the new seed frame is
        sent (``turn.start_time = time.monotonic()`` in the seed block of
        ``_send_stream_frame_inner``), matching the moment WeCom starts its
        10-minute countdown.  Setting it here would be too early — there may
        be a gap between rotate() and the next seed.
        """
        self.stream_id = f"stream_{uuid.uuid4().hex[:12]}"
        self.seeded = False
        self.last_sent_content = ""
        self._last_frame_sent_at = 0.0
        self._intermediate_frames_sent = 0
