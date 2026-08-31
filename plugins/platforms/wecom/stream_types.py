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
from enum import Enum
from typing import Optional


# ── Error-code constants used by WeComStreamExpiredError and the adapter ──────
STREAM_EXPIRED_ERRCODE = 846608  # >6 min without update — stream is dead


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


class WeComStreamExpiredError(RuntimeError):
    """Raised when WeCom returns errcode 846608 or 846604 (stream/req expired).

    WeCom's stream protocol caps a stream session at ~6 minutes from the
    first frame. After that window the server refuses further updates with
    846608 (stream update window) or 846604 (req_id reply-request window) and
    the reply flow is dead — callers must fall back to a proactive
    ``aibot_send_msg`` to deliver the remaining content.
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
