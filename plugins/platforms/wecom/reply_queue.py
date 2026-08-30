"""WeCom ACK/Queue Settlement — serial-coalesce reply queue machinery.

Extracted from ``adapter.py`` for bounded ownership.  The ``ReplyQueueMixin``
provides the per-req_id ack tracking, serial-with-coalesce intermediate frame
delivery, and queue lifecycle management.  The host class (WeComAdapter) must
supply ``_send_json(payload)``, ``_ws``, and ``name``.

Data classes ``ReplyFrame`` and ``ReplyQueue`` are public so tests and the
adapter can reference them by name.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# WeCom command constant (duplicated from adapter to keep this module
# importable without pulling in the full adapter constant set).
_APP_CMD_RESPONSE = "aibot_respond_msg"


@dataclass
class ReplyFrame:
    """A queued reply frame waiting to be sent via aibot_respond_msg.

    Used for ack tracking and FIFO ordering per req_id, aligning with
    the official WeCom SDK's replyStreamNonBlocking semantics.
    """
    body: Dict[str, Any]
    future: asyncio.Future
    is_final: bool = False
    sent_at: Optional[float] = None


class ReplyQueue:
    """Per-req_id pending ack tracker with serial-send + coalesce semantics.

    Ensures:
    - Only one intermediate frame in-flight at a time (serial sending)
    - New content while waiting coalesces into a single pending buffer
    - Final frames wait for pending ack before sending (fence)

    Aligned with official SDK's replyStreamNonBlocking semantics.
    """
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.pending_ack: Optional[ReplyFrame] = None
        # Serial-coalesce state
        self.coalesced_body: Optional[Dict[str, Any]] = None
        self.coalesce_count: int = 0


class ReplyQueueMixin:
    """Mixin providing WeCom ACK/queue settlement logic.

    **Host requirements** (must be present on ``self``):
    - ``_send_json(payload: dict) -> None``  (async)
    - ``_ws``   — the active websocket (has ``.closed`` bool)
    - ``name``  — adapter display name for log messages
    - ``_reply_queues: Dict[str, ReplyQueue]``  (initialised by host ``__init__``)
    """

    # ── Per-req_id Reply Queue (ack tracking) ────────────────────────────
    # Aligns with official SDK replyStreamNonBlocking:
    #   - intermediate frame: skip if pending ack on this req_id
    #   - final frame: wait for pending ack to drain before sending
    #   - ack timeout: 15 seconds
    #
    # Matches the official @wecom/wecom-openclaw-plugin's REPLY_SEND_TIMEOUT_MS
    # = 15_000. The prior 5s value was too aggressive for the bilibili WeCom
    # environment where ack > 5s is not rare on long replies (server-side queue
    # lag, WS jitter, concurrent replies on the same WS). A short window widens
    # the race where the final-frame ack is still in flight while the gateway's
    # normal final-send fires, producing duplicate messages
    # (see docs/rca-wecom-stream-final-ack-timeout-duplicate.md).

    _REPLY_ACK_TIMEOUT = 15.0

    async def _send_reply_queued(
        self,
        reply_req_id: str,
        body: Dict[str, Any],
        *,
        is_final: bool = False,
        skip_if_pending: bool = False,
    ) -> Dict[str, Any]:
        """Send a reply via aibot_respond_msg with per-req_id ack tracking.

        Args:
            reply_req_id: The inbound callback req_id to reply to.
            body: Reply body (msgtype: stream/markdown/...).
            is_final: If True, wait for any pending ack before sending.
            skip_if_pending: If True and a previous frame's ack is pending,
                return immediately with {"skipped": True}.

        Returns:
            Response dict from WeCom, or {"skipped": True} if skipped.
        """
        if not self._ws or self._ws.closed:
            raise RuntimeError("WeCom websocket is not connected")

        normalized = str(reply_req_id or "").strip()
        if not normalized:
            raise ValueError("reply_req_id is required")

        queue = self._reply_queues.get(normalized)
        if queue is None:
            queue = ReplyQueue(normalized)
            self._reply_queues[normalized] = queue

        # NonBlocking semantics: skip if a prior frame ack is pending
        if skip_if_pending and queue.pending_ack is not None:
            return {"skipped": True, "errcode": 0, "errmsg": "pending_ack"}

        # Final frame: wait for pending ack to drain first
        if is_final and queue.pending_ack is not None:
            pending_frame = queue.pending_ack
            fence_start = time.monotonic()
            _pending_stream = pending_frame.body.get("stream", {}) if isinstance(pending_frame.body.get("stream"), dict) else {}
            logger.debug(
                "[%s] _send_reply_queued: final waiting for pending ack drain — "
                "req_id=%s pending_stream_id=%s pending_finish=%s pending_sent_at=%.1fs_ago",
                self.name, normalized,
                _pending_stream.get("id", "N/A"),
                _pending_stream.get("finish", "N/A"),
                time.monotonic() - (pending_frame.sent_at or time.monotonic()),
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(pending_frame.future),
                    timeout=self._REPLY_ACK_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] Reply ack timeout waiting for pending (req_id=%s) — "
                    "pending_stream_id=%s pending_finish=%s elapsed=%.1fs. "
                    "Possible causes: ack cmd filtered, ack req_id mismatch, or WeCom did not ack.",
                    self.name, normalized,
                    _pending_stream.get("id", "N/A"),
                    _pending_stream.get("finish", "N/A"),
                    time.monotonic() - (pending_frame.sent_at or time.monotonic()),
                )
            except BaseException:
                # Catch CancelledError (BaseException in Python 3.9+) and
                # any other exception so the final path always proceeds to
                # send finish=true.
                pass
            fence_elapsed = time.monotonic() - fence_start
            logger.info(
                "[ACK-SERIAL] req_id=%s fence_wait elapsed=%.3fs",
                normalized, fence_elapsed,
            )
            # Clear pending regardless — either resolved or timed out
            queue.pending_ack = None
            # Also clear any coalesced content — final frame supersedes
            queue.coalesced_body = None
            queue.coalesce_count = 0

        # Create future for THIS frame's ack
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        frame = ReplyFrame(body=body, future=future, is_final=is_final)
        frame.sent_at = time.monotonic()

        # Register as pending BEFORE sending to avoid race:
        # If WeCom acks during _send_json await, _dispatch_payload needs
        # to find the pending frame to resolve it. Registering after would
        # miss the ack and timeout.
        #
        # Fix (orphan-queue race): re-attach `queue` to the dict before
        # registering pending_ack. A final frame shares the inbound req_id
        # with the intermediate frames; while it awaits the pending
        # intermediate ack to drain (the `is_final` branch above yields at
        # `await`), that intermediate ack can arrive and _resolve_reply_ack
        # pops the WHOLE queue out of self._reply_queues (the "cleanup empty
        # queue" pop). The local `queue` reference captured at the top is then
        # an ORPHAN — detached from the dict — so registering pending_ack on it
        # is invisible to _dispatch_payload, and the final frame's own ack
        # lands in Unrouted → 15s timeout. Writing the reference back here
        # closes that window: the final frame's ack can always be routed.
        self._reply_queues[normalized] = queue
        queue.pending_ack = frame

        # Diagnostic: log every frame send for ack tracking analysis
        _stream_info = body.get("stream", {}) if isinstance(body.get("stream"), dict) else {}
        logger.debug(
            "[%s] _send_reply_queued: req_id=%s is_final=%s skip_if_pending=%s "
            "stream_id=%s finish=%s content_len=%d",
            self.name, normalized, is_final, skip_if_pending,
            _stream_info.get("id", "N/A"),
            _stream_info.get("finish", "N/A"),
            len(_stream_info.get("content", "") or ""),
        )

        # Send the frame
        try:
            await self._send_json(
                {"cmd": _APP_CMD_RESPONSE, "headers": {"req_id": normalized}, "body": body}
            )
        except Exception as e:
            # Send failed — clear pending and reject future. The future has
            # no awaiter on the send-failure branch (we re-raise immediately),
            # so cancel it instead of setting an exception that would otherwise
            # be logged as "Future exception was never retrieved".
            if queue.pending_ack is frame:
                queue.pending_ack = None
                if not self._reply_queues.get(normalized) or queue.pending_ack is None:
                    self._reply_queues.pop(normalized, None)
            if not future.done():
                future.cancel()
            raise

        # For final frames: await the ack (blocking)
        if is_final:
            try:
                response = await asyncio.wait_for(future, timeout=self._REPLY_ACK_TIMEOUT)
                return response
            except asyncio.TimeoutError:
                # Final-frame ack timeout: WeCom received the frame (we wrote
                # the bytes successfully — _send_json above did not raise) but
                # the ack didn't return within the window.  In practice the
                # server has already rendered the message to the client; the
                # ack is delayed for unrelated reasons (server-side queue lag,
                # WS jitter, concurrent reply on the same WS).
                #
                # The official wecom-openclaw-plugin treats this case as an
                # error and surfaces it to its caller, which then *does not*
                # resend.  Hermes' prior behaviour — raising RuntimeError so
                # the upper layer falls back to a normal markdown send —
                # produced duplicate messages whenever WeCom *had* rendered
                # the streamed frame (see docs/rca-wecom-stream-final-ack-
                # timeout-duplicate.md).
                #
                # Aligning with the official plugin: log a warning and
                # synthesise a success-shaped response so the caller treats
                # the message as delivered.  The thinking-bubble is already
                # closed on the client side by the finish=true frame; if WeCom
                # never queued it (rare), the user sees no answer — same
                # outcome as the official plugin.
                logger.warning(
                    "[%s] Final frame ack timeout (req_id=%s) — treating as "
                    "delivered (matches official wecom-openclaw-plugin "
                    "behaviour). No fallback send.",
                    self.name, normalized,
                )
                return {
                    "errcode": 0,
                    "errmsg": "ack_timeout_assumed_delivered",
                    "ack_pending": True,
                }
            finally:
                if queue.pending_ack is frame:
                    queue.pending_ack = None
                # Cleanup empty queue
                if queue.pending_ack is None:
                    self._reply_queues.pop(normalized, None)
        else:
            # Intermediate frame: fire-and-forget (don't await ack)
            # But the pending_ack stays registered so subsequent frames can
            # check and skip. The ack will be resolved by _dispatch_payload.
            return {"errcode": 0, "errmsg": "sent_nonblocking"}

    def _resolve_reply_ack(self, req_id: str, payload: Dict[str, Any]) -> bool:
        """Resolve a pending reply ack. Returns True if handled."""
        queue = self._reply_queues.get(req_id)
        if queue is None or queue.pending_ack is None:
            return False
        frame = queue.pending_ack
        if not frame.future.done():
            _body = payload.get("body", {}) if isinstance(payload.get("body"), dict) else {}
            elapsed = time.monotonic() - (frame.sent_at or time.monotonic())
            logger.info(
                "[ACK-SERIAL] req_id=%s ack_received elapsed=%.3fs errcode=%s",
                req_id, elapsed, _body.get("errcode", "N/A"),
            )
            frame.future.set_result(payload)
        queue.pending_ack = None
        # Check for coalesced content to flush; if present, schedule send
        if queue.coalesced_body is not None:
            asyncio.get_running_loop().call_soon(self._flush_coalesced, req_id)
        else:
            # Cleanup empty queue
            self._reply_queues.pop(req_id, None)
        return True

    def _fail_reply_queues(self, error: Exception) -> None:
        """Fail all pending reply acks (called on disconnect/error)."""
        for queue in list(self._reply_queues.values()):
            if queue.pending_ack and not queue.pending_ack.future.done():
                queue.pending_ack.future.set_exception(error)
            # Clear coalesced buffer
            queue.coalesced_body = None
            queue.coalesce_count = 0
        self._reply_queues.clear()

    async def _send_intermediate_serial(
        self,
        normalized_req_id: str,
        queue: "ReplyQueue",
        body: Dict[str, Any],
    ) -> None:
        """Send an intermediate frame and register it as the in-flight pending ACK.

        Pure serial: waits for the ACK to arrive (resolved by
        ``_resolve_reply_ack``) before the slot is freed for the next frame.
        No intermediate timeout — if the ACK never arrives, the WebSocket
        heartbeat / reconnect detects the dead connection and
        ``_fail_reply_queues`` cleans up.  This eliminates the race where a
        late ACK for a timed-out frame falsely resolves a successor.
        """
        # Create future for THIS frame's ack
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        frame = ReplyFrame(body=body, future=future, is_final=False)
        frame.sent_at = time.monotonic()

        # Register pending_ack BEFORE sending (so _resolve_reply_ack
        # can route the ack if it arrives during the _send_json await).
        self._reply_queues[normalized_req_id] = queue
        queue.pending_ack = frame

        logger.info(
            "[ACK-SERIAL] req_id=%s frame_sent content_len=%d",
            normalized_req_id,
            len(body.get("stream", {}).get("content", "") or ""),
        )

        try:
            await self._send_json(
                {"cmd": _APP_CMD_RESPONSE, "headers": {"req_id": normalized_req_id}, "body": body}
            )
        except Exception:
            # Send failed — clear pending and cancel future
            if queue.pending_ack is frame:
                queue.pending_ack = None
                if not self._reply_queues.get(normalized_req_id) or queue.pending_ack is None:
                    self._reply_queues.pop(normalized_req_id, None)
            if not future.done():
                future.cancel()
            raise

    def _flush_coalesced(self, req_id: str) -> None:
        """Schedule sending coalesced content after an ACK resolves.

        Called from ``_resolve_reply_ack`` via ``call_soon`` so the ACK
        resolution path stays synchronous.  The actual send is an async task.
        """
        queue = self._reply_queues.get(req_id)
        if queue is None or queue.coalesced_body is None:
            return

        coalesced_body = queue.coalesced_body
        queue.coalesced_body = None
        queue.coalesce_count = 0

        async def _do_flush():
            try:
                await self._send_intermediate_serial(req_id, queue, coalesced_body)
            except Exception:
                logger.warning(
                    "[ACK-SERIAL] req_id=%s failed to flush coalesced frame",
                    req_id, exc_info=True,
                )

        asyncio.get_running_loop().create_task(_do_flush())
