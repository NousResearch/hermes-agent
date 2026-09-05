"""WeCom native streaming mixin (``msgtype: stream`` via aibot_respond_msg): per-turn state, per-req_id
ack tracking (official replyStreamNonBlocking semantics), keep-alive heartbeat, Layer 2 stream rotation
(cross the absolute 10-min wall), finalize clock fallback, tri-state settlement, tool-progress overlay."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("plugins.platforms.wecom.adapter")

APP_CMD_RESPONSE = "aibot_respond_msg"

# Each reply stream lives ~10 minutes (the connection ping does NOT refresh it); afterwards
# 846608 (stream window) / 846604 (req_id window) mean the reply flow is dead. 846609 = ws
# lost its subscription. 6000 = finalize raced a newer frame (bubble already replaced: benign).
STREAM_EXPIRED_ERRCODE = 846608
STREAM_REQUEST_EXPIRED_ERRCODE = 846604
STREAM_NOT_SUBSCRIBED_ERRCODE = 846609
STREAM_VERSION_CONFLICT_ERRCODE = 6000
MAX_STREAM_CONTENT_LENGTH = 20480  # WeCom server-enforced byte limit per frame
# SDK queue is 100 frames per reqId; cap intermediates (openclaw uses 85) so finalize has room.
MAX_INTERMEDIATE_FRAMES = 85

# Two defences against the 10-min window (docs/wecom-stream-keepalive-*.md):
#   Layer 2 — clock rotation (always effective): the frame path AND an active timer both watch
#     StreamTurn.start_time; at STREAM_SAFE_DURATION_SECONDS (9.5 min, safely inside the wall) the
#     adapter seals the current bubble with finish=true and rotates to a FRESH stream_id (new bubble)
#     on the same req_id, so the answer keeps flowing across the wall (errcode 846608) instead of
#     freezing. Rotation runs REGARDLESS of keep-alive — keep-alive frames cannot refresh the absolute
#     window, so rotation is the only mechanism that survives a >10-min turn; the two coexist.
#   Layer 1 — keep-alive heartbeat (OFF by default; opt-in via config): every interval re-send the
#     accumulated text as finish=false to keep the bubble visibly updating WITHIN the window. Never a
#     placeholder (empty text skips). Held under turn.rotation_lock() so it cannot interleave a rotation.
STREAM_SAFE_DURATION_SECONDS = 570.0  # 9.5 min — Layer 2 rotation trigger, inside the 10-min wall.
STREAM_KEEPALIVE_INTERVAL_SECONDS = 120.0  # 2 min — Layer 1 heartbeat cadence
STREAM_KEEPALIVE_ENABLED_DEFAULT = False   # Layer 1 off unless config opts in
ROTATION_CHECK_INTERVAL_SECONDS = 30.0     # active rotation-check cadence when no frames are flowing
ROTATION_LEAD_SECONDS = 15.0               # rotate (safe_duration - lead) so the seal RTT lands in-window
ROTATION_CONTINUATION_SUFFIX = "\n\n---\n⏬⏬⏬"  # appended to the sealed OLD bubble; language-neutral marker


class WeComStreamExpiredError(RuntimeError):
    """Raised on errcode 846608/846604: the stream/req_id reply flow is dead; fall back to ``aibot_send_msg``.

    WeCom caps a stream session at 10 minutes from the first ``finish=false`` frame. This deadline is
    ABSOLUTE — keep-alive frames do NOT refresh it. Callers must rotate to a fresh stream BEFORE the
    deadline (Layer 2) or fall back to a proactive send for the remaining content."""

    def __init__(self, errcode: int = STREAM_EXPIRED_ERRCODE, errmsg: str = ""):
        super().__init__(f"WeCom stream expired (errcode={errcode}): {errmsg or 'no detail'}")
        self.errcode, self.errmsg = errcode, errmsg


class StreamFrameResult(Enum):
    """Tri-state return from ``send_stream_frame`` / ``_send_stream_frame_core``.

    Replaces the bare ``bool`` so the consumer can distinguish a confirmed delivery from an
    *indeterminate* one (frame sent, ACK not received).

    * ``DELIVERED`` — confirmed success (was ``True``).
    * ``INDETERMINATE`` — frame was sent but delivery could not be confirmed (ACK channel poisoned /
      fence timed-out). The consumer should mark ``_final_response_sent`` (don't retry/duplicate) but
      must NOT mark ``_final_content_delivered`` (delivery unconfirmed).
    * ``FAILED`` — definitive dispatch failure (was ``False``); consumer should roll back and fall
      through to the send() fallback.

    DELIVERED and INDETERMINATE are truthy (frame was sent — don't duplicate); FAILED is falsy."""
    DELIVERED = "delivered"
    INDETERMINATE = "indeterminate"
    FAILED = "failed"

    def __bool__(self) -> bool:
        return self is not StreamFrameResult.FAILED


@dataclass
class StreamSendOutcome:
    """Rich return of ``send_stream_frame`` / ``_send_stream_frame_inner``.

    Wraps the tri-state ``StreamFrameResult`` and signals a Layer 2 rotation back to the gateway so it
    can render the NEW bubble with only the incremental (post-split) text instead of the full cumulative
    buffer (which would repeat what the sealed old bubble already showed).

    Backward compatibility: ``__bool__`` proxies ``result`` (FAILED falsy) and ``value`` proxies
    ``result.value`` so the gateway's ``getattr(ok, "value", None) == "indeterminate"`` check still works.

    ``rotated`` is True when a rotation sealed the old bubble on (or before, for the deferred
    active-timer case) this call. The adapter does NOT report a split length — the gateway advances its
    own split offset to its clean seal point (``_native_committed_len``)."""
    result: StreamFrameResult
    rotated: bool = False

    def __bool__(self) -> bool:
        return bool(self.result)

    @property
    def value(self) -> str:
        return self.result.value


@dataclass
class ReplyFrame:
    """A reply frame awaiting its aibot_respond_msg ack (FIFO per req_id)."""
    body: Dict[str, Any]
    future: asyncio.Future
    is_final: bool = False
    sent_at: Optional[float] = None


class ReplyQueue:
    """Per-req_id pending-ack tracker: only ONE intermediate frame is in flight at a time; a frame
    arriving while an ack is pending COALESCES into ``coalesced_body`` (latest cumulative snapshot wins)
    and is flushed the moment that ack resolves. Finals drain the pending ack then wait for their own.

    Coalescing (not dropping) is load-bearing: intermediate frames carry the cumulative buffer, so a burst
    that lands during ack lag would otherwise be dropped and the visible bubble freezes on a stale prefix
    until the next frame happens to catch an idle ack — and a Layer 2 rotation would then seal that frozen
    bubble. Buffering the latest and flushing on ack keeps the bubble current with one frame in flight."""
    def __init__(self, req_id: str):
        self.req_id, self.pending_ack = req_id, None  # pending_ack: Optional[ReplyFrame]
        # Latest cumulative intermediate body buffered while an ack is pending (None = nothing waiting).
        self.coalesced_body: Optional[Dict[str, Any]] = None
        self.coalesce_count: int = 0  # frames folded into the current buffer (diagnostics)


class StreamTurn:
    """Per-turn stream state so concurrent messages never share a stream."""
    def __init__(self, chat_id: str, req_id: str):
        self.chat_id, self.req_id, self.stream_id = chat_id, req_id, f"stream_{uuid.uuid4().hex[:12]}"
        self.accumulated_text = ""
        self.finalized = self.seeded = self.expired = False  # seeded prevents a double seed (errcode 6000)
        self.start_time = time.monotonic()
        self.last_sent_content: str = ""  # content ACTUALLY sent; final frame must differ or WeCom drops it
        self._intermediate_frames_sent: int = 0
        # Per-turn asyncio TimerHandles — each MUST be cancelled on every turn-exit path.
        self.keepalive_handle: Optional[asyncio.TimerHandle] = None       # Layer 1 heartbeat
        self.rotation_check_handle: Optional[asyncio.TimerHandle] = None  # Layer 2 active timer
        # Pending rotation signal awaiting delivery to the gateway. Set True when an ACTIVE-timer
        # rotation seals the old bubble while NO frame-send is in flight (pure tool-call stretch), so the
        # gateway cannot learn about it from a synchronous return. The next _send_stream_frame_core call
        # drains this into its StreamSendOutcome.rotated (deferred by one frame).
        self.pending_rotation_signal: bool = False
        # Per-turn rotation lock (Layer 2 concurrency guard). The active rotation timer runs concurrently
        # with the frame-send path; both mutate stream_id/seeded/last_sent_content across awaits. This
        # lock makes each side's "check state -> act" critical section atomic. Created lazily because
        # __init__ may run outside a running event loop.
        self._rotation_lock: Optional[asyncio.Lock] = None

    def rotation_lock(self) -> asyncio.Lock:
        """Return the per-turn rotation lock, creating it lazily (binds to the loop on first use)."""
        if self._rotation_lock is None:
            self._rotation_lock = asyncio.Lock()
        return self._rotation_lock

    def rotate(self) -> None:
        """Swap in a fresh stream_id for a Layer 2 rotation: new bubble on the SAME req_id.

        A new ``stream_id`` (WeCom keys a bubble by stream id), the seed flag cleared so the next frame
        re-seeds, ``last_sent_content`` cleared so the first frame on the new stream is never dedup-skipped,
        and the intermediate-frame counter reset for a fresh budget. ``req_id`` and ``accumulated_text``
        are preserved. ``start_time`` is NOT reset here — the caller anchors it just before the new seed
        frame (matching when WeCom starts its 10-min countdown)."""
        self.stream_id = f"stream_{uuid.uuid4().hex[:12]}"
        self.seeded = False
        self.last_sent_content = ""
        self._intermediate_frames_sent = 0


def _stream_of(body: Dict[str, Any]) -> Dict[str, Any]:
    return body.get("stream", {}) if isinstance(body.get("stream"), dict) else {}


def _stream_desc(body: Dict[str, Any]) -> tuple:
    stream = _stream_of(body)
    return stream.get("id", "N/A"), stream.get("finish", "N/A")


def _elapsed(since: Optional[float]) -> float:
    return time.monotonic() - (since or time.monotonic())


class WeComStreamMixin:
    """Native streaming mixed into WeComAdapter (uses its ws transport, registries and ``_stream_*`` config)."""

    MAX_STREAM_CONTENT_LENGTH = MAX_STREAM_CONTENT_LENGTH
    _REPLY_ACK_TIMEOUT = 15.0  # official REPLY_SEND_TIMEOUT_MS; shorter widened the double-send race

    async def _send_reply_queued(self, reply_req_id: str, body: Dict[str, Any], *, is_final: bool = False, skip_if_pending: bool = False) -> Dict[str, Any]:
        """aibot_respond_msg with per-req_id ack tracking: is_final drains the pending ack then awaits its own;
        skip_if_pending COALESCES an intermediate frame while a prior ack is pending (buffers the latest
        cumulative body, flushed on ack via ``_resolve_reply_ack``) instead of dropping it — so the visible
        bubble never freezes on a stale prefix under ack lag."""
        self._require_ws()
        normalized = self._require_reply_req_id(reply_req_id)
        queue = self._reply_queues.setdefault(normalized, ReplyQueue(normalized))
        if skip_if_pending and queue.pending_ack is not None:
            # One frame in flight already — coalesce (buffer the latest cumulative snapshot). When the
            # pending ack resolves, _resolve_reply_ack flushes this so the newest content still reaches the
            # wire; dropping it here would freeze the bubble until an unrelated later frame.
            queue.coalesced_body = body
            queue.coalesce_count += 1
            return {"skipped": True, "errcode": 0, "errmsg": "coalesced"}
        if is_final and queue.pending_ack is not None:
            await self._drain_pending_ack(queue, normalized)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        frame = ReplyFrame(body=body, future=future, is_final=is_final, sent_at=time.monotonic())
        # Register BEFORE sending so a mid-send ack routes; re-attach `queue` because the drain
        # above may have let the intermediate ack pop it out of _reply_queues (orphan → timeout).
        self._reply_queues[normalized] = queue
        queue.pending_ack = frame
        logger.debug(
            "[%s] _send_reply_queued: req_id=%s is_final=%s skip_if_pending=%s stream_id=%s finish=%s content_len=%d", self.name, normalized, is_final, skip_if_pending, *_stream_desc(body), len(_stream_of(body).get("content", "") or ""),
        )
        try:
            await self._send_json({"cmd": APP_CMD_RESPONSE, "headers": {"req_id": normalized}, "body": body})
        except Exception:
            # Nobody awaits the future here — cancel it rather than log "exception never retrieved".
            self._release_pending(queue, normalized, frame)
            if not future.done():
                future.cancel()
            raise
        if not is_final:  # fire-and-forget; pending_ack stays registered so later frames can skip
            return {"errcode": 0, "errmsg": "sent_nonblocking"}
        try:
            return await asyncio.wait_for(future, timeout=self._REPLY_ACK_TIMEOUT)
        except asyncio.TimeoutError:
            # Bytes went out, ack is late — WeCom already rendered it; raising caused duplicates.
            logger.warning("[%s] Final frame ack timeout (req_id=%s) — treating as delivered (matches official wecom-openclaw-plugin behaviour). No fallback send.", self.name, normalized)
            return {"errcode": 0, "errmsg": "ack_timeout_assumed_delivered", "ack_pending": True}
        finally:
            self._release_pending(queue, normalized, frame)

    async def _drain_pending_ack(self, queue: ReplyQueue, req_id: str) -> None:
        """Before a final frame: wait (bounded) for the pending intermediate's ack, then clear it."""
        pending_frame = queue.pending_ack
        pending_desc = (self.name, req_id, *_stream_desc(pending_frame.body))
        logger.debug("[%s] _send_reply_queued: final waiting for pending ack drain — req_id=%s pending_stream_id=%s pending_finish=%s pending_sent_at=%.1fs_ago", *pending_desc, _elapsed(pending_frame.sent_at))
        try:
            await asyncio.wait_for(asyncio.shield(pending_frame.future), timeout=self._REPLY_ACK_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] Reply ack timeout waiting for pending (req_id=%s) — pending_stream_id=%s pending_finish=%s elapsed=%.1fs. Possible causes: ack cmd filtered, ack req_id mismatch, or WeCom did not ack.",
                *pending_desc, _elapsed(pending_frame.sent_at),
            )
        except Exception:
            pass
        queue.pending_ack = None  # resolved or timed out either way
        # The final frame supersedes any buffered intermediate — drop it so the flush-on-ack path below
        # never re-sends stale mid-stream content after the finalize.
        queue.coalesced_body = None
        queue.coalesce_count = 0

    def _release_pending(self, queue: ReplyQueue, req_id: str, frame: ReplyFrame) -> None:
        """Clear ``frame`` if it is still the pending ack; drop the queue once fully idle (no pending
        ack AND no coalesced body waiting to flush)."""
        if queue.pending_ack is frame:
            queue.pending_ack = None
        if queue.pending_ack is None and queue.coalesced_body is None:
            self._reply_queues.pop(req_id, None)

    def _resolve_reply_ack(self, req_id: str, payload: Dict[str, Any]) -> bool:
        """Resolve a pending reply ack. Returns True if handled."""
        queue = self._reply_queues.get(req_id)
        if queue is None or queue.pending_ack is None:
            return False
        frame = queue.pending_ack
        if not frame.future.done():
            _body = payload.get("body", {}) if isinstance(payload.get("body"), dict) else {}
            logger.debug("[%s] _resolve_reply_ack: resolved req_id=%s is_final=%s elapsed=%.2fs errcode=%s", self.name, req_id, frame.is_final, _elapsed(frame.sent_at), _body.get("errcode", "N/A"))
            frame.future.set_result(payload)
        queue.pending_ack = None
        # A frame coalesced while this ack was pending — flush the latest cumulative snapshot now that the
        # slot is free, so the newest content reaches the wire instead of freezing the bubble. Scheduled
        # (not awaited) to keep this ack-dispatch callback synchronous; the flush registers its own
        # pending_ack, so the next coalesce/flush cycle chains correctly.
        if queue.coalesced_body is not None:
            with contextlib.suppress(RuntimeError):  # no running loop (defensive)
                asyncio.get_running_loop().create_task(self._flush_coalesced(req_id))
            return True
        self._reply_queues.pop(req_id, None)
        return True

    async def _flush_coalesced(self, req_id: str) -> None:
        """Send the buffered latest intermediate frame after its predecessor's ack resolved.

        Registers the flushed frame as the new in-flight pending ack (one frame in flight at a time), so a
        further burst during THIS frame's ack re-coalesces and chains. A raise is swallowed: the frame is a
        cumulative snapshot the next delta (or the finalize) supersedes."""
        queue = self._reply_queues.get(req_id)
        if queue is None or queue.coalesced_body is None or queue.pending_ack is not None:
            return
        body, queue.coalesced_body, queue.coalesce_count = queue.coalesced_body, None, 0
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        frame = ReplyFrame(body=body, future=future, is_final=False, sent_at=time.monotonic())
        self._reply_queues[req_id] = queue
        queue.pending_ack = frame
        try:
            await self._send_json({"cmd": APP_CMD_RESPONSE, "headers": {"req_id": req_id}, "body": body})
        except Exception as exc:
            logger.debug("[%s] flush of coalesced frame failed (req_id=%s): %s", self.name, req_id, exc)
            self._release_pending(queue, req_id, frame)
            if not future.done():
                future.cancel()

    def _fail_reply_queues(self, error: Exception) -> None:
        for queue in list(self._reply_queues.values()):
            if queue.pending_ack and not queue.pending_ack.future.done():
                queue.pending_ack.future.set_exception(error)
            queue.coalesced_body = None
            queue.coalesce_count = 0
        self._reply_queues.clear()

    def _resolve_stream_req_id(self, chat_id: str, reply_to: Optional[str]) -> Optional[str]:
        """Explicit ``reply_to`` (cached message id) → last inbound req_id for the chat → None."""
        return self._reply_req_id_for_message(reply_to) or self._last_chat_req_ids.get(str(chat_id or "").strip()) or None

    # ── Per-turn timer lifecycle ─────────────────────────────────────────

    @staticmethod
    def _cancel_keepalive(turn: StreamTurn) -> None:
        handle, turn.keepalive_handle = turn.keepalive_handle, None
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    @staticmethod
    def _cancel_rotation_check(turn: StreamTurn) -> None:
        handle, turn.rotation_check_handle = turn.rotation_check_handle, None
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    def _retire_turn(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        """Single choke point for "turn is dead": cancel every timer, then drop it from the registry."""
        self._cancel_keepalive(turn)
        self._cancel_rotation_check(turn)
        self._stream_turns.pop(f"{turn.chat_id}:{turn_id or turn.req_id}", None)

    def _expire_turn(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        turn.expired = True
        self._retire_turn(turn, turn_id)
        self._stream_expired_chats.add(turn.chat_id)

    def _find_active_turn_for_chat(self, chat_id: str) -> Optional[StreamTurn]:
        return next((t for t in self._stream_turns.values() if t.chat_id == chat_id and not t.finalized), None)

    # ── Layer 1 keep-alive heartbeat ─────────────────────────────────────

    def _arm_keepalive(self, turn: StreamTurn, *, turn_id: Optional[str]) -> None:
        """Arm the keep-alive timer if enabled and not already armed (idempotent)."""
        if not self._stream_keepalive_enabled or turn.finalized or turn.expired or turn.keepalive_handle is not None:
            return
        try:
            turn.keepalive_handle = asyncio.get_running_loop().call_later(self._stream_keepalive_interval_seconds, self._on_keepalive_fire, turn, turn_id)
        except RuntimeError:
            pass

    def _on_keepalive_fire(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        turn.keepalive_handle = None
        if not (turn.finalized or turn.expired):
            try:
                asyncio.ensure_future(self._keepalive_send(turn, turn_id))
            except RuntimeError:
                pass

    async def _keepalive_send(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        """Re-send accumulated text as finish=false to refresh the visible bubble, then re-arm.

        Held under ``turn.rotation_lock()`` for the whole read-send-write: keep-alive and Layer 2 rotation
        are BOTH live at once, so a keep-alive frame must not interleave a rotation's seed/rotate and write
        stale content after rotate() cleared last_sent_content. Never a placeholder (empty text skips). On
        an un-seeded turn (a concurrent rotation just rotate()'d it) the tick skips and re-arms — sending
        content onto an un-seeded new stream would double-seed (errcode 6000). On 846604/846608 the turn is
        retired for Layer 2."""
        if turn.finalized or turn.expired or turn._intermediate_frames_sent >= MAX_INTERMEDIATE_FRAMES:
            return  # cap reached: no room for intermediates; let finalize / Layer 2 run
        async with turn.rotation_lock():
            if turn.finalized or turn.expired:
                return
            if not turn.seeded:
                # A concurrent rotation rotate()'d this turn; the new bubble is not seeded yet. Sending a
                # content frame now would push onto an un-seeded stream and can race the pending re-seed
                # into a double-seed (errcode 6000). Skip and re-arm.
                self._arm_keepalive(turn, turn_id=turn_id)
                return
            content = turn.accumulated_text or ""
            if not content.strip():
                self._arm_keepalive(turn, turn_id=turn_id)
                return
            try:
                await self._send_stream_reply(turn.req_id, turn.stream_id, content, finish=False)
            except WeComStreamExpiredError:
                self._expire_turn(turn, turn_id)
                return
            except Exception as exc:
                logger.debug("[%s] keep-alive send failed (chat=%s, turn=%s): %s", self.name, turn.chat_id, turn.stream_id, exc)
                self._arm_keepalive(turn, turn_id=turn_id)  # transient — retry next interval
                return
            turn.last_sent_content = content
        self._arm_keepalive(turn, turn_id=turn_id)

    # ── Layer 2 active rotation check ────────────────────────────────────
    # The passive Layer 2 check in _send_stream_frame_core only fires when a text delta produces a frame.
    # During long tool calls no deltas are produced, so the passive check never runs and the stream would
    # silently exceed the 10-min wall. The active check is a periodic timer that inspects stream age and
    # rotates independently of frame pushes.

    def _arm_rotation_check(self, turn: StreamTurn, *, turn_id: Optional[str]) -> None:
        """Arm the active rotation-check timer (idempotent). No-op until the bubble is seeded (no bubble
        to rotate / no live countdown yet)."""
        if turn.finalized or turn.expired or not turn.seeded or turn.rotation_check_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # Wake BEFORE the rotation threshold (safe_duration - lead), never after: the seal is a network
        # round-trip that must land inside the window. No +epsilon.
        stream_age = time.monotonic() - turn.start_time
        rotate_at = self._stream_safe_duration_seconds - self._rotation_lead_seconds
        remaining = rotate_at - stream_age
        delay = 0.1 if remaining <= 0 else min(self._rotation_check_interval_seconds, remaining)
        turn.rotation_check_handle = loop.call_later(delay, self._on_rotation_check_fire, turn, turn_id)

    def _on_rotation_check_fire(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        """Loop callback — check stream age and trigger rotation if the threshold is reached."""
        turn.rotation_check_handle = None
        if turn.finalized or turn.expired or not turn.seeded:
            return
        stream_age = time.monotonic() - turn.start_time
        rotate_at = self._stream_safe_duration_seconds - self._rotation_lead_seconds
        if stream_age >= rotate_at:
            logger.info(
                "[%s] Active rotation check: stream age %.0fs >= rotate-at %.0fs (safe %.0fs - lead %.0fs) "
                "for chat %s — rotating to a fresh bubble (Layer 2 active timer, no frames were pushing).",
                self.name, stream_age, rotate_at, self._stream_safe_duration_seconds, self._rotation_lead_seconds, turn.chat_id,
            )
            try:
                asyncio.ensure_future(self._rotation_check_execute(turn, turn_id))
            except RuntimeError:
                pass
        else:
            self._arm_rotation_check(turn, turn_id=turn_id)  # not yet — re-arm

    async def _rotation_check_execute(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        """Execute the active rotation under the per-turn lock, then re-arm for the fresh bubble.

        _rotate_stream (the locking wrapper) seals the old bubble + rotate()s the turn. The belt-and-
        suspenders re-arm no-ops while the new bubble is un-seeded (safe: no live countdown), and arms
        immediately if a concurrent frame re-seeded during the await. The seed block in
        _send_stream_frame_core also arms, so protection is guaranteed on every seed path."""
        if turn.finalized or turn.expired:
            return
        try:
            rotated = await self._rotate_stream(turn, turn_id)
        except Exception as exc:
            logger.warning("[%s] Active rotation check: _rotate_stream raised %s — turn will expire naturally.", self.name, exc)
            return
        if not rotated:
            return  # _rotate_stream already retired the turn on failure
        self._arm_rotation_check(turn, turn_id=turn_id)

    async def _rotate_stream(self, turn: StreamTurn, turn_id: Optional[str]) -> bool:
        """Locked rotation entrypoint — acquires the per-turn lock around seal-old + rotate().

        LOCKING CONTRACT: the ACTIVE timer path and external callers invoke THIS (it acquires
        ``turn.rotation_lock()``). The PASSIVE frame-send path ALREADY holds the lock and therefore calls
        ``_rotate_stream_locked`` DIRECTLY — calling this wrapper there would deadlock on the non-reentrant
        lock."""
        async with turn.rotation_lock():
            return await self._rotate_stream_locked(turn, turn_id)

    async def _rotate_stream_locked(self, turn: StreamTurn, turn_id: Optional[str]) -> bool:
        """Close the current stream and rotate the turn to a fresh bubble. Caller MUST hold the lock.

        Sends ``finish=true`` on the current stream so the existing bubble seals cleanly while still in the
        window, then rotate()s the turn (new stream_id, cleared seed flag) on the SAME req_id. Returns True
        when the old stream was sealed and the turn is ready to continue on a fresh bubble; False when the
        close failed (stream already dead) — the caller retires the turn and falls back to send()."""
        if not turn.seeded:
            # Another rotation (passive or active) already ran during an await interleave; no bubble to seal.
            logger.debug("[%s] _rotate_stream: skipping — turn %s already un-seeded (concurrent rotation likely completed first).", self.name, turn.stream_id)
            return False
        old_stream_id = turn.stream_id
        self._cancel_keepalive(turn)
        self._cancel_rotation_check(turn)
        # Seal the old bubble with the canonical BODY text only. accumulated_text is pure body (overlay
        # frames never update it, see the is_overlay_frame gate in _send_stream_frame_core) — NOT
        # last_sent_content, which mirrors the exact wire frame for dedup and may be a tool-progress
        # overlay ("⠹ 💻 Running… (Ns)"). Falling back to it would freeze that timer line permanently into
        # the sealed bubble whenever rotation fires before any body text exists (long pure-tool turn).
        close_text = (turn.accumulated_text or "") + ROTATION_CONTINUATION_SUFFIX
        if close_text and close_text == turn.last_sent_content:
            close_text = close_text + "​"  # zero-width space so the server never drops the seal
        try:
            await self._send_stream_reply(turn.req_id, old_stream_id, close_text, finish=True)
        except WeComStreamExpiredError:
            logger.info("[%s] Stream rotation: old stream %s already expired on close (chat=%s) — retiring turn, falling back to send.", self.name, old_stream_id, turn.chat_id)
            self._expire_turn(turn, turn_id)
            return False
        except Exception as exc:
            logger.warning("[%s] Stream rotation: failed to close old stream %s (chat=%s): %s — retiring turn, falling back to send.", self.name, old_stream_id, turn.chat_id, exc)
            self._retire_turn(turn, turn_id)
            return False
        turn.rotate()
        # Signal the gateway to split its cumulative buffer and send only incremental text on the fresh
        # bubble. The passive path drains this into its synchronous return; the active-timer path (no frame
        # in flight) leaves it set so the NEXT _send_stream_frame_core call reports it (deferred one frame).
        turn.pending_rotation_signal = True
        logger.info("[%s] Stream rotated for chat %s: %s -> %s (new bubble, req_id unchanged), continuing remaining content.", self.name, turn.chat_id, old_stream_id, turn.stream_id)
        return True

    # ── Frame send ───────────────────────────────────────────────────────

    @staticmethod
    def _truncate_stream_content(content: str, limit: int) -> str:
        """Truncate to ``limit`` UTF-8 bytes (WeCom caps frames by bytes, not codepoints)."""
        encoded = content.encode("utf-8")
        return content if len(encoded) <= limit else encoded[:limit].decode("utf-8", errors="ignore")

    async def _send_stream_reply(self, reply_req_id: str, stream_id: str, content: str, finish: bool = False) -> Dict[str, Any]:
        """Send one ``msgtype: "stream"`` frame: intermediates non-blocking/skip-if-pending, the final frame
        awaits its ack so 846608/6000 are detected. Raises WeComStreamExpiredError on expiry; propagates a
        ``settlement_indeterminate`` errmsg (frame sent, delivery unconfirmed) without raising."""
        truncated = self._truncate_stream_content(content or "", self.MAX_STREAM_CONTENT_LENGTH)
        if len(content or "") != len(truncated):
            logger.warning("[%s] Stream content truncated for stream_id=%s", self.name, stream_id)
        body: Dict[str, Any] = {"msgtype": "stream", "stream": {"id": stream_id, "finish": bool(finish), "content": truncated}}
        if not finish:
            return await self._send_reply_queued(reply_req_id, body, is_final=False, skip_if_pending=True)
        response = await self._send_reply_queued(reply_req_id, body, is_final=True, skip_if_pending=False)
        errcode = response.get("errcode", 0)
        if errcode in (STREAM_EXPIRED_ERRCODE, STREAM_REQUEST_EXPIRED_ERRCODE):
            raise WeComStreamExpiredError(errcode=errcode, errmsg=str(response.get("errmsg") or ""))
        if errcode == STREAM_VERSION_CONFLICT_ERRCODE:
            # Content is already on screen; raising would pop the turn and duplicate via send().
            logger.info("[%s] finalize hit errcode 6000 (version conflict) — bubble already replaced by a newer frame; treating as delivered.", self.name)
            return response
        if response.get("errmsg") == "settlement_indeterminate":
            # The final frame was sent but the ACK channel was poisoned and the fence timed out, so
            # delivery cannot be confirmed. Propagate as-is (errcode=0 passes _raise_for_wecom_error) — the
            # caller checks errmsg and avoids setting turn.finalized = True.
            logger.warning("[%s] finalize returned settlement_indeterminate — delivery unconfirmed for req_id=%s", self.name, reply_req_id)
            return response
        self._raise_for_wecom_error(response, "send stream reply")
        return response

    async def send_stream_frame(self, text: str, *, finalize: bool = False, chat_id: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> Union[StreamSendOutcome, StreamFrameResult]:
        """Gateway streaming entry point: first call seeds the turn, later calls push cumulative text,
        ``finalize=True`` closes it; ``turn_id`` kwarg keys concurrent turns.

        Returns a ``StreamSendOutcome`` wrapping the tri-state ``StreamFrameResult`` (DELIVERED /
        INDETERMINATE / FAILED) and a ``rotated`` flag; both proxy ``__bool__``/``value`` so a caller doing
        ``if ok:`` or ``getattr(ok, "value", None)`` still works. ``is_overlay_frame`` marks a transient
        tool-progress overlay (must not become the sealed body of record); ``body_text`` is the
        authoritative pure body for THIS bubble."""
        chat = (chat_id or "").strip()
        if not chat:
            logger.warning("[%s] send_stream_frame: chat_id required", self.name)
            return StreamFrameResult.FAILED
        turn_id = kwargs.get("turn_id")
        # Chat-level expiry only blocks NEW turn creation; a known turn_id may still finalize.
        if not turn_id and chat in self._stream_expired_chats:
            return StreamFrameResult.FAILED
        is_overlay_frame = bool(kwargs.get("is_overlay_frame", False))
        body_text = kwargs.get("body_text", None)
        inner = lambda: self._send_stream_frame_inner(  # noqa: E731
            text, chat=chat, reply_to=reply_to, finalize=finalize, turn_id=turn_id,
            is_overlay_frame=is_overlay_frame, body_text=body_text,
        )
        # Finalize counts toward 30/min → control lane; intermediates are unmetered (no queue).
        outcome = await self._enqueue_chat_send(chat, inner, is_control=True) if finalize else await inner()
        # Public contract: return the bare tri-state ``StreamFrameResult`` unless a rotation occurred, in
        # which case the ``StreamSendOutcome`` (carrying ``rotated=True``) is returned so the gateway can
        # advance its split offset. A non-rotating call returns the plain enum so callers doing
        # ``result is StreamFrameResult.X`` keep working; the consumer reads ``getattr(ok, "rotated", False)``
        # either way.
        return outcome if outcome.rotated else outcome.result

    def _locate_turn(self, chat: str, reply_to: Optional[str], finalize: bool, turn_id: Optional[str]) -> Optional[StreamTurn]:
        """Find or create the StreamTurn (None = unavailable); a turn locks to its creation req_id."""
        if turn_id:
            turn = self._stream_turns.get(f"{chat}:{turn_id}")
            if turn:
                return turn
            if finalize:  # never create a turn on finalize: caller must fall back, not seed+finish
                logger.debug("[%s] send_stream_frame: cannot finalize non-existent turn (turn_id=%s, chat=%s)", self.name, turn_id, chat)
                return None
        elif existing_turn := self._find_active_turn_for_chat(chat):  # direct callers without turn_id reuse the chat's active (unfinalized) turn
            logger.debug("[%s] send_stream_frame: reusing existing turn %s for chat %s", self.name, existing_turn.stream_id, chat)
            return existing_turn
        suffix = f" (turn_id={turn_id})" if turn_id else ""
        req_id = None if chat in self._stream_expired_chats else self._resolve_stream_req_id(chat, reply_to)
        if not req_id:
            why = "chat %s is expired, cannot create new turn%s" if chat in self._stream_expired_chats else "no req_id available for chat %s%s"
            logger.debug("[%s] send_stream_frame: " + why, self.name, chat, suffix)
            return None
        key = f"{chat}:{turn_id or req_id}"
        turn = (None if turn_id else self._stream_turns.get(key)) or StreamTurn(chat, req_id)
        self._stream_turns[key] = turn
        logger.debug("[%s] send_stream_frame: created new turn %s (%s) for chat %s", self.name, turn.stream_id, f"turn_id={turn_id}, req_id={req_id}" if turn_id else f"req_id={req_id}", chat)
        return turn

    async def _send_stream_frame_inner(self, text: str, *, chat: str, reply_to: Optional[str] = None, finalize: bool = False, turn_id: Optional[str] = None, is_overlay_frame: bool = False, body_text: Optional[str] = None) -> StreamSendOutcome:
        """Thin wrapper over ``_send_stream_frame_core`` that reports rotation to the gateway.

        Runs the core frame logic, then drains the operated turn's ``pending_rotation_signal`` into a
        ``StreamSendOutcome`` (set by _rotate_stream_locked for BOTH the passive path — this call rotated,
        reported synchronously — and the active-timer path — an earlier tick rotated with no frame in
        flight, reported on this next call, deferred by one frame)."""
        holder: dict = {}
        result = await self._send_stream_frame_core(
            text, chat=chat, reply_to=reply_to, finalize=finalize, turn_id=turn_id,
            _turn_holder=holder, is_overlay_frame=is_overlay_frame, body_text=body_text,
        )
        rotated = False
        turn = holder.get("turn")
        if turn is not None and turn.pending_rotation_signal:
            rotated = True
            turn.pending_rotation_signal = False  # drained — report once
        return StreamSendOutcome(result=result, rotated=rotated)

    async def _send_stream_frame_core(self, text: str, *, chat: str, reply_to: Optional[str] = None, finalize: bool = False, turn_id: Optional[str] = None, _turn_holder: Optional[dict] = None, is_overlay_frame: bool = False, body_text: Optional[str] = None) -> StreamFrameResult:
        """Core stream-frame logic with per-turn state, Layer 2 rotation, and overlay/body_text handling."""
        turn: Optional[StreamTurn] = None
        try:
            turn = self._locate_turn(chat, reply_to, finalize, turn_id)
            if turn is None or turn.expired:
                return StreamFrameResult.FAILED
            if _turn_holder is not None:
                _turn_holder["turn"] = turn

            # ── Layer 2 clock rotation + seed (LOCKED critical section) ───────
            # Hold the per-turn rotation lock across "check state -> rotate/seed -> mutate" so the ACTIVE
            # rotation timer cannot interleave here across an await. We already hold the lock, so the
            # passive rotation calls _rotate_stream_locked DIRECTLY (the locking wrapper would deadlock).
            _seed_only_return = False
            async with turn.rotation_lock():
                if turn.seeded and not turn.expired:
                    stream_age = time.monotonic() - turn.start_time
                    if stream_age >= self._stream_safe_duration_seconds:
                        logger.info("[%s] Stream age %.0fs >= safe duration %.0fs for chat %s — rotating to a fresh bubble (Layer 2). finalize=%s", self.name, stream_age, self._stream_safe_duration_seconds, chat, finalize)
                        rotated = await self._rotate_stream_locked(turn, turn_id)
                        if not rotated:
                            # Close failed — turn already retired; fall back to the consumer's send().
                            return StreamFrameResult.FAILED
                        # turn.rotate() cleared last_sent_content, so the seed/frame below is never
                        # dedup-skipped against the sealed old bubble.
                if not turn.seeded and not turn.finalized:
                    # Official THINKING_MESSAGE seed; `seeded` prevents a double seed (6000). Anchor the age
                    # clock BEFORE the send — WeCom starts its 10-min countdown on the first finish=false.
                    turn.start_time = time.monotonic()
                    await self._send_stream_reply(turn.req_id, turn.stream_id, "<think></think>", finish=False)
                    turn.seeded = True
                    self._arm_keepalive(turn, turn_id=turn_id)
                    self._arm_rotation_check(turn, turn_id=turn_id)
                    if not text and not finalize:
                        _seed_only_return = True  # consumer's explicit seed call — defer the return past the lock
            if _seed_only_return:
                return StreamFrameResult.DELIVERED

            # Defer the body of an intermediate frame that just rotated. A rotation is pending delivery to
            # the gateway (either THIS frame triggered it, or an earlier active-timer tick did and this is
            # the first frame to re-seed the fresh bubble). The gateway has not yet advanced its split
            # offset (it learns that from rotated=True this call reports), so the frame's text is still the
            # pre-rotation full slice — sending it would repeat the sealed prefix. Skip the body: the fresh
            # bubble keeps just its seed, and this frame's increment is carried by the NEXT frame's full
            # slice, cut at the correct seal point. Finalize is exempt (it must close the bubble).
            if turn.pending_rotation_signal and not finalize and turn.seeded:
                logger.debug("[%s] rotation pending — fresh bubble %s re-seeded, body deferred to next delta (turn=%s)", self.name, turn.stream_id, turn_id)
                return StreamFrameResult.DELIVERED

            if finalize:
                return await self._finalize_turn(turn, text, chat, turn_id)

            # Fire-and-forget intermediate. Overlay frames (body + "---" + tool-timer lines) are a
            # transient display layer the next pure-body frame overwrites — they MUST NOT touch
            # turn.accumulated_text, the canonical body used to seal the old bubble on rotation. The gateway
            # passes the pure body via body_text (post-rotation slice, no overlay / completed-tool history)
            # — the authoritative source for what the sealed bubble should show; use it directly rather than
            # the wire `text` (which may carry a timer line or persistent "✓ …" history).
            if body_text is not None:
                turn.accumulated_text = body_text
            elif not is_overlay_frame:
                turn.accumulated_text = text
            if turn._intermediate_frames_sent >= MAX_INTERMEDIATE_FRAMES or text == turn.last_sent_content:
                return StreamFrameResult.DELIVERED  # cap reached (finalize drains the rest) or nothing new
            await self._send_stream_reply(turn.req_id, turn.stream_id, text, finish=False)
            turn._intermediate_frames_sent += 1
            turn.last_sent_content = text
            return StreamFrameResult.DELIVERED
        except WeComStreamExpiredError:
            # Intermediates are overwritten by the next frame anyway; expiring here would duplicate.
            if not finalize:
                logger.info("[%s] Intermediate stream frame expired (errcode=%d) for chat %s — dropping frame, stream stays live", self.name, STREAM_EXPIRED_ERRCODE, chat)
                return StreamFrameResult.DELIVERED
            logger.info("[%s] Stream expired (errcode=%d) for chat %s — switching to proactive send", self.name, STREAM_EXPIRED_ERRCODE, chat)
            if turn is None:
                self._stream_expired_chats.add(chat)
            else:
                self._expire_turn(turn, turn_id)
            return StreamFrameResult.FAILED
        except Exception as exc:
            if not finalize:  # same intermediate/final split as above
                logger.info("[%s] Intermediate stream frame failed (chat=%s): %s — dropping frame, stream stays live", self.name, chat, exc)
                return StreamFrameResult.DELIVERED
            logger.warning("[%s] Stream frame failed (chat=%s): %s", self.name, chat, exc)
            if turn is not None:
                self._retire_turn(turn, turn_id)
            return StreamFrameResult.FAILED

    async def _finalize_turn(self, turn: StreamTurn, text: str, chat: str, turn_id: Optional[str]) -> StreamFrameResult:
        """Send the finish=true frame. Layer 2 rotation already ran up-front, so if the stream was near the
        wall the turn has been sealed+rotated and this finalize lands on the fresh bubble. Maps a
        ``settlement_indeterminate`` result to INDETERMINATE (frame sent, delivery unconfirmed) without
        marking the turn finalized."""
        self._cancel_keepalive(turn)
        self._cancel_rotation_check(turn)
        # A final frame identical to the last intermediate is silently dropped — differ via ZWSP.
        final_text = text + "​" if text and text == turn.last_sent_content else text
        response = await self._send_stream_reply(turn.req_id, turn.stream_id, final_text, finish=True)
        is_indeterminate = response.get("errmsg") == "settlement_indeterminate"
        if not is_indeterminate:
            turn.finalized = True
        self._stream_turns.pop(f"{chat}:{turn_id or turn.req_id}", None)
        return StreamFrameResult.INDETERMINATE if is_indeterminate else StreamFrameResult.DELIVERED

    def supports_native_streaming(self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Stream frames work in DMs and groups alike (groups just need a cached inbound req_id)."""
        del chat_type, metadata
        return True

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No-op: the stream consumer's seed frame triggers WeCom typing; repeated calls would open orphan streams."""
        del chat_id, metadata
