"""WeCom native streaming mixin (``msgtype: stream`` via aibot_respond_msg): per-turn state, per-req_id
ack tracking (official replyStreamNonBlocking semantics), keep-alive heartbeat, finalize clock fallback."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("plugins.platforms.wecom.adapter")

APP_CMD_RESPONSE = "aibot_respond_msg"

# Each reply stream lives ~6 minutes (the connection ping does NOT refresh it); afterwards
# 846608 (stream window) / 846604 (req_id window) mean the reply flow is dead. 846609 = ws
# lost its subscription. 6000 = finalize raced a newer frame (bubble already replaced: benign).
STREAM_EXPIRED_ERRCODE = 846608
STREAM_REQUEST_EXPIRED_ERRCODE = 846604
STREAM_NOT_SUBSCRIBED_ERRCODE = 846609
STREAM_VERSION_CONFLICT_ERRCODE = 6000
MAX_STREAM_CONTENT_LENGTH = 20480  # WeCom server-enforced byte limit per frame
# SDK queue is 100 frames per reqId; cap intermediates (openclaw uses 85) so finalize has room.
MAX_INTERMEDIATE_FRAMES = 85

# Two defences against the 6-min window (docs/wecom-stream-keepalive-*.md): Layer 2 clock
# fallback (always on) declines finish=true past STREAM_SAFE_DURATION_SECONDS so the consumer's
# send() delivers; Layer 1 keep-alive (OFF by default) re-sends accumulated text as finish=false
# every interval — off because an extra frame widens the ack race double-send relies on.
STREAM_SAFE_DURATION_SECONDS = 330.0
STREAM_KEEPALIVE_INTERVAL_SECONDS = 120.0
STREAM_KEEPALIVE_ENABLED_DEFAULT = False


class WeComStreamExpiredError(RuntimeError):
    """Raised on errcode 846608/846604: the stream/req_id reply flow is dead; fall back to ``aibot_send_msg``."""

    def __init__(self, errcode: int = STREAM_EXPIRED_ERRCODE, errmsg: str = ""):
        super().__init__(f"WeCom stream expired (errcode={errcode}): {errmsg or 'no detail'}")
        self.errcode, self.errmsg = errcode, errmsg


@dataclass
class ReplyFrame:
    """A reply frame awaiting its aibot_respond_msg ack (FIFO per req_id)."""
    body: Dict[str, Any]
    future: asyncio.Future
    is_final: bool = False
    sent_at: Optional[float] = None


class ReplyQueue:
    """Per-req_id pending-ack tracker: intermediates skip while an ack is pending, finals wait."""
    def __init__(self, req_id: str):
        self.req_id, self.pending_ack = req_id, None  # pending_ack: Optional[ReplyFrame]


class StreamTurn:
    """Per-turn stream state so concurrent messages never share a stream."""
    def __init__(self, chat_id: str, req_id: str):
        self.chat_id, self.req_id, self.stream_id = chat_id, req_id, f"stream_{uuid.uuid4().hex[:12]}"
        self.accumulated_text = ""
        self.finalized = self.seeded = self.expired = False  # seeded prevents a double seed (errcode 6000)
        self.start_time = time.monotonic()
        self.last_sent_content: str = ""  # content ACTUALLY sent; final frame must differ or WeCom drops it
        self._intermediate_frames_sent: int = 0
        self.keepalive_handle: Optional[asyncio.TimerHandle] = None  # cancel on EVERY turn-exit path


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
        skip_if_pending returns ``{"skipped": True}`` while a prior ack is pending."""
        self._require_ws()
        normalized = self._require_reply_req_id(reply_req_id)
        queue = self._reply_queues.setdefault(normalized, ReplyQueue(normalized))
        if skip_if_pending and queue.pending_ack is not None:
            return {"skipped": True, "errcode": 0, "errmsg": "pending_ack"}
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

    def _release_pending(self, queue: ReplyQueue, req_id: str, frame: ReplyFrame) -> None:
        """Clear ``frame`` if it is still the pending ack; drop the queue once empty."""
        if queue.pending_ack is frame:
            queue.pending_ack = None
        if queue.pending_ack is None:
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
        self._release_pending(queue, req_id, frame)
        return True

    def _fail_reply_queues(self, error: Exception) -> None:
        for queue in list(self._reply_queues.values()):
            if queue.pending_ack and not queue.pending_ack.future.done():
                queue.pending_ack.future.set_exception(error)
        self._reply_queues.clear()

    def _resolve_stream_req_id(self, chat_id: str, reply_to: Optional[str]) -> Optional[str]:
        """Explicit ``reply_to`` (cached message id) → last inbound req_id for the chat → None."""
        return self._reply_req_id_for_message(reply_to) or self._last_chat_req_ids.get(str(chat_id or "").strip()) or None

    @staticmethod
    def _cancel_keepalive(turn: StreamTurn) -> None:
        handle, turn.keepalive_handle = turn.keepalive_handle, None
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    def _retire_turn(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        """Single choke point for "turn is dead": cancel the timer, then drop it from the registry."""
        self._cancel_keepalive(turn)
        self._stream_turns.pop(f"{turn.chat_id}:{turn_id or turn.req_id}", None)

    def _expire_turn(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        turn.expired = True
        self._retire_turn(turn, turn_id)
        self._stream_expired_chats.add(turn.chat_id)

    def _find_active_turn_for_chat(self, chat_id: str) -> Optional[StreamTurn]:
        return next((t for t in self._stream_turns.values() if t.chat_id == chat_id and not t.finalized), None)

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
        """Re-send accumulated text as finish=false to refresh the window, then re-arm. Never a placeholder
        (empty text skips); on 846604/846608 the turn is retired for Layer 2."""
        if turn.finalized or turn.expired or turn._intermediate_frames_sent >= MAX_INTERMEDIATE_FRAMES:
            return  # cap reached: no room for intermediates; let finalize / Layer 2 run
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

    @staticmethod
    def _truncate_stream_content(content: str, limit: int) -> str:
        """Truncate to ``limit`` UTF-8 bytes (WeCom caps frames by bytes, not codepoints)."""
        encoded = content.encode("utf-8")
        return content if len(encoded) <= limit else encoded[:limit].decode("utf-8", errors="ignore")

    async def _send_stream_reply(self, reply_req_id: str, stream_id: str, content: str, finish: bool = False) -> Dict[str, Any]:
        """Send one ``msgtype: "stream"`` frame: intermediates non-blocking/skip-if-pending, the final frame awaits
        its ack so 846608/6000 are detected. Raises WeComStreamExpiredError on expiry."""
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
        self._raise_for_wecom_error(response, "send stream reply")
        return response

    async def send_stream_frame(self, text: str, *, finalize: bool = False, chat_id: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> bool:
        """Gateway streaming entry point: first call seeds the turn, later calls push cumulative text, ``finalize=True``
        closes it; ``turn_id`` kwarg keys concurrent turns. Returns False when unavailable — caller falls back to send()."""
        chat = (chat_id or "").strip()
        if not chat:
            logger.warning("[%s] send_stream_frame: chat_id required", self.name)
            return False
        turn_id = kwargs.get("turn_id")
        # Chat-level expiry only blocks NEW turn creation; a known turn_id may still finalize.
        if not turn_id and chat in self._stream_expired_chats:
            return False
        inner = lambda: self._send_stream_frame_inner(text, chat=chat, reply_to=reply_to, finalize=finalize, turn_id=turn_id)  # noqa: E731
        # Finalize counts toward 30/min → control lane; intermediates are unmetered (no queue).
        return await self._enqueue_chat_send(chat, inner, is_control=True) if finalize else await inner()

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

    async def _finalize_turn(self, turn: StreamTurn, text: str, chat: str, turn_id: Optional[str]) -> bool:
        """Send the finish=true frame (or decline via the Layer 2 clock fallback)."""
        # Layer 2: an old stream would hit 846604/846608 on finish=true, so decline up front and let
        # send() deliver once. Skipped with Layer 1 on: the heartbeat refreshed the window.
        if not self._stream_keepalive_enabled:
            stream_age = time.monotonic() - turn.start_time
            if stream_age >= self._stream_safe_duration_seconds:
                logger.info(
                    "[%s] Stream age %.0fs >= safe duration %.0fs for chat %s — declining finalize frame, falling back to proactive send (Layer 2 clock fallback).",
                    self.name, stream_age, self._stream_safe_duration_seconds, chat,
                )
                self._expire_turn(turn, turn_id)
                return False
        self._cancel_keepalive(turn)
        # A final frame identical to the last intermediate is silently dropped — differ via ZWSP.
        final_text = text + "\u200b" if text and text == turn.last_sent_content else text
        await self._send_stream_reply(turn.req_id, turn.stream_id, final_text, finish=True)
        turn.finalized = True
        self._stream_turns.pop(f"{chat}:{turn_id or turn.req_id}", None)
        return True

    async def _send_stream_frame_inner(self, text: str, *, chat: str, reply_to: Optional[str] = None, finalize: bool = False, turn_id: Optional[str] = None) -> bool:
        turn: Optional[StreamTurn] = None
        try:
            turn = self._locate_turn(chat, reply_to, finalize, turn_id)
            if turn is None or turn.expired:
                return False
            if not turn.seeded and not turn.finalized:
                # Official THINKING_MESSAGE seed; `seeded` prevents a double seed (6000).
                await self._send_stream_reply(turn.req_id, turn.stream_id, "<think></think>", finish=False)
                turn.seeded = True
                self._arm_keepalive(turn, turn_id=turn_id)
                if not text and not finalize:
                    return True  # consumer's explicit seed call — nothing more to send
            if finalize:
                return await self._finalize_turn(turn, text, chat, turn_id)
            # Fire-and-forget: the gateway decides when to push (identity dedup in stream_consumer.py).
            turn.accumulated_text = text
            if turn._intermediate_frames_sent >= MAX_INTERMEDIATE_FRAMES or text == turn.last_sent_content:
                return True  # cap reached (finalize drains the rest) or nothing new
            await self._send_stream_reply(turn.req_id, turn.stream_id, text, finish=False)
            turn._intermediate_frames_sent += 1
            turn.last_sent_content = text
            return True
        except WeComStreamExpiredError:
            # Intermediates are overwritten by the next frame anyway; expiring here would duplicate.
            if not finalize:
                logger.info("[%s] Intermediate stream frame expired (errcode=%d) for chat %s — dropping frame, stream stays live", self.name, STREAM_EXPIRED_ERRCODE, chat)
                return True
            logger.info("[%s] Stream expired (errcode=%d) for chat %s — switching to proactive send", self.name, STREAM_EXPIRED_ERRCODE, chat)
            if turn is None:
                self._stream_expired_chats.add(chat)
            else:
                self._expire_turn(turn, turn_id)
        except Exception as exc:
            if not finalize:  # same intermediate/final split as above
                logger.info("[%s] Intermediate stream frame failed (chat=%s): %s — dropping frame, stream stays live", self.name, chat, exc)
                return True
            logger.warning("[%s] Stream frame failed (chat=%s): %s", self.name, chat, exc)
            if turn is not None:
                self._retire_turn(turn, turn_id)
        return False

    def supports_native_streaming(self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Stream frames work in DMs and groups alike (groups just need a cached inbound req_id)."""
        del chat_type, metadata
        return True

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No-op: the stream consumer's seed frame triggers WeCom typing; repeated calls would open orphan streams."""
        del chat_id, metadata
