"""WeCom-specific stream consumer — uses WeCom stream API instead of edit.

Replaces GatewayStreamConsumer for WeCom platform.  WeCom doesn't support
editing sent messages, so the edit-based streaming used by Telegram/Discord
doesn't work here.  Instead we use WeCom's native stream message API which
supports progressive updates and <think> tag rendering.

Design follows OpenClaw's WeCom ws-monitor.js implementation.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("gateway.wecom_stream_consumer")

# Sentinel values
_DONE = object()
_NEW_SEGMENT = object()
_COMMENTARY = object()
_ERROR = object()

# WeCom stream constraints
MAX_INTERMEDIATE_STREAM_MESSAGES = 85  # SDK queue limit is 100, reserve headroom
STREAM_MAX_LIFETIME_SECONDS = 5 * 60   # Hard limit is 6 minutes, rotate at 5
STREAM_KEEPALIVE_INTERVAL_SECONDS = 4 * 60  # Send keepalive every 4 minutes
WAITING_MODEL_TICK_SECONDS = 1

# WeCom single-message length limit.  When a stream update would produce
# content longer than this, we rotate the stream *before* sending so each
# message stays self-contained.  This prevents the SDK from silently splitting
# a long message across multiple frames, which breaks <think> tag context and
# causes the client parser to mis-identify backticks as think-block markers.
STREAM_MAX_CONTENT_LENGTH = 3500

# Tag constants (constructed via chr to avoid self-matching in source)
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)


def _escape_think_tags(text: str) -> str:
    """Escape <think> and </think> in text content.

    Prevents nested or premature tag closure from breaking WeCom stream
    rendering when models emit literal tag text in their output.
    """
    text = text.replace(_THINK_OPEN, "&lt;think&gt;")
    text = text.replace(_THINK_CLOSE, "&lt;/think&gt;")
    return text


def _escape_for_think_block(text: str) -> str:
    """Escape content destined for inside a <think> block.

    Think tags are escaped.  Backticks are also replaced with \u02cb
    (modifier letter grave accent) because WeCom's stream parser
    misidentifies ` as a think-block delimiter even inside a properly
    closed <think> block.
    """
    text = _escape_think_tags(text)
    # Replace backticks to prevent WeCom parser from breaking think blocks.
    # \u02cb (modifier letter grave accent) looks like ` but isn't one.
    text = text.replace("`", "\u02cb")
    return text


def _escape_for_visible(text: str) -> str:
    """Escape content for the visible (post-think) section of a message.

    Think tags are escaped.  Backticks are left as-is for proper markdown
    code rendering — they are outside the <think> block so WeCom's parser
    won't misidentify them.
    """
    return _escape_think_tags(text)


def build_ws_stream_content(
    reasoning_text: str = "",
    visible_text: str = "",
    finish: bool = False,
    error_text: str = "",
) -> str:
    """Build WeCom stream content with <think> tags for reasoning.

    When reasoning_text is non-empty, wraps it in <think> tags so WeCom
    clients render it as a collapsible thinking block.

    Args:
        reasoning_text: Internal reasoning from the model.
        visible_text: The visible response text.
        finish: Whether to close the think block.
        error_text: Optional error message appended after think block.
    """
    nr = _escape_for_think_block((reasoning_text or "").strip())
    nv = _escape_for_visible((visible_text or "").strip())
    err = (error_text or "").strip()

    if not nr and not err:
        return nv

    if nr:
        should_close = finish or bool(nv) or bool(err)
        if should_close:
            think_block = _THINK_OPEN + nr + _THINK_CLOSE
        else:
            think_block = _THINK_OPEN + "\n" + nr
    else:
        think_block = ""

    parts = []
    if think_block:
        parts.append(think_block)
    if nv:
        parts.append(nv)
    if err:
        parts.append(f"⚠️ {err}")
    return "\n".join(parts) if parts else ""


def build_waiting_model_content(seconds: int) -> str:
    """Build '等待模型响应 Ns' content as a completed think block."""
    seconds = max(1, seconds)
    lines = [f"等待模型响应 {i}s" for i in range(1, seconds + 1)]
    return _THINK_OPEN + "\n".join(lines) + _THINK_CLOSE


class WeComStreamConsumer:
    """Async consumer that sends streamed tokens via WeCom stream API.

    Unlike GatewayStreamConsumer which edits messages (Telegram/Discord),
    this uses WeCom's native stream message type for progressive updates.

    Features:
    - Visible text streaming (逐字输出)
    - Reasoning token forwarding with <think> tags
    - "等待模型响应" waiting indicator before first tokens arrive
    - Stream lifecycle: keepalive, rotation (5min limit), message count limit,
      and content-length limit to prevent SDK message splitting.
    - Error handling: graceful fallback when API calls fail mid-stream
    """

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        reply_req_id: str,
        stream_id: str,
        config: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.adapter = adapter
        self.chat_id = chat_id
        self.reply_req_id = reply_req_id
        self.stream_id = stream_id
        self.metadata = metadata

        self._queue: queue.Queue = queue.Queue()
        self._accumulated_visible = ""
        self._reasoning_text = ""
        self._already_sent = False
        self._final_response_sent = False

        # Stream lifecycle
        self._stream_created_at = time.monotonic()
        self._stream_messages_sent = 0
        self._last_send_time = 0.0

        # Throttle intervals (seconds)
        self._reasoning_throttle = 0.8
        self._visible_throttle = 0.8
        self._last_reasoning_send = 0.0
        self._last_visible_send = 0.0
        self._pending_reasoning = False
        self._pending_visible = False

        # Waiting model indicator
        self._waiting_model_active = True
        self._waiting_model_seconds = 0

        # Error state
        self._error_text: str = ""

    # ------------------------------------------------------------------
    # Public API — thread-safe callbacks from agent worker thread
    # ------------------------------------------------------------------

    def on_delta(self, text: Optional[str]) -> None:
        """Called with each visible text delta from the agent."""
        if text is None:
            self._queue.put(("segment_break", None))
            return
        if text:
            self._queue.put(("visible", text))

    def on_reasoning(self, text: str) -> None:
        """Called with each reasoning token delta from the agent."""
        if text:
            self._queue.put(("reasoning", text))

    def on_commentary(self, text: str) -> None:
        """Called with completed interim assistant commentary."""
        if text:
            self._queue.put(("commentary", text))

    def on_segment_break(self) -> None:
        """Finalize current segment, start fresh message below tool output."""
        self._queue.put(("segment_break", None))

    def on_error(self, error_msg: str) -> None:
        """Called when the API call fails (connection error, timeout, etc.).

        Signals the consumer to close the stream with an error hint
        so the user isn't left staring at a silent think block.
        """
        if error_msg:
            self._error_text = error_msg
            self._queue.put(("error", error_msg))

    def finish(self) -> None:
        """Signal that the agent stream is complete."""
        self._queue.put(("done", None))

    # ------------------------------------------------------------------
    # Properties for gateway run.py compatibility
    # ------------------------------------------------------------------

    @property
    def already_sent(self) -> bool:
        return self._already_sent

    @property
    def final_response_sent(self) -> bool:
        return self._final_response_sent

    # ------------------------------------------------------------------
    # Main async loop
    # ------------------------------------------------------------------

    def _cancel_thinking_loop(self) -> None:
        """Cancel the adapter's pre-processing thinking loop if running."""
        adapter = self.adapter
        task = getattr(adapter, "_thinking_task", None)
        if task and not task.done():
            adapter._thinking_cancelled = True
            task.cancel()
            logger.debug("[WeComStream] Cancelled adapter thinking loop")

    def _should_rotate_for_length(self) -> bool:
        """Check if the next send would exceed the per-message length limit."""
        content = self._build_stream_content(finish=False)
        return len(content) >= STREAM_MAX_CONTENT_LENGTH

    async def run(self) -> None:
        """Drain the queue and send updates via WeCom stream API."""
        # Don't cancel the thinking loop yet — let it keep running until
        # we have actual reasoning/visible tokens to send.  The thinking
        # loop provides the "等待模型响应" indicator while the model is
        # loading; we cancel it only when we have real content to show.

        pending_reasoning = ""
        pending_visible = ""
        thinking_loop_cancelled = False

        try:
            while True:
                # Batch-drain all available items
                got_done = False
                got_segment_break = False
                commentary_text = None

                while True:
                    try:
                        kind, data = self._queue.get_nowait()
                        if kind == "done":
                            got_done = True
                            break
                        elif kind == "segment_break":
                            got_segment_break = True
                            break
                        elif kind == "commentary":
                            commentary_text = data
                            break
                        elif kind == "error":
                            self._error_text = data
                            got_done = True
                            break
                        elif kind == "reasoning":
                            if not thinking_loop_cancelled:
                                self._cancel_thinking_loop()
                                thinking_loop_cancelled = True
                            self._waiting_model_active = False
                            pending_reasoning += data
                            self._reasoning_text += data
                        elif kind == "visible":
                            if not thinking_loop_cancelled:
                                self._cancel_thinking_loop()
                                thinking_loop_cancelled = True
                            self._waiting_model_active = False
                            pending_visible += data
                            self._accumulated_visible += data
                    except queue.Empty:
                        break

                # Check stream lifecycle
                elapsed = time.monotonic() - self._stream_created_at
                if elapsed >= STREAM_MAX_LIFETIME_SECONDS:
                    await self._rotate_stream()
                    continue

                # Check message count limit
                if self._stream_messages_sent >= MAX_INTERMEDIATE_STREAM_MESSAGES:
                    await self._rotate_stream()
                    continue

                # Check content length — rotate BEFORE the SDK splits the message.
                # This is the primary defense against think-block context loss.
                if self._should_rotate_for_length():
                    await self._rotate_stream()
                    continue

                # Send reasoning tokens (throttled)
                if pending_reasoning:
                    now = time.monotonic()
                    if now - self._last_reasoning_send >= self._reasoning_throttle:
                        await self._send_update()
                        pending_reasoning = ""
                        self._last_reasoning_send = now
                    else:
                        self._pending_reasoning = True

                # Send visible text (throttled)
                if pending_visible:
                    now = time.monotonic()
                    if now - self._last_visible_send >= self._visible_throttle:
                        await self._send_update()
                        pending_visible = ""
                        self._last_visible_send = now
                    else:
                        self._pending_visible = True

                # Send keepalive if idle too long
                idle = time.monotonic() - self._last_send_time
                if idle >= STREAM_KEEPALIVE_INTERVAL_SECONDS:
                    content = self._build_stream_content(finish=False)
                    if content:
                        await self._send_raw(content, finish=False)

                # Done — send final content and close stream
                if got_done:
                    await self._send_final()
                    return

                # Commentary — send as a separate message
                if commentary_text:
                    try:
                        await self.adapter.send(
                            self.chat_id,
                            commentary_text,
                            metadata=self.metadata,
                        )
                    except Exception as e:
                        logger.error("[WeComStream] Commentary send failed: %s", e)

                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            if self._already_sent:
                self._final_response_sent = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_stream_content(self, finish: bool = False) -> str:
        """Build the stream content to send."""
        reasoning = self._reasoning_text
        visible = self._accumulated_visible
        return build_ws_stream_content(
            reasoning_text=reasoning,
            visible_text=visible,
            finish=finish,
            error_text=self._error_text,
        )

    def _build_waiting_model_content(self) -> str:
        """Build '等待模型响应' content."""
        return build_waiting_model_content(self._waiting_model_seconds)

    async def _send_raw(self, content: str, finish: bool = False) -> None:
        """Send a raw stream message via the adapter."""
        if not content.strip() and not finish:
            return
        try:
            await self.adapter._send_reply_stream(
                self.reply_req_id,
                content,
                stream_id=self.stream_id,
                finish=finish,
            )
            self._stream_messages_sent += 1
            self._last_send_time = time.monotonic()
            self._already_sent = True
        except Exception as e:
            logger.warning("[WeComStream] Stream send failed: %s", e)

    async def _send_update(self) -> None:
        """Send current accumulated content as a stream update."""
        content = self._build_stream_content(finish=False)
        if content:
            await self._send_raw(content, finish=False)

    async def _send_final(self) -> None:
        """Send final content and close the stream."""
        content = self._build_stream_content(finish=True)
        if content:
            await self._send_raw(content, finish=True)
        self._final_response_sent = True

    async def _rotate_stream(self) -> None:
        """Rotate stream before lifetime/message/content-length limit is hit.

        Finishes the current stream (closes any open <think> block) and starts
        a new one.  The new stream starts fresh — reasoning is cleared and
        visible text resets so the continuation message doesn't carry stale
        think-block context that could confuse the client parser.
        """
        # Finish old stream — close the <think> block if reasoning exists
        visible = self._accumulated_visible or "⏳ 处理中…"
        finish_text = build_ws_stream_content(
            reasoning_text=self._reasoning_text,
            visible_text=visible,
            finish=True,
        )
        try:
            await self._send_raw(finish_text, finish=True)
        except Exception:
            pass

        # Create new stream
        self.stream_id = self.adapter._new_req_id("stream")
        self._stream_created_at = time.monotonic()
        self._stream_messages_sent = 0
        self._last_send_time = time.monotonic()

        # Reset accumulators — new stream starts fresh so the next message
        # is self-contained with its own <think> block (if reasoning resumes).
        self._reasoning_text = ""
        self._accumulated_visible = ""

        logger.info(
            "[WeComStream] Stream rotated: new_id=%s",
            self.stream_id,
        )
