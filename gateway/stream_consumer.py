"""Gateway streaming consumer — bridges sync agent callbacks to async platform delivery.

The agent fires stream_delta_callback(text) synchronously from its worker thread.
GatewayStreamConsumer:
  1. Receives deltas via on_delta() (thread-safe, sync)
  2. Queues them to an asyncio task via queue.Queue
  3. The async run() task buffers, rate-limits, and progressively edits
     a single message on the target platform

Design: Uses the edit transport (send initial message, then editMessageText).
This is universally supported across Telegram, Discord, and Slack.

Credit: jobless0x (#774, #1312), OutThisLife (#798), clicksingh (#697).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("gateway.stream_consumer")

# Sentinel to signal the stream is complete
_DONE = object()

# Sentinel to signal a tool boundary — finalize current message and start a
# new one so that subsequent text appears below tool progress messages.
_NEW_SEGMENT = object()

# Queue marker for a completed assistant commentary message emitted between
# API/tool iterations (for example: "I'll inspect the repo first.").
_COMMENTARY = object()


@dataclass
class StreamConsumerConfig:
    """Runtime config for a single stream consumer instance."""
    edit_interval: float = 1.0
    buffer_threshold: int = 40
    cursor: str = " ▉"
    buffer_only: bool = False
    # When >0, the final edit for a streamed response is delivered as a
    # fresh message if the original preview has been visible for at least
    # this many seconds.  This makes the platform's visible timestamp
    # reflect completion time instead of first-token time for long-running
    # responses (e.g. reasoning models that stream slowly).  Ported from
    # openclaw/openclaw#72038.  Default 0 = always edit in place (legacy
    # behavior).  The gateway enables this selectively per-platform.
    fresh_final_after_seconds: float = 0.0


class GatewayStreamConsumer:
    """Async consumer that progressively edits a platform message with streamed tokens.

    Usage::

        consumer = GatewayStreamConsumer(adapter, chat_id, config, metadata=metadata)
        # Pass consumer.on_delta as stream_delta_callback to AIAgent
        agent = AIAgent(..., stream_delta_callback=consumer.on_delta)
        # Start the consumer as an asyncio task
        task = asyncio.create_task(consumer.run())
        # ... run agent in thread pool ...
        consumer.finish()  # signal completion
        await task         # wait for final edit
    """

    # After this many consecutive flood-control failures, permanently disable
    # progressive edits for the remainder of the stream.
    _MAX_FLOOD_STRIKES = 3

    # Reasoning/thinking tags that models emit inline in content.
    # Must stay in sync with cli.py _OPEN_TAGS/_CLOSE_TAGS and
    # run_agent.py _strip_think_blocks() tag variants.
    _OPEN_THINK_TAGS = (
        "<REASONING_SCRATCHPAD>", "<think>", "<reasoning>",
        "<THINKING>", "<thinking>", "<thought>",
    )
    _CLOSE_THINK_TAGS = (
        "</REASONING_SCRATCHPAD>", "</think>", "</reasoning>",
        "</THINKING>", "</thinking>", "</thought>",
    )

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        config: Optional[StreamConsumerConfig] = None,
        metadata: Optional[dict] = None,
        on_new_message: Optional[callable] = None,
    ):
        self.adapter = adapter
        self.chat_id = chat_id
        self.cfg = config or StreamConsumerConfig()
        self.metadata = metadata
        # Fired whenever a fresh content bubble is created on the platform
        # (first-send of a new message, commentary, overflow chunk, or
        # fallback continuation). The gateway uses this to linearize the
        # tool-progress bubble: when content resumes after a tool batch,
        # the next tool.started should open a NEW progress bubble below
        # the content, not edit the old bubble above it.
        # Called with no arguments. Exceptions are swallowed.
        self._on_new_message = on_new_message
        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        self._message_id: Optional[str] = None
        # Wall-clock timestamp (time.monotonic) when ``_message_id`` was
        # first assigned from a successful first-send.  Used by the
        # fresh-final logic to detect long-lived previews whose edit
        # timestamps would be stale by completion time.  Ported from
        # openclaw/openclaw#72038.
        self._message_created_ts: Optional[float] = None
        self._already_sent = False
        self._edit_supported = True  # Disabled when progressive edits are no longer usable
        self._last_edit_time = 0.0
        self._last_sent_text = ""   # Track last-sent text to skip redundant edits
        self._fallback_final_send = False
        self._fallback_prefix = ""
        self._flood_strikes = 0         # Consecutive flood-control edit failures
        self._current_edit_interval = self.cfg.edit_interval  # Adaptive backoff
        self._final_response_sent = False
        # Cache adapter lifecycle capability: only platforms that need an
        # explicit finalize call (e.g. DingTalk AI Cards) force us to make
        # a redundant final edit.  Everyone else keeps the fast path.
        # Use ``is True`` (not ``bool(...)``) so MagicMock attribute access
        # in tests doesn't incorrectly enable this path.
        self._adapter_requires_finalize: bool = (
            getattr(adapter, "REQUIRES_EDIT_FINALIZE", False) is True
        )

        # Think-block filter state (mirrors CLI's _stream_delta tag suppression)
        self._in_think_block = False
        self._think_buffer = ""

    @property
    def already_sent(self) -> bool:
        """True if at least one message was sent or edited during the run."""
        return self._already_sent

    @property
    def final_response_sent(self) -> bool:
        """True when the stream consumer delivered the final assistant reply."""
        return self._final_response_sent

    def on_segment_break(self) -> None:
        """Finalize the current stream segment and start a fresh message."""
        self._queue.put(_NEW_SEGMENT)

    def on_commentary(self, text: str) -> None:
        """Queue a completed interim assistant commentary message."""
        if text:
            self._queue.put((_COMMENTARY, text))

    def _notify_new_message(self) -> None:
        """Fire the on_new_message callback, swallowing any errors."""
        cb = self._on_new_message
        if cb is None:
            return
        try:
            cb()
        except Exception:
            logger.debug("on_new_message callback error", exc_info=True)

    def _reset_segment_state(self, *, preserve_no_edit: bool = False) -> None:
        if preserve_no_edit and self._message_id == "__no_edit__":
            return
        self._message_id = None
        self._message_created_ts = None
        self._accumulated = ""
        self._last_sent_text = ""
        self._fallback_final_send = False
        self._fallback_prefix = ""

    def on_delta(self, text: str) -> None:
        """Thread-safe callback — called from the agent's worker thread.

        When *text* is ``None``, signals a tool boundary: the current message
        is finalized and subsequent text will be sent as a new message so it
        appears below any tool-progress messages the gateway sent in between.
        """
        if text:
            self._queue.put(text)
        elif text is None:
            self.on_segment_break()

    def finish(self) -> None:
        """Signal that the stream is complete."""
        self._queue.put(_DONE)

    # ── Think-block filtering ────────────────────────────────────────
    # Models like MiniMax emit inline <think>...</think> blocks in their
    # content.  The CLI's _stream_delta suppresses these via a state
    # machine; we do the same here so gateway users never see raw
    # reasoning tags.  The agent also strips them from the final
    # response (run_agent.py _strip_think_blocks), but the stream
    # consumer sends intermediate edits before that stripping happens.

    def _filter_and_accumulate(self, text: str) -> None:
        """Add a text delta to the accumulated buffer, suppressing think blocks.

        Uses a state machine that tracks whether we are inside a
        reasoning/thinking block.  Text inside such blocks is silently
        discarded.  Partial tags at buffer boundaries are held back in
        ``_think_buffer`` until enough characters arrive to decide.
        """
        buf = self._think_buffer + text
        self._think_buffer = ""

        while buf:
            if self._in_think_block:
                # Look for the earliest closing tag
                best_idx = -1
                best_len = 0
                for tag in self._CLOSE_THINK_TAGS:
                    idx = buf.find(tag)
                    if idx != -1 and (best_idx == -1 or idx < best_idx):
                        best_idx = idx
                        best_len = len(tag)

                if best_len:
                    # Found closing tag — discard block, process remainder
                    self._in_think_block = False
                    buf = buf[best_idx + best_len:]
                else:
                    # No closing tag yet — hold tail that could be a
                    # partial closing tag prefix, discard the rest.
                    max_tag = max(len(t) for t in self._CLOSE_THINK_TAGS)
                    self._think_buffer = buf[-max_tag:] if len(buf) > max_tag else buf
                    return
            else:
                # Look for earliest opening tag at a block boundary
                # (start of text / preceded by newline + optional whitespace).
                # This prevents false positives when models *mention* tags
                # in prose (e.g. "the <think> tag is used for…").
                best_idx = -1
                best_len = 0
                for tag in self._OPEN_THINK_TAGS:
                    search_start = 0
                    while True:
                        idx = buf.find(tag, search_start)
                        if idx == -1:
                            break
                        # Block-boundary check (mirrors cli.py logic)
                        if idx == 0:
                            is_boundary = (
                                not self._accumulated
                                or self._accumulated.endswith("\n")
                            )
                        else:
                            preceding = buf[:idx]
                            last_nl = preceding.rfind("\n")
                            if last_nl == -1:
                                is_boundary = (
                                    (not self._accumulated
                                     or self._accumulated.endswith("\n"))
                                    and preceding.strip() == ""
                                )
                            else:
                                is_boundary = preceding[last_nl + 1:].strip() == ""

                        if is_boundary and (best_idx == -1 or idx < best_idx):
                            best_idx = idx
                            best_len = len(tag)
                            break  # first boundary hit for this tag is enough
                        search_start = idx + 1

                if best_len:
                    # Emit text before the tag, enter think block
                    self._accumulated += buf[:best_idx]
                    self._in_think_block = True
                    buf = buf[best_idx + best_len:]
                else:
                    # No opening tag — check for a partial tag at the tail
                    held_back = 0
                    for tag in self._OPEN_THINK_TAGS:
                        for i in range(1, len(tag)):
                            if buf.endswith(tag[:i]) and i > held_back:
                                held_back = i
                    if held_back:
                        self._accumulated += buf[:-held_back]
                        self._think_buffer = buf[-held_back:]
                    else:
                        self._accumulated += buf
                    return

    def _flush_think_buffer(self) -> None:
        """Flush any held-back partial-tag buffer into accumulated text.

        Called when the stream ends (got_done) so that partial text that
        was held back waiting for a possible opening tag is not lost.
        """
        if self._think_buffer and not self._in_think_block:
            self._accumulated += self._think_buffer
            self._think_buffer = ""

    async def run(self) -> None:
        """Async task that drains the queue and edits the platform message."""
        # Platform message length limit — leave room for cursor + formatting
        _raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        _safe_limit = max(500, _raw_limit - len(self.cfg.cursor) - 100)

        try:
            while True:
                # Drain all available items from the queue
                got_done = False
                got_segment_break = False
                commentary_text = None
                while True:
                    try:
                        item = self._queue.get_nowait()
                        if item is _DONE:
                            got_done = True
                            break
                        if item is _NEW_SEGMENT:
                            got_segment_break = True
                            break
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _COMMENTARY:
                            commentary_text = item[1]
                            break
                        self._filter_and_accumulate(item)
                    except queue.Empty:
                        break

                # Flush any held-back partial-tag buffer on stream end
                # so trailing text that was waiting for a potential open
                # tag is not lost.
                if got_done:
                    self._flush_think_buffer()

                # Decide whether to flush an edit
                now = time.monotonic()
                elapsed = now - self._last_edit_time
                should_edit = (
                    got_done
                    or got_segment_break
                    or commentary_text is not None
                )
                if not self.cfg.buffer_only:
                    should_edit = should_edit or (
                        (elapsed >= self._current_edit_interval
                            and self._accumulated)
                        or len(self._accumulated) >= self.cfg.buffer_threshold
                    )

                current_update_visible = False
                if should_edit and self._accumulated:
                    # Split overflow: if accumulated text exceeds the platform
                    # limit, split into properly sized chunks.
                    if (
                        len(self._accumulated) > _safe_limit
                        and self._message_id is None
                    ):
                        # No existing message to edit (first message or after a
                        # segment break).  Use truncate_message — the same
                        # helper the non-streaming path uses — to split with
                        # proper word/code-fence boundaries and chunk
                        # indicators like "(1/2)".
                        chunks = self.adapter.truncate_message(
                            self._accumulated, _safe_limit
                        )
                        for chunk in chunks:
                            await self._send_new_chunk(chunk, self._message_id)
                        self._accumulated = ""
                        self._last_sent_text = ""
                        self._last_edit_time = time.monotonic()
                        if got_done:
                            self._final_response_sent = self._already_sent
                            return
                        if got_segment_break:
                            self._message_id = None
                            self._fallback_final_send = False
                            self._fallback_prefix = ""
                        continue

                    # Existing message: edit it with the first chunk, then
                    # start a new message for the overflow remainder.
                    while (
                        len(self._accumulated) > _safe_limit
                        and self._message_id is not None
                        and self._edit_supported
                    ):
                        split_at = self._accumulated.rfind("\n", 0, _safe_limit)
                        if split_at < _safe_limit // 2:
                            split_at = _safe_limit
                        chunk = self._accumulated[:split_at]
                        ok = await self._send_or_edit(chunk)
                        if self._fallback_final_send or not ok:
                            # Edit failed (or backed off due to flood control)
                            # while attempting to split an oversized message.
                            # Keep the full accumulated text intact so the
                            # fallback final-send path can deliver the remaining
                            # continuation without dropping content.
                            break
                        self._accumulated = self._accumulated[split_at:].lstrip("\n")
                        self._message_id = None
                        self._last_sent_text = ""

                    display_text = self._accumulated
                    if not got_done and not got_segment_break and commentary_text is None:
                        display_text += self.cfg.cursor

                    # Segment break: finalize the current message so platforms
                    # that need explicit closure (e.g. DingTalk AI Cards) don't
                    # leave the previous segment stuck in a loading state when
                    # the next segment (tool progress, next chunk) creates a
                    # new message below it.  got_done has its own finalize
                    # path below so we don't finalize here for it.
                    current_update_visible = await self._send_or_edit(
                        display_text,
                        finalize=(got_done or got_segment_break),
                    )
                    self._last_edit_time = time.monotonic()

                if got_done:
                    # Final edit without cursor. If progressive editing failed
                    # mid-stream, send a single continuation/fallback message
                    # here instead of letting the base gateway path send the
                    # full response again.
                    if self._accumulated:
                        if self._fallback_final_send:
                            await self._send_fallback_final(self._accumulated)
                        elif (
                            current_update_visible
                            and not self._adapter_requires_finalize
                        ):
                            # Mid-stream edit above already delivered the
                            # final accumulated content.  Skip the redundant
                            # final edit — but only for adapters that don't
                            # need an explicit finalize signal.
                            self._final_response_sent = True
                        elif self._message_id:
                            # Either the mid-stream edit didn't run (no
                            # visible update this tick) OR the adapter needs
                            # explicit finalize=True to close the stream.
                            self._final_response_sent = await self._send_or_edit(
                                self._accumulated, finalize=True,
                            )
                        elif not self._already_sent:
                            self._final_response_sent = await self._send_or_edit(self._accumulated)
                    return

                if commentary_text is not None:
                    self._reset_segment_state()
                    await self._send_commentary(commentary_text)
                    self._last_edit_time = time.monotonic()
                    self._reset_segment_state()

                # Tool boundary: reset message state so the next text chunk
                # creates a fresh message below any tool-progress messages.
                #
                # Exception: when _message_id is "__no_edit__" the platform
                # never returned a real message ID (e.g. Signal, webhook with
                # github_comment delivery).  Resetting to None would re-enter
                # the "first send" path on every tool boundary and post one
                # platform message per tool call — that is what caused 155
                # comments under a single PR.  Instead, preserve the sentinel
                # so the full continuation is delivered once via
                # _send_fallback_final.
                # (When editing fails mid-stream due to flood control the id is
                # a real string like "msg_1", not "__no_edit__", so that case
                # still resets and creates a fresh segment as intended.)
                if got_segment_break:
                    # If the segment-break edit failed to deliver the
                    # accumulated content (flood control that has not yet
                    # promoted to fallback mode, or fallback mode itself),
                    # _accumulated still holds pre-boundary text the user
                    # never saw. Flush that tail as a continuation message
                    # before the reset below wipes _accumulated — otherwise
                    # text generated before the tool boundary is silently
                    # dropped (issue #8124).
                    if (
                        self._accumulated
                        and not current_update_visible
                        and self._message_id
                        and self._message_id != "__no_edit__"
                    ):
                        await self._flush_segment_tail_on_edit_failure()
                    self._reset_segment_state(preserve_no_edit=True)

                await asyncio.sleep(0.05)  # Small yield to not busy-loop

        except asyncio.CancelledError:
            # Best-effort final edit on cancellation
            _best_effort_ok = False
            if self._accumulated and self._message_id:
                try:
                    _best_effort_ok = bool(await self._send_or_edit(self._accumulated))
                except Exception:
                    pass
            # Only confirm final delivery if the best-effort send above
            # actually succeeded OR if the final response was already
            # confirmed before we were cancelled.  Previously this
            # promoted any partial send (already_sent=True) to
            # final_response_sent — which suppressed the gateway's
            # fallback send even when only intermediate text (e.g.
            # "Let me search…") had been delivered, not the real answer.
            if _best_effort_ok and not self._final_response_sent:
                self._final_response_sent = True
        except Exception as e:
            logger.error("Stream consumer error: %s", e)

    # Pattern to strip MEDIA:<path> tags (including optional surrounding quotes).
    # Matches the simple cleanup regex used by the non-streaming path in
    # gateway/platforms/base.py for post-processing.
    _MEDIA_RE = re.compile(r'''[`"']?MEDIA:\s*\S+[`"']?''')

    @staticmethod
    def _clean_for_display(text: str) -> str:
        """Strip MEDIA: directives and internal markers from text before display.

        The streaming path delivers raw text chunks that may include
        ``MEDIA:<path>`` tags and ``[[audio_as_voice]]`` directives meant for
        the platform adapter's post-processing.  The actual media files are
        delivered separately via ``_deliver_media_from_response()`` after the
        stream finishes — we just need to hide the raw directives from the
        user.
        """
        if "MEDIA:" not in text and "[[audio_as_voice]]" not in text:
            return text
        cleaned = text.replace("[[audio_as_voice]]", "")
        cleaned = GatewayStreamConsumer._MEDIA_RE.sub("", cleaned)
        # Collapse excessive blank lines left behind by removed tags
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        # Strip trailing whitespace/newlines but preserve leading content
        return cleaned.rstrip()

    async def _send_new_chunk(self, text: str, reply_to_id: Optional[str]) -> Optional[str]:
        """Send a new message chunk, optionally threaded to a previous message.

        Returns the message_id so callers can thread subsequent chunks.
        """
        text = self._clean_for_display(text)
        if not text.strip():
            return reply_to_id
        try:
            meta = dict(self.metadata) if self.metadata else {}
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                reply_to=reply_to_id,
                metadata=meta,
            )
            if result.success and result.message_id:
                self._message_id = str(result.message_id)
                self._already_sent = True
                self._last_sent_text = text
                # Fresh content bubble — close off any stale tool bubble
                # above so the next tool starts a new bubble below.
                self._notify_new_message()
                return str(result.message_id)
            else:
                self._edit_supported = False
                return reply_to_id
        except Exception as e:
            logger.error("Stream send chunk error: %s", e)
            return reply_to_id

    def _visible_prefix(self) -> str:
        """Return the visible text already shown in the streamed message."""
        prefix = self._last_sent_text or ""
        if self.cfg.cursor and prefix.endswith(self.cfg.cursor):
            prefix = prefix[:-len(self.cfg.cursor)]
        return self._clean_for_display(prefix)

    def _continuation_text(self, final_text: str) -> str:
        """Return only the part of final_text the user has not already seen."""
        prefix = self._fallback_prefix or self._visible_prefix()
        if prefix and final_text.startswith(prefix):
            return final_text[len(prefix):].lstrip()
        return final_text

    @staticmethod
    def _split_text_chunks(text: str, limit: int) -> list[str]:
        """Split text into reasonably sized chunks for fallback sends."""
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind("\n", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
        if remaining:
            chunks.append(remaining)
        return chunks

    async def _send_fallback_final(self, text: str) -> None:
        """Send the final continuation after streaming edits stop working.

        Retries each chunk once on flood-control failures with a short delay.
        """
        final_text = self._clean_for_display(text)
        continuation = self._continuation_text(final_text)
        self._fallback_final_send = False
        if not continuation.strip():
            # Nothing new to send — the visible partial already matches final text.
            # BUT: if final_text itself has meaningful content (e.g. a timeout
            # message after a long tool call), the prefix-based continuation
            # calculation may wrongly conclude "already shown" because the
            # streamed prefix was from a *previous* segment (before the tool
            # boundary).  In that case, send the full final_text as-is (#10807).
            if final_text.strip() and final_text != self._visible_prefix():
                continuation = final_text
            else:
                # Defence-in-depth for #7183: the last edit may still show the
                # cursor character because fallback mode was entered after an
                # edit failure left it stuck.  Try one final edit to strip it
                # so the message doesn't freeze with a visible ▉.  Best-effort
                # — if this edit also fails (flood control still active),
                # _try_strip_cursor has already been called on fallback entry
                # and the adaptive-backoff retries will have had their shot.
                if (
                    self._message_id
                    and self._last_sent_text
                    and self.cfg.cursor
                    and self._last_sent_text.endswith(self.cfg.cursor)
                ):
                    clean_text = self._last_sent_text[:-len(self.cfg.cursor)]
                    try:
                        result = await self.adapter.edit_message(
                            chat_id=self.chat_id,
                            message_id=self._message_id,
                            content=clean_text,
                        )
                        if result.success:
                            self._last_sent_text = clean_text
                    except Exception:
                        pass
                self._already_sent = True
                self._final_response_sent = True
                return

        raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        safe_limit = max(500, raw_limit - 100)
        chunks = self._split_text_chunks(continuation, safe_limit)

        last_message_id: Optional[str] = None
        last_successful_chunk = ""
        sent_any_chunk = False
        for chunk in chunks:
            # Try sending with one retry on flood-control errors.
            result = None
            for attempt in range(2):
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=chunk,
                    metadata=self.metadata,
                )
                if result.success:
                    break
                if attempt == 0 and self._is_flood_error(result):
                    logger.debug(
                        "Flood control on fallback send, retrying in 3s"
                    )
                    await asyncio.sleep(3.0)
                else:
                    break  # non-flood error or second attempt failed

            if not result or not result.success:
                if sent_any_chunk:
                    # Some continuation text already reached the user. Suppress
                    # the base gateway final-send path so we don't resend the
                    # full response and create another duplicate.
                    self._already_sent = True
                    self._final_response_sent = True
                    self._message_id = last_message_id
                    self._last_sent_text = last_successful_chunk
                    self._fallback_prefix = ""
                    return
                # No fallback chunk reached the user — allow the normal gateway
                # final-send path to try one more time.
                self._already_sent = False
                self._message_id = None
                self._last_sent_text = ""
                self._fallback_prefix = ""
                return
            sent_any_chunk = True
            last_successful_chunk = chunk
            last_message_id = result.message_id or last_message_id
            # Each fallback chunk is a fresh platform message — notify
            # so any stale tool-progress bubble gets closed off.
            self._notify_new_message()

        self._message_id = last_message_id
        self._already_sent = True
        self._final_response_sent = True
        self._last_sent_text = chunks[-1]
        self._fallback_prefix = ""

    def _is_flood_error(self, result) -> bool:
        """Check if a SendResult failure is due to flood control / rate limiting."""
        err = getattr(result, "error", "") or ""
        err_lower = err.lower()
        return "flood" in err_lower or "retry after" in err_lower or "rate" in err_lower

    async def _flush_segment_tail_on_edit_failure(self) -> None:
        """Deliver un-sent tail content before a segment-break reset.

        When an edit fails (flood control, transport error) and a tool
        boundary arrives before the next retry, ``_accumulated`` holds text
        that was generated but never shown to the user. Without this flush,
        the segment reset would discard that tail and leave a frozen cursor
        in the partial message.

        Sends the tail that sits after the last successfully-delivered
        prefix as a new message, and best-effort strips the stuck cursor
        from the previous partial message.
        """
        if not self._fallback_final_send:
            await self._try_strip_cursor()
        visible = self._fallback_prefix or self._visible_prefix()
        tail = self._accumulated
        if visible and tail.startswith(visible):
            tail = tail[len(visible):].lstrip()
        tail = self._clean_for_display(tail)
        if not tail.strip():
            return
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=tail,
                metadata=self.metadata,
            )
            if result.success:
                self._already_sent = True
        except Exception as e:
            logger.error("Segment-break tail flush error: %s", e)

    async def _try_strip_cursor(self) -> None:
        """Best-effort edit to remove the cursor from the last visible message.

        Called when entering fallback mode so the user doesn't see a stuck
        cursor (▉) in the partial message.
        """
        if not self._message_id or self._message_id == "__no_edit__":
            return
        prefix = self._visible_prefix()
        if not prefix or not prefix.strip():
            return
        try:
            await self.adapter.edit_message(
                chat_id=self.chat_id,
                message_id=self._message_id,
                content=prefix,
            )
            self._last_sent_text = prefix
        except Exception:
            pass  # best-effort — don't let this block the fallback path

    async def _send_commentary(self, text: str) -> bool:
        """Send a completed interim assistant commentary message."""
        text = self._clean_for_display(text)
        if not text.strip():
            return False
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                metadata=self.metadata,
            )
            # Note: do NOT set _already_sent = True here.
            # Commentary messages are interim status updates (e.g. "Using browser
            # tool..."), not the final response. Setting already_sent would cause
            # the final response to be incorrectly suppressed when there are
            # multiple tool calls. See: https://github.com/NousResearch/hermes-agent/issues/10454
            if result.success:
                # Commentary counts as fresh content — close off any
                # stale tool bubble above it so the next tool starts a
                # new bubble below.
                self._notify_new_message()
            return result.success
        except Exception as e:
            logger.error("Commentary send error: %s", e)
            return False

    def _should_send_fresh_final(self) -> bool:
        """Return True when a long-lived preview should be replaced with a
        fresh final message instead of an edit.

        Conditions:
        - Fresh-final is enabled (``fresh_final_after_seconds > 0``).
        - We have a real preview message id (not the ``__no_edit__`` sentinel
          and not ``None``).
        - The preview has been visible for at least the configured threshold.

        Ported from openclaw/openclaw#72038.
        """
        threshold = getattr(self.cfg, "fresh_final_after_seconds", 0.0) or 0.0
        if threshold <= 0:
            return False
        if not self._message_id or self._message_id == "__no_edit__":
            return False
        if self._message_created_ts is None:
            return False
        age = time.monotonic() - self._message_created_ts
        return age >= threshold

    async def _try_fresh_final(self, text: str) -> bool:
        """Send ``text`` as a brand-new message (best-effort delete the old
        preview) so the platform's visible timestamp reflects completion
        time.  Returns True on successful delivery, False on any failure so
        the caller falls back to the normal edit path.

        Ported from openclaw/openclaw#72038.
        """
        old_message_id = self._message_id
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                metadata=self.metadata,
            )
        except Exception as e:
            logger.debug("Fresh-final send failed, falling back to edit: %s", e)
            return False
        if not getattr(result, "success", False):
            return False
        # Successful fresh send — try to delete the stale preview so the
        # user doesn't see the old edit-stuck message underneath.  Cleanup
        # is best-effort; platforms that don't implement ``delete_message``
        # just leave the preview behind (still an acceptable outcome —
        # the visible final timestamp is the important part).
        if old_message_id and old_message_id != "__no_edit__":
            delete_fn = getattr(self.adapter, "delete_message", None)
            if delete_fn is not None:
                try:
                    await delete_fn(self.chat_id, old_message_id)
                except Exception as e:
                    logger.debug(
                        "Fresh-final preview cleanup failed (%s): %s",
                        old_message_id, e,
                    )
        # Adopt the new message id as the current message so subsequent
        # callers (e.g. overflow split loops, finalize retries) see a
        # consistent state.
        new_message_id = getattr(result, "message_id", None)
        if new_message_id:
            self._message_id = new_message_id
            self._message_created_ts = time.monotonic()
        else:
            # Send succeeded but platform didn't return an id — treat the
            # delivery as final-only and fall back to "__no_edit__" so we
            # don't try to edit something we can't address.
            self._message_id = "__no_edit__"
            self._message_created_ts = None
        self._already_sent = True
        self._last_sent_text = text
        self._final_response_sent = True
        return True

    async def _send_or_edit(self, text: str, *, finalize: bool = False) -> bool:
        """Send or edit the streaming message.

        Returns True if the text was successfully delivered (sent or edited),
        False otherwise.  Callers like the overflow split loop use this to
        decide whether to advance past the delivered chunk.

        ``finalize`` is True when this is the last edit in a streaming
        sequence.
        """
        # Strip MEDIA: directives so they don't appear as visible text.
        # Media files are delivered as native attachments after the stream
        # finishes (via _deliver_media_from_response in gateway/run.py).
        text = self._clean_for_display(text)
        # A bare streaming cursor is not meaningful user-visible content and
        # can render as a stray tofu/white-box message on some clients.
        visible_without_cursor = text
        if self.cfg.cursor:
            visible_without_cursor = visible_without_cursor.replace(self.cfg.cursor, "")
        _visible_stripped = visible_without_cursor.strip()
        if not _visible_stripped:
            return True  # cursor-only / whitespace-only update
        if not text.strip():
            return True  # nothing to send is "success"
        # Guard: do not create a brand-new standalone message when the only
        # visible content is a handful of characters alongside the streaming
        # cursor.  During rapid tool-calling the model often emits 1-2 tokens
        # before switching to tool calls; the resulting "X ▉" message risks
        # leaving the cursor permanently visible if the follow-up edit (to
        # strip the cursor on segment break) is rate-limited by the platform.
        # This was reported on Telegram, Matrix, and other clients where the
        # ▉ block character renders as a visible white box ("tofu").
        # Existing messages (edits) are unaffected — only first sends gated.
        _MIN_NEW_MSG_CHARS = 4
        if (self._message_id is None
                and self.cfg.cursor
                and self.cfg.cursor in text
                and len(_visible_stripped) < _MIN_NEW_MSG_CHARS):
            return True  # too short for a standalone message — accumulate more
        try:
            if self._message_id is not None:
                if self._edit_supported:
                    # Skip if text is identical to what we last sent.
                    # Exception: adapters that require an explicit finalize
                    # call (REQUIRES_EDIT_FINALIZE) must still receive the
                    # finalize=True edit even when content is unchanged, so
                    # their streaming UI can transition out of the in-
                    # progress state.  Everyone else short-circuits.
                    if text == self._last_sent_text and not (
                        finalize and self._adapter_requires_finalize
                    ):
                        return True
                    # Fresh-final for long-lived previews: when finalizing
                    # the last edit in a streaming sequence, if the
                    # original preview has been visible for at least
                    # ``fresh_final_after_seconds``, send the completed
                    # reply as a fresh message so the platform's visible
                    # timestamp reflects completion time instead of the
                    # preview creation time.  Best-effort cleanup of the
                    # old preview follows.  Ported from
                    # openclaw/openclaw#72038.  Gated by config so the
                    # legacy edit-in-place path stays the default.
                    if (
                        finalize
                        and self._should_send_fresh_final()
                        and await self._try_fresh_final(text)
                    ):
                        return True
                    # Edit existing message
                    result = await self.adapter.edit_message(
                        chat_id=self.chat_id,
                        message_id=self._message_id,
                        content=text,
                        finalize=finalize,
                    )
                    if result.success:
                        self._already_sent = True
                        self._last_sent_text = text
                        # Successful edit — reset flood strike counter
                        self._flood_strikes = 0
                        return True
                    else:
                        # Edit failed.  If this looks like flood control / rate
                        # limiting, use adaptive backoff: double the edit interval
                        # and retry on the next cycle.  Only permanently disable
                        # edits after _MAX_FLOOD_STRIKES consecutive failures.
                        if self._is_flood_error(result):
                            self._flood_strikes += 1
                            self._current_edit_interval = min(
                                self._current_edit_interval * 2, 10.0,
                            )
                            logger.debug(
                                "Flood control on edit (strike %d/%d), "
                                "backoff interval → %.1fs",
                                self._flood_strikes,
                                self._MAX_FLOOD_STRIKES,
                                self._current_edit_interval,
                            )
                            if self._flood_strikes < self._MAX_FLOOD_STRIKES:
                                # Don't disable edits yet — just slow down.
                                # Update _last_edit_time so the next edit
                                # respects the new interval.
                                self._last_edit_time = time.monotonic()
                                return False

                        # Non-flood error OR flood strikes exhausted: enter
                        # fallback mode — send only the missing tail once the
                        # final response is available.
                        logger.debug(
                            "Edit failed (strikes=%d), entering fallback mode",
                            self._flood_strikes,
                        )
                        self._fallback_prefix = self._visible_prefix()
                        self._fallback_final_send = True
                        self._edit_supported = False
                        self._already_sent = True
                        # Best-effort: strip the cursor from the last visible
                        # message so the user doesn't see a stuck ▉.
                        await self._try_strip_cursor()
                        return False
                else:
                    # Editing not supported — skip intermediate updates.
                    # The final response will be sent by the fallback path.
                    return False
            else:
                # First message — send new
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=text,
                    metadata=self.metadata,
                )
                if result.success:
                    if result.message_id:
                        self._message_id = result.message_id
                        # Track when the preview first became visible to
                        # the user so fresh-final logic can detect stale
                        # preview timestamps on long-running responses.
                        self._message_created_ts = time.monotonic()
                    else:
                        self._edit_supported = False
                    self._already_sent = True
                    self._last_sent_text = text
                    if not result.message_id:
                        self._fallback_prefix = self._visible_prefix()
                        self._fallback_final_send = True
                        # Sentinel prevents re-entering the first-send path on
                        # every delta/tool boundary when platforms accept a
                        # message but do not return an editable message id.
                        self._message_id = "__no_edit__"
                    # Notify the gateway that a fresh content bubble was
                    # created so any accumulated tool-progress bubble above
                    # gets closed off — the next tool fires into a new
                    # bubble below, preserving chronological order.
                    self._notify_new_message()
                    return True
                else:
                    # Initial send failed — disable streaming for this session
                    self._edit_supported = False
                    return False
        except Exception as e:
            logger.error("Stream send/edit error: %s", e)
            return False
