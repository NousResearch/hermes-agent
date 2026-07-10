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
import inspect
import logging
import queue
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from gateway.platforms.base import BasePlatformAdapter as _BasePlatformAdapter
from gateway.platforms.base import _custom_unit_to_cp
from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE
from gateway.config import (
    DEFAULT_STREAMING_EDIT_INTERVAL as _DEFAULT_STREAMING_EDIT_INTERVAL,
    DEFAULT_STREAMING_BUFFER_THRESHOLD as _DEFAULT_STREAMING_BUFFER_THRESHOLD,
    DEFAULT_STREAMING_CURSOR as _DEFAULT_STREAMING_CURSOR,
)
from gateway.response_filters import (
    is_intentional_silence_response as _is_intentional_silence_response,
    is_partial_silence_marker as _is_partial_silence_marker,
)

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
    edit_interval: float = _DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = _DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = _DEFAULT_STREAMING_CURSOR
    buffer_only: bool = False
    # When >0, the final edit for a streamed response is delivered as a
    # fresh message if the original preview has been visible for at least
    # this many seconds.  This makes the platform's visible timestamp
    # reflect completion time instead of first-token time for long-running
    # responses (e.g. reasoning models that stream slowly).  Ported from
    # openclaw/openclaw#72038.  Default 0 = always edit in place (legacy
    # behavior).  The gateway enables this selectively per-platform.
    fresh_final_after_seconds: float = 0.0
    # Streaming transport selection:
    #   "auto"  — prefer native draft streaming (e.g. Telegram sendMessageDraft)
    #             when the adapter + chat supports it; fall back to edit.
    #   "draft" — explicitly request native draft streaming; fall back to
    #             edit when unsupported.
    #   "edit"  — progressive editMessageText (legacy/default behavior).
    #   "off"   — handled by the gateway before the consumer is even built.
    transport: str = "edit"
    # Hint for the consumer about the originating chat type (e.g. "dm",
    # "group", "supergroup", "forum").  Used to gate native draft streaming,
    # which is platform-specific (Telegram drafts are DM-only).
    chat_type: str = ""


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

    # Class-wide monotonic counter for native-streaming draft ids.  Telegram
    # animates a draft when the same draft_id is reused across consecutive
    # calls in the same chat, so we need a fresh non-zero id per response.
    _draft_id_counter: int = 0

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        config: Optional[StreamConsumerConfig] = None,
        metadata: Optional[dict] = None,
        on_new_message: Optional[callable] = None,
        on_before_finalize: Optional[Callable[[], Any]] = None,
        initial_reply_to_id: Optional[str] = None,
        run_still_current: Optional[Callable[[], bool]] = None,
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
        # Fired once when the stream transitions into its finalization path.
        # Gateway callers use this to pause typing refreshes before a slow
        # final rich-text edit (Telegram MarkdownV2 finalize, etc.).
        self._on_before_finalize = on_before_finalize
        self._initial_reply_to_id = initial_reply_to_id
        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        self._message_id: Optional[str] = None
        # Wall-clock timestamp (time.monotonic) when ``_message_id`` was
        # first assigned from a successful first-send.  Used by the
        # fresh-final logic to detect long-lived previews whose edit
        # timestamps would be stale by completion time.  Ported from
        # openclaw/openclaw#72038.
        self._message_created_ts: Optional[float] = None
        # Every real preview message id the consumer has put on screen during
        # this response (first send + any continuation messages from oversized
        # edits/sends).  The fresh-final path deletes all of them when it
        # re-delivers the completed answer as a single (rich) message, so a
        # reply that was split across the platform's edit limit while streaming
        # doesn't leave stale fragments above the final message.
        self._preview_message_ids: "set[str]" = set()
        self._already_sent = False
        self._edit_supported = True  # Disabled when progressive edits are no longer usable
        self._last_edit_time = 0.0
        self._last_sent_text = ""   # Track last-sent text to skip redundant edits
        # True when the most recent _send_or_edit split-and-delivered across
        # continuation messages (the adapter adopted a new message id).
        self._last_edit_overflowed = False
        self._fallback_final_send = False
        self._fallback_prefix = ""
        # True when fallback is sending only the missing tail after a partial
        # Telegram overflow delivery.  In that case the already-visible prefix
        # is intentional content, not a stale preview to delete.
        self._fallback_preserve_partial_messages = False
        self._flood_strikes = 0         # Consecutive flood-control edit failures
        self._current_edit_interval = self.cfg.edit_interval  # Adaptive backoff
        self._final_response_sent = False
        # Set when the final response content was sent to the user via
        # streaming, even if the final edit (cursor removal etc.)
        # subsequently failed.
        self._final_content_delivered = False
        self._delivered_commentary_texts: list[str] = []
        # Cache adapter lifecycle capability: only platforms that need an
        # explicit finalize call (e.g. DingTalk AI Cards) force us to make
        # a redundant final edit.  Everyone else keeps the fast path.
        # Use ``is True`` (not ``bool(...)``) so MagicMock attribute access
        # in tests doesn't incorrectly enable this path.
        self._adapter_requires_finalize: bool = (
            getattr(adapter, "REQUIRES_EDIT_FINALIZE", False) is True
        )

        # Session staleness guard — when set to False (e.g. after /new or
        # /stop), the run() loop will abandon the stream early instead of
        # continuing to edit and deliver stale deltas.
        self._run_still_current = run_still_current or (lambda: True)

        # Think-block filter state (mirrors CLI's _stream_delta tag suppression)
        self._in_think_block = False
        self._think_buffer = ""

        # Native draft-streaming state.  Resolved at the start of run() based
        # on cfg.transport, cfg.chat_type, and the adapter's
        # supports_draft_streaming() probe.  When True, the consumer emits
        # animated draft frames via adapter.send_draft instead of progressive
        # edits via adapter.edit_message.  The final answer still goes
        # through the normal first-send path so the user gets a real message
        # in their chat history (drafts have no message_id).
        self._use_draft_streaming = False
        self._draft_id: Optional[int] = None
        # Cumulative draft-frame failure count for this consumer.  After the
        # first failure we permanently disable drafts for the remainder of
        # this response and route through edit-based for graceful degradation.
        self._draft_failures = 0
        self._before_finalize_notified = False

    def _metadata_for_send(
        self,
        *,
        final: bool = False,
        expect_edits: bool = False,
    ) -> dict | None:
        """Return per-send metadata for stream-created messages.

        Mattermost treats notify-worthy sends as user-visible final content
        when deciding whether a broken thread root may fall back flat.  Preview
        and progress sends keep their original metadata and remain thread-strict.

        ``expect_edits`` preserves the upstream Telegram streaming contract:
        preview messages that may be edited later must stay on the editable
        legacy send path, while fresh/fallback final sends can still use richer
        final-message delivery.
        """
        meta = dict(self.metadata) if self.metadata else {}
        if expect_edits:
            meta["expect_edits"] = True
        if final:
            meta["notify"] = True
        return meta or None

    @property
    def already_sent(self) -> bool:
        """True if at least one message was sent or edited during the run."""
        return self._already_sent

    @property
    def final_response_sent(self) -> bool:
        """True when the stream consumer delivered the final assistant reply."""
        return self._final_response_sent

    @property
    def message_id(self) -> str | None:
        """The Discord/chat message ID of the last-sent or edited message."""
        return self._message_id

    @property
    def final_content_delivered(self) -> bool:
        """True when the final response content reached the user, even if
        the subsequent cosmetic edit (cursor removal) failed."""
        return self._final_content_delivered

    async def _notify_before_finalize(self) -> None:
        """Run the pre-finalize hook exactly once, swallowing hook errors."""
        if self._before_finalize_notified:
            return
        self._before_finalize_notified = True
        if self._on_before_finalize is None:
            return
        try:
            result = self._on_before_finalize()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    async def _edit_message(
        self,
        *,
        message_id: str,
        content: str,
        finalize: bool = False,
    ):
        """Edit via the adapter, passing routing metadata when supported."""
        kwargs = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "content": content,
        }
        # Keep the long-standing stream-consumer contract: concrete adapters
        # must accept finalize= even when it is False (guarded by tests).
        kwargs["finalize"] = finalize

        if self.metadata:
            try:
                params = inspect.signature(self.adapter.edit_message).parameters
                if "metadata" in params or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in params.values()
                ):
                    kwargs["metadata"] = self.metadata
            except (TypeError, ValueError):
                pass
        return await self.adapter.edit_message(**kwargs)

    def has_delivered_text(self, text: str) -> bool:
        """Return True if *text* was already delivered as visible chat content."""
        target = self._clean_for_display(text or "").strip()
        if not target:
            return False
        visible_prefix = self._visible_prefix().strip()
        if visible_prefix == target:
            return True
        return any(sent.strip() == target for sent in self._delivered_commentary_texts)

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
        self._fallback_preserve_partial_messages = False
        # #29346: a tool/segment boundary means what we delivered was an interim
        # preamble, not the final answer — clear the flags so a premature setter
        # can't fool the gateway. Safe: got_done returns before any reset, and
        # run.py reads these only after the consumer task exits.
        self._final_response_sent = False
        self._final_content_delivered = False
        # Native draft streaming: bump the draft_id so the next text segment
        # animates as a fresh preview below the tool-progress bubbles, not
        # over the prior segment's already-finalized draft.  This is how
        # we avoid the "inter-tool-call text leak" failure mode openclaw
        # documented in their issue #32535 — each text block becomes its
        # own visible message via the finalize, then a new draft animates
        # for the next one.
        if self._use_draft_streaming:
            type(self)._draft_id_counter += 1
            self._draft_id = type(self)._draft_id_counter

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
            # Case-insensitive matching: models emit mixed-case tag
            # variants (<Think>, <THINKING>, …). Match against a
            # lowercased view of the buffer with lowercased tag names so
            # every case variant is caught with a single canonical form.
            lower_buf = buf.lower()
            if self._in_think_block:
                # Look for the earliest closing tag
                best_idx = -1
                best_len = 0
                for tag in self._CLOSE_THINK_TAGS:
                    idx = lower_buf.find(tag.lower())
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
                    tag_lower = tag.lower()
                    search_start = 0
                    while True:
                        idx = lower_buf.find(tag_lower, search_start)
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
                        tag_lower = tag.lower()
                        for i in range(1, len(tag)):
                            if lower_buf.endswith(tag_lower[:i]) and i > held_back:
                                held_back = i
                    if held_back:
                        self._accumulated += buf[:-held_back]
                        self._think_buffer = buf[-held_back:]
                    else:
                        # No (partial) open tag — but the model may have
                        # emitted an orphan close tag like </think> on its
                        # own (e.g. when a thinking-mode toggle drops the
                        # matched open, or when upstream stripping is
                        # incomplete). Strip those before accumulating so
                        # they never reach the user.
                        self._accumulated += self._strip_orphan_close_tags(buf)
                    return

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove any close tags from *text* that have no matching open.

        Mirrors ``agent/think_scrubber.py::StreamingThinkScrubber.
        _strip_orphan_close_tags`` so the progressive-display filter
        behaves the same as the post-stream final-response scrubber.
        An orphan close tag is always noise — stripped along with any
        trailing whitespace so surrounding prose flows naturally.
        """
        if "</" not in text:
            return text
        text_lower = text.lower()
        out: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            if text_lower[i:i + 2] == "</":
                for tag in cls._CLOSE_THINK_TAGS:
                    tag_lower = tag.lower()
                    tag_len = len(tag_lower)
                    if text_lower[i:i + tag_len] == tag_lower:
                        j = i + tag_len
                        while j < len(text) and text[j] in " \t\n\r":
                            j += 1
                        i = j
                        matched = True
                        break
            if not matched:
                out.append(text[i])
                i += 1
        return "".join(out)

    def _flush_think_buffer(self) -> None:
        """Flush any held-back partial-tag buffer into accumulated text.

        Called when the stream ends (got_done) so that partial text that
        was held back waiting for a possible opening tag is not lost.
        """
        if self._think_buffer and not self._in_think_block:
            # Strip any orphan close tags that may have been held back —
            # see _filter_and_accumulate for context.
            self._accumulated += self._strip_orphan_close_tags(self._think_buffer)
            self._think_buffer = ""

    async def run(self) -> None:
        """Async task that drains the queue and edits the platform message."""
        # Platform message length limit — leave room for cursor + formatting.
        # Use the adapter's length function (e.g. utf16_len for Telegram) so
        # overflow detection matches what the platform actually enforces.
        # Gate on isinstance(BasePlatformAdapter) so test MagicMocks (whose
        # auto-attributes return mock objects, not callables) fall back to len.
        _len_fn: "Callable[[str], int]" = (
            self.adapter.message_len_fn
            if isinstance(self.adapter, _BasePlatformAdapter)
            else len
        )
        # Rich-capable adapters (Telegram rich messages) raise this above the
        # legacy per-message limit so a reply that fits one rich send/draft
        # isn't fragmented at 4096 while streaming.  See _raw_message_limit.
        _raw_limit = self._raw_message_limit()
        _safe_limit = max(500, _raw_limit - _len_fn(self.cfg.cursor) - 100)

        # Resolve native draft streaming once per run.  When enabled the
        # consumer routes mid-stream frames through adapter.send_draft and
        # leaves _message_id=None so the existing got_done path delivers the
        # final answer as a regular sendMessage (drafts have no message_id
        # to edit).
        self._use_draft_streaming = self._resolve_draft_streaming()
        if self._use_draft_streaming:
            type(self)._draft_id_counter += 1
            self._draft_id = type(self)._draft_id_counter
            logger.debug(
                "Stream consumer using native-draft transport (chat=%s draft_id=%s)",
                self.chat_id, self._draft_id,
            )

        try:
            while True:
                # Abandon the stream early if the session has been reset
                # (e.g. /new or /stop). Prevents stale deltas from being
                # delivered after the user has already moved on.
                if not self._run_still_current():
                    return

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

                    # Intentional-silence suppression.  When the agent chose
                    # not to reply it emits a bare control marker (NO_REPLY /
                    # [SILENT] / …).  The gateway's whole-response filter
                    # (gateway/run.py) suppresses this on the non-streaming
                    # path, but by the time it runs the stream consumer has
                    # already edited the raw marker onto the screen.  Detect
                    # the exact-marker final buffer here and retract any
                    # preview instead of finalizing it, so the marker never
                    # reaches the chat.  Substantive prose that merely mentions
                    # a marker is NOT suppressed (see is_intentional_silence_response).
                    if _is_intentional_silence_response(
                        self._clean_for_display(self._accumulated)
                    ):
                        await self._suppress_silence_marker()
                        return

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
                        # buffer_threshold is intentionally codepoint-based:
                        # it's a debounce heuristic ("send updates roughly
                        # every N visible characters"), not a platform-limit
                        # check. _len_fn is reserved for overflow detection.
                        or len(self._accumulated) >= self.cfg.buffer_threshold
                    )

                current_update_visible = False
                # Hold back mid-stream edits while the buffer so far could
                # still resolve to an intentional-silence marker.  Without
                # this, a partial marker (e.g. "NO_REPLY" streamed as
                # "NO"→"NO_REPLY") would flash onto the screen on an interval
                # tick before got_done can suppress it.  Only defers display —
                # got_done above always resolves the buffer (suppress if it's
                # an exact marker, otherwise fall through and flush normally),
                # so genuine prose that merely starts marker-like is never lost.
                if (
                    should_edit
                    and not got_done
                    and not got_segment_break
                    and commentary_text is None
                    and _is_partial_silence_marker(
                        self._clean_for_display(self._accumulated)
                    )
                ):
                    should_edit = False
                if should_edit and self._accumulated:
                    # Split overflow: if accumulated text exceeds the platform
                    # limit, split into properly sized chunks.
                    if (
                        _len_fn(self._accumulated) > _safe_limit
                        and self._message_id is None
                    ):
                        # No existing message to edit (first message or after a
                        # segment break).  Use truncate_message — the same
                        # helper the non-streaming path uses — to split with
                        # proper word/code-fence boundaries and chunk
                        # indicators like "(1/2)".
                        chunks = self.adapter.truncate_message(
                            self._accumulated, _safe_limit, len_fn=_len_fn,
                        )
                        chunks_delivered = False
                        reply_to = self._message_id or self._initial_reply_to_id
                        for chunk in chunks:
                            new_id = await self._send_new_chunk(
                                chunk,
                                reply_to,
                                final=got_done,
                            )
                            if new_id is not None and new_id != reply_to:
                                chunks_delivered = True
                        self._accumulated = ""
                        self._last_sent_text = ""
                        self._last_edit_time = time.monotonic()
                        if got_done:
                            # Only claim final delivery if THESE chunks actually
                            # landed.  ``_already_sent`` may be True from prior
                            # tool-progress edits or fallback-mode promotion (#10748)
                            # — that doesn't mean the final answer reached the user.
                            self._final_response_sent = chunks_delivered
                            if chunks_delivered:
                                self._final_content_delivered = True
                            return
                        if got_segment_break:
                            self._message_id = None
                            self._fallback_final_send = False
                            self._fallback_prefix = ""
                        continue

                    # Existing message: edit it with the first chunk, then
                    # start a new message for the overflow remainder.
                    while (
                        _len_fn(self._accumulated) > _safe_limit
                        and self._message_id is not None
                        and self._edit_supported
                    ):
                        _cp_budget = _custom_unit_to_cp(
                            self._accumulated, _safe_limit, _len_fn,
                        )
                        split_at = self._accumulated.rfind("\n", 0, _cp_budget)
                        if split_at < _safe_limit // 2:
                            split_at = _safe_limit
                        chunk = self._accumulated[:split_at]
                        # finalize=True so the adapter applies platform-specific
                        # rich-text markup (e.g. Telegram MarkdownV2). This
                        # sealed chunk will never be edited again — _message_id
                        # is reset to None right below — so it must receive its
                        # final formatting pass now, or early split messages
                        # render raw markdown while only the last chunk renders.
                        # is_turn_final=False: this is the first of several split
                        # messages, NOT the turn-final answer, so the fresh-final
                        # path (opt-in fresh_final_after_seconds) must not mark
                        # the turn delivered on it (#29346 semantics).
                        ok = await self._send_or_edit(
                            chunk, finalize=True, is_turn_final=False,
                        )
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
                        # A segment-break finalize closes a preamble, not the
                        # turn-final answer — only got_done marks delivered (#29346).
                        is_turn_final=got_done,
                    )
                    self._last_edit_time = time.monotonic()

                if got_done:
                    if self._accumulated or self._message_id is not None or self._already_sent:
                        await self._notify_before_finalize()
                    # Final edit without cursor. If progressive editing failed
                    # mid-stream, send a single continuation/fallback message
                    # here instead of letting the base gateway path send the
                    # full response again.
                    if self._accumulated:
                        if self._fallback_final_send:
                            await self._send_fallback_final(self._accumulated)
                        elif self._final_response_sent:
                            # A finalize=True tick above already delivered the
                            # final answer via the adapter's fresh-final path
                            # (_try_fresh_final sent a fresh rich message and
                            # deleted the preview).  Running a second finalize
                            # edit here would duplicate the message / re-delete,
                            # so just record delivery and stop.
                            self._final_content_delivered = True
                        elif (
                            current_update_visible
                            and (
                                not self._adapter_requires_finalize
                                or self._last_edit_overflowed
                            )
                        ):
                            # Mid-stream edit above already delivered the
                            # final accumulated content.  Skip the redundant
                            # final edit for adapters that don't need an
                            # explicit finalize signal, and for any adapter
                            # when that edit split-and-delivered across
                            # continuations: the split edit carried
                            # finalize=True itself, and re-finalizing with
                            # the full text would overflow-split again into
                            # the adopted continuation, duplicating chunks
                            # on screen.
                            self._final_response_sent = True
                            self._final_content_delivered = True
                        elif self._message_id:
                            # Either the mid-stream edit didn't run (no
                            # visible update this tick) OR the adapter needs
                            # explicit finalize=True to close the stream.
                            self._final_response_sent = await self._send_or_edit(
                                self._accumulated, finalize=True,
                            )
                            if self._final_response_sent:
                                self._final_content_delivered = True
                            elif self._fallback_final_send:
                                # The final edit attempt itself may be the one
                                # that exhausts flood-control strikes and
                                # promotes the consumer into fallback mode.  Do
                                # not return to the gateway with a full-response
                                # fallback still pending; send only the unsent
                                # tail here so the normal gateway send path does
                                # not duplicate the visible prefix.
                                await self._send_fallback_final(self._accumulated)
                        elif not self._already_sent:
                            self._final_response_sent = await self._send_or_edit(self._accumulated)
                            if self._final_response_sent:
                                self._final_content_delivered = True
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
            # Best-effort final edit on cancellation.  finalize=True so
            # REQUIRES_EDIT_FINALIZE platforms (Telegram) apply final
            # formatting — a plain edit here would leave the entire reply
            # rendered as a raw streaming preview while the success flags
            # below suppress the gateway's formatted re-send.
            # is_turn_final=False keeps _try_fresh_final from setting
            # _final_response_sent itself; this handler owns the flags.
            _best_effort_ok = False
            if self._accumulated and self._message_id:
                try:
                    _best_effort_ok = bool(
                        await self._send_or_edit(
                            self._accumulated, finalize=True, is_turn_final=False,
                        )
                    )
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
                self._final_content_delivered = True
        except Exception as e:
            logger.error("Stream consumer error: %s", e)

    # Strip MEDIA:<path> tags before display. Uses the shared anchored
    # MEDIA_TAG_CLEANUP_RE from gateway/platforms/base.py — only tags whose
    # path ends in a deliverable extension are removed, so an unknown-extension
    # path stays visible instead of being silently dropped (issue #34517).
    # Streaming and non-streaming paths share the same regex, so a tag is
    # treated identically whichever path delivered the text.
    _MEDIA_RE = MEDIA_TAG_CLEANUP_RE

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
        return _BasePlatformAdapter.strip_media_directives_for_display(text)

    async def _send_new_chunk(
        self,
        text: str,
        reply_to_id: Optional[str],
        *,
        final: bool = False,
    ) -> Optional[str]:
        """Send a new message chunk, optionally threaded to a previous message.

        Returns the message_id so callers can thread subsequent chunks.
        """
        text = self._clean_for_display(text)
        if not text.strip():
            return reply_to_id
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                reply_to=reply_to_id,
                metadata=self._metadata_for_send(final=final, expect_edits=True),
            )
            if result.success and result.message_id:
                self._message_id = str(result.message_id)
                self._track_preview_ids_from_result(result)
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
    def _split_text_chunks(
        text: str, limit: int,
        len_fn: "Callable[[str], int]" = len,
    ) -> list[str]:
        """Split text into reasonably sized chunks for fallback sends."""
        if len_fn(text) <= limit:
            return [text]
        chunks: list[str] = []
        remaining = text
        while len_fn(remaining) > limit:
            _cp_budget = _custom_unit_to_cp(remaining, limit, len_fn)
            split_at = remaining.rfind("\n", 0, _cp_budget)
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
            # Some platforms treat a successful streaming preview as durable
            # delivery. Telegram clients can instead lose or retain only part
            # of that preview after a failed final edit, so opt-in adapters
            # commit the completed answer with a fresh final send.
            if (
                final_text.strip()
                and final_text == self._visible_prefix()
                and getattr(
                    self.adapter,
                    "RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK",
                    False,
                ) is True
            ):
                delivery = await self._send_empty_fallback_final(final_text)
                if delivery == "delivered":
                    return
                self._already_sent = True
                self._fallback_prefix = ""
                self._fallback_preserve_partial_messages = False
                if delivery == "ambiguous":
                    # A timeout may mean Telegram accepted the send but the
                    # client never received the response. Preserve duplicate
                    # suppression for that one uncertain outcome.
                    self._final_content_delivered = True
                else:
                    # A confirmed failure leaves the gateway free to perform
                    # its normal final send.
                    self._final_response_sent = False
                    self._final_content_delivered = False
                return
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
                        result = await self._edit_message(
                            message_id=self._message_id,
                            content=clean_text,
                        )
                        if result.success:
                            self._last_sent_text = clean_text
                    except Exception:
                        pass
                self._already_sent = True
                self._final_response_sent = True
                self._final_content_delivered = True
                return

        raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        _len_fn: "Callable[[str], int]" = (
            self.adapter.message_len_fn
            if isinstance(self.adapter, _BasePlatformAdapter)
            else len
        )
        safe_limit = max(500, raw_limit - 100)
        chunks = self._split_text_chunks(continuation, safe_limit, len_fn=_len_fn)

        stale_message_id = self._message_id  # partial message to clean up
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
                    metadata=self._metadata_for_send(final=True),
                )
                if result.success:
                    break
                if attempt == 0 and self._is_flood_error(result):
                    delay = float(getattr(result, "retry_after", None) or 3.0)
                    logger.debug(
                        "Flood control on fallback send, retrying in %.1fs",
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    break  # non-flood error or second attempt failed

            if not result or not result.success:
                if sent_any_chunk:
                    # Some continuation text already reached the user, but not
                    # the full response. Do NOT set _final_response_sent — the
                    # base gateway final-send path should still deliver the
                    # complete response so the user gets the full answer.
                    # Suppress only _already_sent to avoid a duplicate send
                    # of the same partial content.
                    self._already_sent = True
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

        # Remove the frozen partial message so the user only sees the
        # complete fallback response.  ONLY safe when the fallback re-sent
        # the FULL final text (continuation == final_text).  When the
        # prefix-based dedup above sent only the missing TAIL, the partial
        # message IS the head of the answer — deleting it leaves the user
        # with only the last part of the response (the "Gemini sent only
        # the second half" symptom).  Best-effort — if the platform doesn't
        # implement ``delete_message``, the delete fails (flood control still
        # active, bot lacks permission, message too old to delete), the
        # partial remains but at least the full answer was delivered.
        if (
            stale_message_id
            and stale_message_id != last_message_id
            and not self._fallback_preserve_partial_messages
            and continuation == final_text
        ):
            delete_fn = getattr(self.adapter, "delete_message", None)
            if delete_fn is not None:
                try:
                    await delete_fn(self.chat_id, stale_message_id)
                except Exception as e:
                    logger.debug(
                        "Fallback partial cleanup failed (%s): %s",
                        stale_message_id, e,
                    )

        self._message_id = last_message_id
        self._already_sent = True
        self._final_response_sent = True
        self._final_content_delivered = True
        self._last_sent_text = chunks[-1]
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False

    async def _send_empty_fallback_final(self, final_text: str) -> str:
        """Commit a completed answer after Telegram finalization fails.

        Returns ``delivered`` on confirmed success, ``failed`` when the
        gateway can safely retry, and ``ambiguous`` when a timeout may have
        reached the platform already.
        """
        stale_ids = set(self._preview_message_ids)
        if self._message_id and self._message_id != "__no_edit__":
            stale_ids.add(str(self._message_id))

        result = None
        for attempt in range(2):
            try:
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=final_text,
                    metadata=self._metadata_for_send(final=True),
                )
            except Exception as exc:
                logger.debug("Empty fallback final send failed: %s", exc)
                return (
                    "ambiguous"
                    if self._send_failure_may_have_delivered(exc)
                    else "failed"
                )

            if getattr(result, "success", False):
                break
            if attempt == 0 and self._is_flood_error(result):
                delay = float(getattr(result, "retry_after", None) or 3.0)
                logger.debug(
                    "Flood control on empty fallback final send; retrying in %.1fs",
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            return (
                "ambiguous"
                if self._send_failure_may_have_delivered(result)
                else "failed"
            )

        new_message_id = getattr(result, "message_id", None)
        delete_fn = getattr(self.adapter, "delete_message", None)
        if delete_fn is not None:
            for stale_id in stale_ids:
                if not stale_id or stale_id == new_message_id:
                    continue
                try:
                    await delete_fn(self.chat_id, stale_id)
                except Exception as exc:
                    logger.debug(
                        "Empty fallback preview cleanup failed (%s): %s",
                        stale_id,
                        exc,
                    )

        self._preview_message_ids = set()
        self._message_id = new_message_id or "__no_edit__"
        self._already_sent = True
        self._final_response_sent = True
        self._final_content_delivered = True
        self._last_sent_text = final_text
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False
        self._notify_new_message()
        return "delivered"

    @staticmethod
    def _send_failure_may_have_delivered(result_or_exc: Any) -> bool:
        """Return True for timeout failures where retrying may duplicate."""
        if getattr(result_or_exc, "retryable", None) is True:
            return False
        error = str(getattr(result_or_exc, "error", None) or result_or_exc).lower()
        name = result_or_exc.__class__.__name__.lower()
        return "timeout" in error or "timed out" in error or "timeout" in name

    def _is_flood_error(self, result) -> bool:
        """Check if a SendResult failure is due to flood control / rate limiting."""
        err = getattr(result, "error", "") or ""
        err_lower = err.lower()
        return "flood" in err_lower or "retry after" in err_lower or "rate" in err_lower

    def _resolve_draft_streaming(self) -> bool:
        """Decide whether this run should use native draft streaming.

        Honors ``cfg.transport``:
          * ``"edit"``  → never use drafts (legacy progressive-edit path).
          * ``"draft"`` → require draft support; gracefully fall back to edit
            when the adapter declines.  Logs the downgrade at debug.
          * ``"auto"``  → use drafts when the adapter supports them for this
            chat type; otherwise edit.

        Adapter eligibility is checked via
        :meth:`BasePlatformAdapter.supports_draft_streaming`, which considers
        the chat type (e.g. Telegram drafts are DM-only) and platform-version
        gates (e.g. python-telegram-bot 22.6+).
        """
        transport = (self.cfg.transport or "edit").lower()
        if transport == "edit":
            return False
        # "off" is filtered upstream by the gateway; treat as edit defensively.
        if transport == "off":
            return False
        # Test adapters are MagicMocks that don't subclass BasePlatformAdapter;
        # default them to edit so existing test behaviour is preserved.
        if not isinstance(self.adapter, _BasePlatformAdapter):
            return False
        try:
            supported = self.adapter.supports_draft_streaming(
                chat_type=self.cfg.chat_type or None,
                metadata=self.metadata,
            )
        except Exception:
            logger.debug("supports_draft_streaming probe raised", exc_info=True)
            supported = False
        if not supported:
            if transport == "draft":
                logger.debug(
                    "Draft streaming requested but unsupported (chat=%s, type=%r) — "
                    "falling back to edit",
                    self.chat_id, self.cfg.chat_type,
                )
            return False
        return True

    async def _send_draft_frame(self, text: str) -> bool:
        """Emit a single animated draft frame for the current accumulated text.

        Returns True when the frame landed.  On any failure, permanently
        disables drafts for the remainder of this run so subsequent frames
        flow through the edit-based path (which can adapt with flood-control
        backoff, etc.).  Drafts have no message_id and clear naturally on
        the client when the response finalizes via a regular sendMessage.
        """
        if self._draft_id is None:
            # Defensive: should never happen — _use_draft_streaming gate is
            # set in tandem with _draft_id in run().  Disable to be safe.
            self._use_draft_streaming = False
            return False
        try:
            result = await self.adapter.send_draft(
                chat_id=self.chat_id,
                draft_id=self._draft_id,
                content=text,
                metadata=self.metadata,
            )
        except Exception as e:
            logger.debug(
                "send_draft raised, disabling draft transport for this run: %s", e,
            )
            self._draft_failures += 1
            self._use_draft_streaming = False
            return False
        if not getattr(result, "success", False):
            logger.debug(
                "send_draft returned success=False, disabling draft transport: %s",
                getattr(result, "error", "unknown"),
            )
            self._draft_failures += 1
            self._use_draft_streaming = False
            return False
        # Frame delivered.  Track text for parity with edit-based no-op skip.
        self._last_sent_text = text
        return True

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
            result = await self._edit_message(
                message_id=self._message_id,
                content=prefix,
            )
            if getattr(result, "success", False):
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
                # Record the exact delivered text so run.py can confirm whether
                # an interim "preview" actually carried the final response, vs.
                # unrelated commentary delivered during a session split (#14238).
                self._delivered_commentary_texts.append(text)
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

    def _raw_message_limit(self) -> int:
        """Per-message length budget (in the adapter's ``message_len_fn`` units)
        before the consumer splits an overflowing reply.

        Adapters with a richer send/draft path (e.g. Telegram rich messages)
        can raise this above ``MAX_MESSAGE_LENGTH`` via
        ``streaming_overflow_limit`` so a reply that fits one rich message isn't
        fragmented at the legacy edit limit.  Falls back to
        ``MAX_MESSAGE_LENGTH`` (4096 default) for everyone else.
        """
        base = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        # isinstance gate: MagicMock adapters return mock objects (truthy, not
        # ints) for arbitrary attribute access — keep them on the base limit.
        if isinstance(self.adapter, _BasePlatformAdapter):
            try:
                cap = self.adapter.streaming_overflow_limit()
            except Exception as e:
                logger.debug("streaming_overflow_limit check failed: %s", e)
                cap = None
            if isinstance(cap, int) and cap > base:
                return cap
        return base

    def _track_preview_id(self, message_id: Optional[str]) -> None:
        """Record a real preview message id for fresh-final cleanup."""
        if message_id and message_id != "__no_edit__":
            self._preview_message_ids.add(str(message_id))

    def _track_preview_ids_from_result(self, result: Any) -> None:
        """Record every message id a send/edit result exposes: the primary id
        plus any continuation ids from an oversized split
        (``continuation_message_ids`` or ``raw_response['message_ids']``)."""
        self._track_preview_id(getattr(result, "message_id", None))
        for mid in (getattr(result, "continuation_message_ids", None) or ()):
            self._track_preview_id(mid)
        raw = getattr(result, "raw_response", None) or {}
        if isinstance(raw, dict):
            for mid in (raw.get("message_ids") or ()):
                self._track_preview_id(mid)

    def _adapter_prefers_fresh_final(self, text: str) -> bool:
        """Return True when the adapter would rather finalize a streamed reply
        by sending a fresh message and deleting the preview than by editing the
        preview in place — e.g. Telegram, whose ``sendRichMessage`` send path
        currently renders richer markdown than Hermes' MarkdownV2 edit path.

        Returns False when there is no real preview to replace (no message id,
        or the ``__no_edit__`` sentinel), when the adapter doesn't expose the
        hook, or on any error (the consumer then keeps the edit-in-place path).
        """
        if not self._message_id or self._message_id == "__no_edit__":
            return False
        fn = getattr(self.adapter, "prefers_fresh_final_streaming", None)
        if fn is None:
            return False
        try:
            try:
                result = fn(text, metadata=self.metadata)
            except TypeError:
                # Adapter / test double whose hook doesn't accept the metadata
                # keyword — fall back to the positional-only form.
                result = fn(text)
        except Exception as e:
            logger.debug("prefers_fresh_final_streaming check failed: %s", e)
            return False
        # ``is True`` (not ``bool(...)``) so a MagicMock adapter's auto-child
        # method — truthy by default in tests — does not wrongly enable the
        # fresh-final path.  Mirrors the REQUIRES_EDIT_FINALIZE gate in __init__.
        return result is True

    async def _try_fresh_final(self, text: str, *, is_turn_final: bool = True) -> bool:
        """Send ``text`` as a brand-new message (best-effort delete the old
        preview) so the platform's visible timestamp reflects completion
        time.  Returns True on successful delivery, False on any failure so
        the caller falls back to the normal edit path.

        ``is_turn_final`` is False when finalizing an interim segment at a tool
        boundary (a preamble) rather than the turn-final answer; the
        final-delivery flag is then left unset so the gateway still delivers the
        real answer from the next API call (#29346).

        Ported from openclaw/openclaw#72038.
        """
        # Every preview message the user has seen for this response: the
        # current one plus any continuation fragments tracked while streaming
        # (an oversized reply split across the platform's edit limit).  All of
        # them are replaced by the single fresh message below.
        stale_ids = set(self._preview_message_ids)
        if self._message_id and self._message_id != "__no_edit__":
            stale_ids.add(self._message_id)
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                metadata=self._metadata_for_send(final=True),
            )
        except Exception as e:
            logger.debug("Fresh-final send failed, falling back to edit: %s", e)
            return False
        if not getattr(result, "success", False):
            return False
        # Adopt the new message id as the current message so subsequent
        # callers (e.g. overflow split loops, finalize retries) see a
        # consistent state.
        new_message_id = getattr(result, "message_id", None)
        # Successful fresh send — try to delete the stale preview(s) so the
        # user doesn't see the old edit-stuck message(s) underneath.  Cleanup
        # is best-effort; platforms that don't implement ``delete_message``
        # just leave the preview behind (still an acceptable outcome — the
        # visible final timestamp is the important part).  Never delete the
        # message we just sent.
        delete_fn = getattr(self.adapter, "delete_message", None)
        if delete_fn is not None:
            for stale_id in stale_ids:
                if not stale_id or stale_id == "__no_edit__" or stale_id == new_message_id:
                    continue
                try:
                    await delete_fn(self.chat_id, stale_id)
                except Exception as e:
                    logger.debug(
                        "Fresh-final preview cleanup failed (%s): %s",
                        stale_id, e,
                    )
        self._preview_message_ids = set()
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
        if is_turn_final:
            self._final_response_sent = True
        return True

    async def _suppress_silence_marker(self) -> None:
        """Retract any streamed preview when the final reply is a silence marker.

        The agent chose not to respond and emitted a bare control marker.  Any
        preview message the consumer already put on screen (a partial marker
        flushed on an interval tick, or a preamble before a tool boundary) must
        be removed so the raw marker is never left visible.  Deletion reuses the
        same best-effort ``delete_message`` path as :meth:`_try_fresh_final`.

        Crucially, the delivery flags (``_final_response_sent`` /
        ``_final_content_delivered``) are left **False**: nothing was delivered.
        The gateway then does not mistake the marker for a delivered reply, and
        its own whole-response filter turns the marker into "" so no fallback
        send happens either.  ``_already_sent`` is likewise cleared so the
        gateway's ``already_sent`` short-circuits do not fire.
        """
        stale_ids = set(self._preview_message_ids)
        if self._message_id and self._message_id != "__no_edit__":
            stale_ids.add(self._message_id)
        delete_fn = getattr(self.adapter, "delete_message", None)
        if delete_fn is not None:
            for stale_id in stale_ids:
                if not stale_id or stale_id == "__no_edit__":
                    continue
                try:
                    await delete_fn(self.chat_id, stale_id)
                except Exception as e:
                    logger.debug(
                        "Silence-marker preview cleanup failed (%s): %s",
                        stale_id, e,
                    )
        self._preview_message_ids = set()
        self._message_id = None
        self._accumulated = ""
        self._last_sent_text = ""
        self._already_sent = False
        self._final_response_sent = False
        self._final_content_delivered = False
        logger.info(
            "Suppressed streamed intentional-silence marker (chat=%s)",
            self.chat_id,
        )

    async def _send_or_edit(
        self, text: str, *, finalize: bool = False, is_turn_final: bool = True,
    ) -> bool:
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

        # Native draft streaming: route mid-stream frames through send_draft.
        # The final answer is delivered via the regular sendMessage path
        # below — drafts have no message_id so we can't finalize them
        # in-place; the regular sendMessage clears the draft naturally on
        # the client and gives the user a real message in their history.
        # Skip when:
        #   * finalize=True (this is the final answer; needs to be a real message)
        #   * an edit path is already established (message_id is set, e.g. after
        #     a tool-boundary segment break where the prior text was finalized
        #     as a real sendMessage and the next text segment continues editing
        #     that one — staying on edit-based for that segment is correct).
        if (
            self._use_draft_streaming
            and not finalize
            and self._message_id is None
        ):
            # No-op skip: identical to the last frame we sent.
            if text == self._last_sent_text:
                return True
            ok = await self._send_draft_frame(text)
            if ok:
                # Drafts mark "we put something on screen" but DO NOT set
                # _already_sent — that flag gates the gateway's fallback
                # final-send path and we still need that to fire so the
                # user gets a real message (drafts have no message_id).
                return True
            # Failure already disabled drafts for this run; fall through to
            # the regular edit/send path below.
        self._last_edit_overflowed = False
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
                    #
                    # Adapters can also opt in regardless of the time threshold
                    # via prefers_fresh_final_streaming (e.g. Telegram, whose
                    # send path renders richer markdown than its edit path):
                    # finalizing through edit would visibly downgrade a rich
                    # preview, so re-deliver as a fresh message + delete the
                    # preview instead.
                    #
                    # When the adapter exposes prefers_fresh_final_streaming
                    # and explicitly returns False, the time-based threshold
                    # must NOT override that decision.  On Telegram the
                    # fresh-final path sends a Rich Message (sendRichMessage)
                    # that overlaps with the legacy MarkdownV2 preview already
                    # visible from streaming — both remain on screen because
                    # the old message is only best-effort deleted.  Adapters
                    # without the hook still get the time-based fresh-final.
                    # (#47048)
                    # Check the *class* for the hook so MagicMock adapters
                    # (which auto-create attributes on access) are not
                    # falsely detected as having it.  Also check instance
                    # __dict__ for test doubles that explicitly assign the
                    # attribute (e.g. adapter.prefers_fresh_final_streaming
                    # = MagicMock(return_value=False)).
                    _has_prefers_hook = (
                        hasattr(type(self.adapter),
                                "prefers_fresh_final_streaming")
                        or "prefers_fresh_final_streaming"
                            in getattr(self.adapter, "__dict__", {})
                    )
                    _prefers_fresh = self._adapter_prefers_fresh_final(text)
                    if (
                        finalize
                        and (
                            _prefers_fresh
                            or (
                                not _has_prefers_hook
                                and self._should_send_fresh_final()
                            )
                        )
                        and await self._try_fresh_final(
                            text, is_turn_final=is_turn_final,
                        )
                    ):
                        return True
                    # Edit existing message
                    result = await self._edit_message(
                        message_id=self._message_id,
                        content=text,
                        finalize=finalize,
                    )
                    if result.success:
                        self._already_sent = True
                        # Record any continuation fragments an oversized edit
                        # split off, so fresh-final can clean them all up.
                        self._track_preview_ids_from_result(result)
                        # Adapter may have split-and-delivered an oversized
                        # edit across the original message + N continuations.
                        # When that happens, ``message_id`` is the LAST visible
                        # continuation and ``_last_sent_text`` no longer reflects
                        # the on-screen content (the new message only holds the
                        # final chunk's text), so subsequent edits must target
                        # the new id and skip-if-same comparisons must reset.
                        # Fire on_new_message so tool-progress bubbles linearize
                        # below the new continuation, not the original.
                        # ``getattr`` with default keeps backwards compat with
                        # SimpleNamespace mocks in tests that pre-date the field.
                        _continuation_ids = getattr(result, "continuation_message_ids", ()) or ()
                        if (
                            _continuation_ids
                            and result.message_id
                            and result.message_id != self._message_id
                        ):
                            self._last_edit_overflowed = True
                            self._message_id = str(result.message_id)
                            self._message_created_ts = time.monotonic()
                            self._last_sent_text = ""
                            self._notify_new_message()
                        else:
                            self._last_sent_text = text
                        # Successful edit — reset flood strike counter
                        self._flood_strikes = 0
                        return True
                    else:
                        immediate_final_fallback = False
                        if (
                            finalize
                            and is_turn_final
                            and self.cfg.cursor
                            and self._last_sent_text.endswith(self.cfg.cursor)
                            and self._visible_prefix() == text
                        ):
                            # The final clean-up edit failed, but the complete
                            # answer is already visible from the last streaming
                            # frame (usually with only the cursor still stuck on
                            # screen).  Mark the content delivered so the
                            # gateway suppresses its normal full final send;
                            # otherwise users see the same long answer twice
                            # when Telegram/Discord rate-limit this cosmetic
                            # final edit (#36965, #25349).
                            self._final_content_delivered = True
                        raw_response = getattr(result, "raw_response", None)
                        if isinstance(raw_response, dict) and raw_response.get("partial_overflow"):
                            # Telegram edited/sent one or more overflow chunks,
                            # but not the complete response.  Preserve the
                            # visible prefix so the got_done fallback sends the
                            # missing tail instead of marking a clipped topic
                            # reply as final delivery.
                            self._message_id = str(
                                raw_response.get("last_message_id")
                                or result.message_id
                                or self._message_id
                            )
                            delivered_prefix = raw_response.get("delivered_prefix")
                            if isinstance(delivered_prefix, str) and delivered_prefix:
                                self._last_sent_text = delivered_prefix
                                self._fallback_prefix = delivered_prefix
                                self._fallback_preserve_partial_messages = text.startswith(
                                    delivered_prefix
                                )
                            else:
                                self._fallback_prefix = self._visible_prefix()
                                self._fallback_preserve_partial_messages = False
                            self._fallback_final_send = True
                            self._edit_supported = False
                            self._already_sent = True
                            if getattr(result, "continuation_message_ids", ()):
                                self._notify_new_message()
                            return False

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
                            immediate_final_fallback = (
                                finalize
                                and is_turn_final
                                and getattr(
                                    self.adapter,
                                    "FALLBACK_ON_FINAL_EDIT_FLOOD",
                                    False,
                                ) is True
                            )
                            if (
                                self._flood_strikes < self._MAX_FLOOD_STRIKES
                                and not immediate_final_fallback
                            ):
                                # Don't disable edits yet — just slow down.
                                # Update _last_edit_time so the next edit
                                # respects the new interval.
                                self._last_edit_time = time.monotonic()
                                return False

                            if immediate_final_fallback:
                                logger.debug(
                                    "Turn-final edit hit flood control; "
                                    "entering fallback immediately"
                                )

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
                        # message so the user doesn't see a stuck ▉. A
                        # turn-final Telegram flood skips this cosmetic edit:
                        # another edit would consume the same flood budget and
                        # delay the fallback send that carries the answer.
                        if not immediate_final_fallback:
                            await self._try_strip_cursor()
                        return False
                else:
                    # Editing not supported — skip intermediate updates.
                    # The final response will be sent by the fallback path.
                    return False
            else:
                # First message — send new, threaded to the original user message
                # so it lands in the correct topic/thread.
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=text,
                    reply_to=self._initial_reply_to_id,
                    metadata=self._metadata_for_send(
                        final=finalize,
                        expect_edits=True,
                    ),
                )
                if result.success:
                    if result.message_id:
                        self._message_id = result.message_id
                        # Track when the preview first became visible to
                        # the user so fresh-final logic can detect stale
                        # preview timestamps on long-running responses.
                        self._message_created_ts = time.monotonic()
                        # Record this (and any continuation fragments from an
                        # oversized first send) for fresh-final cleanup.
                        self._track_preview_ids_from_result(result)
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
