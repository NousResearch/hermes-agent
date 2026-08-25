"""Gateway streaming consumer — bridges sync agent callbacks to async platform delivery.

on_delta() queues deltas from the agent's worker thread; the async run() task buffers,
rate-limits and progressively edits one platform message (send, then editMessageText;
draft/native transports are optional per adapter).
Credit: jobless0x (#774, #1312), OutThisLife (#798), clicksingh (#697).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import logging
import queue
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from gateway.platforms.base import BasePlatformAdapter as _BasePlatformAdapter
from gateway.platforms.base import _custom_unit_to_cp
from gateway.config import (
    DEFAULT_STREAMING_EDIT_INTERVAL as _DEFAULT_STREAMING_EDIT_INTERVAL,
    DEFAULT_STREAMING_BUFFER_THRESHOLD as _DEFAULT_STREAMING_BUFFER_THRESHOLD,
    DEFAULT_STREAMING_CURSOR as _DEFAULT_STREAMING_CURSOR)
from gateway.response_filters import (
    is_intentional_silence_response as _is_intentional_silence_response,
    is_partial_silence_marker as _is_partial_silence_marker)
from gateway.stream_consumer_fences import ensure_closed_code_fences
from gateway.stream_consumer_transport import StreamTransportMixin
from gateway.stream_consumer_fallback import StreamFallbackMixin
from gateway.stream_consumer_think import StreamThinkFilterMixin

logger = logging.getLogger("gateway.stream_consumer")

# Queue sentinels (see _drain_queue()).  Bare: _DONE, _NEW_SEGMENT (finalize, start a
# fresh message), _REOPEN_SEED (EAGER native re-seed after a clarify answer — WeCom
# typing is driven by the seed frame; lazy re-seed measured 48s of dead air).  Tuples:
# (_COMMENTARY, text); (_TOOL_PROGRESS, line) native-bubble overlay; (_FINAL_TEXT, text)
# authoritative final_response incl. post-stream augmentation, queued just before _DONE;
# (_FLUSH, threading.Event) barrier; (_APPROVAL_BOUNDARY, future, cancelled_flag).
_DONE = object()
_NEW_SEGMENT = object()
_COMMENTARY = object()
# Authoritative turn-final payload, enqueued by ``finish(final_text=...)``
# just before ``_DONE``.  Carries the completed ``final_response`` —
# including post-stream augmentation (file-mutation verifier footer,
# turn-completion explainer) — so the finalize/seal delivers the TRUE final
# and the recorded payload reconciles (#71643 / live finding #11: the
# footer-bearing final previously arrived only via a separate plain send).
_FINAL_TEXT = object()

# Queue marker for a synchronous flush barrier.  Enqueued as
# ``(_FLUSH, threading.Event)``; the drain loop finalizes and delivers any
# buffered segment, then sets the event.  A caller on the agent worker thread
# uses this (via ``flush_pending_sync``) to block until everything queued
# BEFORE the marker has actually landed on the platform — needed before
# sending a blocking interactive prompt (clarify poll) so the prompt is the
# last thing on screen, not racing ahead of buffered prose.
_FLUSH = object()
_APPROVAL_BOUNDARY = object()
_REOPEN_SEED = object()
_FUTURE_TYPES = (asyncio.Future, concurrent.futures.Future)

# Boundary finalize text when nothing has accumulated yet (overridable per boundary).
_DEFAULT_BOUNDARY_PLACEHOLDER = "⏸ 等待审批中..."


@dataclass
class StreamConsumerConfig:
    """Runtime config for a single stream consumer instance."""
    edit_interval: float = _DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = _DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = _DEFAULT_STREAMING_CURSOR
    buffer_only: bool = False
    # >0: final goes out as a fresh message once the preview has been visible this
    # long (timestamp reflects completion); 0 = always edit in place.
    # This makes the platform's visible timestamp reflect completion time instead of first-token time for
    # long-running responses (e.g. reasoning models that stream slowly). Ported from
    # openclaw/openclaw#72038. The gateway enables this selectively per-platform.
    fresh_final_after_seconds: float = 0.0
    # "auto"/"draft": native drafts when adapter+chat support it, else "edit"
    # (progressive editMessageText).  "off" is handled by the gateway.
    transport: str = "edit"
    chat_type: str = ""  # originating chat type; gates platform-specific drafts


@dataclass
class _Tick:
    """Everything one drain of the queue decided."""
    got_done: bool = False
    got_segment_break: bool = False
    got_flush: bool = False
    flush_event: Any = None
    got_reopen_seed: bool = False
    approval_boundary: Optional[tuple] = None  # (future, cancelled_flag)
    commentary_text: Optional[str] = None
    # Set by _push_update for _finalize_turn / _end_segment.
    update_visible: bool = False
    draft_final_fresh_send: bool = False

    @property
    def is_interim(self) -> bool:
        """Mid-stream tick: not finalizing, not a segment break, no commentary."""
        return not self.got_done and not self.got_segment_break and self.commentary_text is None


class GatewayStreamConsumer(StreamTransportMixin, StreamFallbackMixin, StreamThinkFilterMixin):
    """Async consumer that progressively edits a platform message with streamed tokens.
    Usage: ``agent.stream_delta_callback = consumer.on_delta``; ``create_task(consumer.run())``;
    after the agent finishes ``consumer.finish()`` then ``await task`` for the final edit."""

    _MAX_FLOOD_STRIKES = 3  # consecutive flood failures before edits are disabled

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
    #
    # Seeded from a RANDOM process nonce, not zero and not the clock (PR
    # 85796 review, B3 + r2 follow-up): draft_id is the wire identity for
    # the relay connector's per-(channel, draft_id) sealed-stream
    # tombstones, which outlive this process. Relay gateways are
    # disposable by design (scale-to-zero), so a counter restarting at 1
    # replays ids the connector already sealed — it then answers frames
    # from the NEW turn out of the OLD tombstone (zero platform calls,
    # old message identity) and the user's reply is silently dropped.
    # An epoch-ms seed (the first fix) still collides on same-millisecond
    # starts, forks, and clock steps; 49 random bits make collision
    # probability negligible while keeping ids + realistic turn counts
    # comfortably inside the connector's JS number range (2^53).
    _draft_id_counter: int = secrets.randbits(49)

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        config: Optional[StreamConsumerConfig] = None,
        metadata: Optional[dict] = None,
        on_new_message: Optional[callable] = None,
        on_before_finalize: Optional[Callable[[], Any]] = None,
        initial_reply_to_id: Optional[str] = None,
        run_still_current: Optional[Callable[[], bool]] = None):
        self.adapter = adapter
        self.chat_id = chat_id
        self.cfg = config or StreamConsumerConfig()
        self.metadata = metadata
        # Hooks (exceptions swallowed): on_new_message per fresh content bubble (next
        # tool-progress bubble goes BELOW it); on_before_finalize once (pause typing).
        self._on_new_message = on_new_message
        self._on_before_finalize = on_before_finalize
        self._initial_reply_to_id = initial_reply_to_id
        self._turn_id = str(uuid.uuid4())  # keys send_stream_frame() per concurrent consumer
        # Returns False after /new or /stop; run() then abandons the stream.
        self._run_still_current = run_still_current or (lambda: True)
        # Whether this consumer is fed the final reply's stream deltas. A consumer built only to
        # relay interim commentary (text streaming off, ``display.interim_assistant_messages`` on)
        # never receives the final's deltas, so the duplicate-risk diagnostic in
        # ``_run_agent_mark_streamed_delivery`` must not fire for it (#105341). Default True: every
        # other construction site (incl. the proxy path) creates consumers only when streaming is on.
        self.stream_deltas_enabled = True
        # Only platforms needing an explicit finalize call (DingTalk AI Cards) force a
        # redundant final edit; ``is True`` keeps MagicMock adapters out.
        self._adapter_requires_finalize = getattr(adapter, "REQUIRES_EDIT_FINALIZE", False) is True
        # Telegram bounds edit retries at 5s; a fallback must not wait longer.
        self._max_fallback_flood_retry_seconds = 5.0

        self._queue: queue.Queue = queue.Queue()
        # Every real preview id on screen this response (fresh-final deletes them all);
        # the per-segment set holds only the active segment so failure recovery never
        # deletes an earlier finalized preamble/commentary.
        # Wall-clock timestamp (time.monotonic) when ``_message_id`` was first assigned from a successful
        # first-send. Used by the fresh-final logic to detect long-lived previews whose edit timestamps
        # would be stale by completion time. Ported from openclaw/openclaw#72038.
        self._preview_message_ids: "set[str]" = set()
        self._already_sent = False
        self._edit_supported = True  # False once progressive edits stop working
        self._last_edit_time = 0.0
        self._last_edit_overflowed = False  # last _send_or_edit split into continuations
        self._flood_strikes = 0
        self._current_edit_interval = self.cfg.edit_interval  # adaptive backoff
        self._delivered_commentary_texts: list[str] = []
        self._delivered_segment_texts: list[str] = []  # finalized text per past segment
        self._in_think_block = False  # think-tag filter state (mirrors CLI _stream_delta)
        self._think_buffer = ""
        self._before_finalize_notified = False
        self._reset_message_state()

        # Transports, resolved in run().  Draft: animated frames via adapter.send_draft;
        # the final still uses first-send; the first failure disables drafts.  Native
        # (WeCom msgtype "stream"): the ONLY channel — any failure falls back to edit/send.
        self._use_draft_streaming = False
        self._draft_id: Optional[int] = None
        self._draft_failures = 0
        # TERMINAL authorization refusal for THIS RUN (see _send_draft_frame).
        # Per-run state, constructed fresh each turn, so a refusal can never
        # mute a healthy destination on a later turn.
        self._egress_declined = False
        self._use_native_streaming = False
        self._native_stream_opened = False  # seed sent: bubble open, zero content
        self._native_last_pushed_len = 0    # throttle under WeCom's 30 frames/min
        # Boundary state from close_for_approval_prompt() (boundaries are processed
        # serially).  reopen=True (clarify) keeps native enabled so post-prompt output
        # re-opens a fresh stream; approval degrades to send().
        self._boundary_placeholder = _DEFAULT_BOUNDARY_PLACEHOLDER
        self._boundary_reason = "Approval"
        self._boundary_reopen = False
        # Reopen requested but nothing re-seeded: got_done must not open a stream just
        # to emit a lone "✅"; an EAGER re-seed opened a bubble that got_done MUST close.
        self._awaiting_reopen_after_boundary = False
        self._reopen_seeded_eagerly = False

    def _stream_is_message(self) -> bool:
        """Whether THIS chat's transport treats the stream as the message.

        Prefers the adapter's per-chat probe (multi-platform relay: one
        adapter fronts N platforms, and the class attribute can only
        reflect the primary identity — review r2, finding 2). Falls back
        to the legacy attribute for adapters without the probe. Both are
        resolved on the CLASS to stay MagicMock-safe (auto-created
        instance attributes are truthy).
        """
        probe = getattr(type(self.adapter), "stream_is_message_for_chat", None)
        if callable(probe):
            try:
                return probe(self.adapter, str(self.chat_id)) is True
            except Exception:
                return False
        return getattr(self.adapter, "draft_stream_is_message", False) is True

    def _metadata_for_send(
        self,
        *,
        final: bool = False,
        expect_edits: bool = False,
    ) -> dict | None:
        """Return per-send metadata for stream-created messages.

    def _clear_turn_final_flags(self) -> None:
        """Reset every turn-final delivery flag to "nothing delivered yet".
        ``_delivered_final_text`` is the cleaned turn-final payload the gateway compares to
        the completed final_response before trusting the flags (a successful finalize edit
        may carry a stale preview); None = legacy trust.  A payload-less
        ``_turn_split_delivery`` must NOT inherit legacy trust; ``_delivery_ambiguous`` (a
        full-final send timed out but MAY have landed) is the only case that does."""
        # #29346: a tool/segment boundary means what we delivered was an interim preamble, not the final
        # answer — clear the flags so a premature setter can't fool the gateway. Safe: got_done returns
        # before any reset, and run.py reads these only after the consumer task exits.
        self._final_response_sent = False
        self._final_content_delivered = False  # content landed even if the cosmetic edit failed
        self._delivered_final_text: Optional[str] = None
        self._turn_split_delivery = False
        # True when a full-final send timed out in a way that MAY have reached the platform
        # (``_send_empty_fallback_final`` → "ambiguous"). The only case where a payload-less delivery flag
        # keeps legacy trust in ``delivered_final_matches`` (#95382 tightening) — re-sending there risks a
        # duplicate rather than recovering a loss.
        self._delivery_ambiguous = False

    def _stream_is_message(self) -> bool:
        """Whether THIS chat's transport treats the stream as the message: per-chat probe
        first (a relay adapter's class attribute only reflects its primary identity), else
        the legacy attribute; both on the CLASS (MagicMock-safe)."""
        probe = getattr(type(self.adapter), "stream_is_message_for_chat", None)
        if not callable(probe):
            return getattr(self.adapter, "draft_stream_is_message", False) is True
        try:
            return probe(self.adapter, str(self.chat_id)) is True
        except Exception:
            return False

    @property
    def accepts_tool_progress(self) -> bool:
        """True only when native streaming is active (gates in-stream tool progress)."""
        return self._use_native_streaming

    def on_tool_progress(self, line: str) -> None:
        """Thread-safe: overlay a tool-progress line in the native bubble until the next delta."""
        if line:
            self._queue.put((_TOOL_PROGRESS, line))

    def _compose_frame_content(self) -> str:
        """Native frame content: text, with any tool-progress lines below a rule."""
        progress = "\n".join(self._tool_progress_lines)
        return "\n\n---\n".join(p for p in (self._accumulated, progress) if p)

    def _metadata_for_send(self, *, final: bool = False, expect_edits: bool = False) -> dict | None:
        """Per-send metadata.  ``final`` → notify=True (Mattermost treats notify-worthy sends
        as final when a broken thread root may fall back flat); ``expect_edits`` keeps
        editable previews on Telegram's legacy send path."""
        meta = dict(self.metadata) if self.metadata else {}
        if self._initial_reply_to_id:
            meta["reply_to_message_id"] = self._initial_reply_to_id
        if expect_edits:
            meta["expect_edits"] = True
        if final:
            meta["notify"] = True
        return meta or None

    # Read-only views for the gateway (flag semantics: see _clear_turn_final_flags).
    already_sent = property(lambda self: self._already_sent)
    final_response_sent = property(lambda self: self._final_response_sent)
    message_id = property(lambda self: self._message_id)
    final_content_delivered = property(lambda self: self._final_content_delivered)

    async def _notify_before_finalize(self) -> None:
        """Run the pre-finalize hook exactly once, swallowing hook errors."""
        if self._before_finalize_notified:
            return
        self._before_finalize_notified = True
        if self._on_before_finalize is not None:
            with contextlib.suppress(Exception):
                result = self._on_before_finalize()
                if inspect.isawaitable(result):
                    await result

    def _append_accumulated(self, text: str) -> None:
        """Append to the live buffer and the split-stable stream ledger."""
        if not text:
            return
        if self._tool_progress_lines:  # real text overwrites the overlay
            self._tool_progress_lines.clear()
            self._tool_progress_active = False
        self._accumulated += text
        self._stream_ledger += text

    def _mark_skip_redundant_finalize(self) -> None:
        """Mark the turn final as delivered by a prior mid-stream edit.  Records what was
        ACKED on the wire, not ``_accumulated``: a throttled stream's last ack may be an
        older cursor-suffixed preview, which must not suppress the corrective send."""
        acked = self._last_sent_text or self._accumulated
        if self.cfg.cursor and acked.endswith(self.cfg.cursor):
            acked = acked[: -len(self.cfg.cursor)]
        self._mark_final_delivered(record=acked)

    def _mark_final_delivered(self, record: Optional[str] = None) -> None:
        """Set both turn-final flags; ``record`` also records the delivered payload."""
        self._final_response_sent = True
        # Only claim final delivery if the sealed chunks and final tail actually landed. ``_already_sent``
        # may be True from prior progress/fallback state (#10748).
        # The final clean-up edit failed, but the complete answer is already visible from the last streaming
        # frame (usually with only the cursor still stuck on screen). Mark the content delivered so the
        # gateway suppresses its normal full final send; otherwise users see the same long answer twice when
        # Telegram/Discord rate-limit this cosmetic final edit (#36965, #25349).
        self._final_content_delivered = True
        if record is not None:
            self._record_turn_final_payload(record)

    def _display_payload(self, text: str) -> str:
        """Normalize like ``_send_or_edit`` output: directive strip + fence close + strip."""
        return ensure_closed_code_fences(self._clean_for_display(text or "")).strip()

    def _record_turn_final_payload(self, text: str) -> None:
        """Record what the user actually saw as this turn's final answer.  On a split ``text``
        is only the trailing chunk, so the un-truncated ``_stream_ledger`` is recorded — else
        the gateway sees a mismatch and re-sends an answer the user already received."""
        if self._turn_split_delivery and self._stream_ledger:
            text = self._stream_ledger
        self._delivered_final_text = self._display_payload(text)

    def delivered_final_matches(self, final_text: str) -> Optional[bool]:
        """Tri-state reconcile of the recorded turn-final payload against ``final_text`` (a
        *successful* finalize edit can still carry a stale preview, so call success alone
        must not confirm delivery).  True: recorded payload (or an earlier segment /
        commentary) matches.  False: payload differs, or payload-less split.  None: nothing
        recorded on a legacy/ambiguous path (caller trusts flags)."""
        target = self._display_payload(final_text)
        if not target:
            return None
        if self._delivered_final_text is not None:
            # A segment break / commentary may have delivered it under another record.
            return (self._delivered_final_text.strip() == target
                    or self.has_delivered_text(final_text))
        if self._turn_split_delivery:
            return False
        # No recorded payload: judge against the FINAL content, not the flag.
        # ``_already_sent`` gates the match: draft frames set ``_last_sent_text`` but
        # deliberately not ``_already_sent``.
        # #95382 / #98552 class fix: a delivery flag with NO recorded payload must still be judged against
        # the FINAL content, not trusted blindly. Every internal flag-setting site records a payload; a
        # record-less consumer whose visible/streamed text does not contain the completed response has
        # demonstrably NOT delivered it (first-edit prefix, mid-stream truncation) — the flag alone must not
        # suppress the corrective send. ``_already_sent`` gates the visible-text match: draft frames set
        # ``_last_sent_text`` for dedupe but are ephemeral (they deliberately do not set ``_already_sent``),
        # so draft-only visibility must not count as durable delivery.
        if self._already_sent and self.has_delivered_text(final_text):
            return True
        # Only a timed-out full-final send that MAY have landed keeps legacy trust.
        return None if self._delivery_ambiguous else False

    def has_delivered_text(self, text: str) -> bool:
        """Return True if *text* was already delivered as visible chat content."""
        target = self._clean_for_display(text or "").strip()
        seen = (self._visible_prefix(), *self._delivered_commentary_texts,
                *self._delivered_segment_texts)
        return bool(target) and any(sent.strip() == target for sent in seen)

    def on_segment_break(self) -> None:
        """Finalize the current stream segment and start a fresh message."""
        self._queue.put(_NEW_SEGMENT)

    def close_for_approval_prompt(
        self, placeholder: str | None = None, reason: str = "Approval", reopen: bool = False,
    ) -> asyncio.Future:
        """Queue an interaction boundary (approval / clarify prompt) from sync context.
        run() finalizes the current native stream (``placeholder`` when empty), then per
        ``reopen``: False (approval; unbounded waits) degrades to one send() at got_done;
        True (clarify) keeps native enabled so post-prompt output re-opens a fresh stream.
        Returns (Future, cancelled_flag); the Future resolves True once processed
        (cancelled_flag is legacy, no longer read).  Without native streaming returns a
        bare, already-resolved Future."""
        loop = None
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()
        boundary_future = loop.create_future() if loop else concurrent.futures.Future()
        if not self._use_native_streaming:
            boundary_future.set_result(True)
            return boundary_future
        # Instance attributes are race-free: boundaries are processed one at a time.
        self._boundary_placeholder = placeholder or _DEFAULT_BOUNDARY_PLACEHOLDER
        self._boundary_reason = reason or "Approval"
        self._boundary_reopen = bool(reopen)
        cancelled_flag = {"cancelled": False}
        self._queue.put((_APPROVAL_BOUNDARY, boundary_future, cancelled_flag))
        return boundary_future, cancelled_flag

    def on_commentary(self, text: str) -> None:
        """Queue a completed interim assistant commentary message."""
        if text:
            self._queue.put((_COMMENTARY, text))

    def flush_pending_sync(self, timeout: float = 5.0) -> bool:
        """Block the agent worker thread until everything queued so far is delivered:
        ``(_FLUSH, Event)`` barrier — run() drains earlier items (FIFO), finalizes the
        segment, sets the event.  False on timeout (consumer task may not be running)."""
        evt = threading.Event()
        try:
            self._queue.put((_FLUSH, evt))
        except Exception:
            return False
        return evt.wait(timeout=max(0.0, float(timeout)))

    def _reopen_seed_pending(self) -> bool:
        """Native stream, reopen requested after a boundary, nothing open yet."""
        return (self._use_native_streaming and self._awaiting_reopen_after_boundary
                and not self._native_stream_opened)

    def request_reopen_seed(self) -> None:
        """Thread-safe: request an EAGER native re-seed after a clarify answer.  No-op unless
        reopen-pending, so a stray call can't open a spurious bubble mid-stream or on approval."""
        if self._reopen_seed_pending():
            self._queue.put(_REOPEN_SEED)

    def _notify_new_message(self) -> None:
        """Fire the on_new_message callback, swallowing any errors."""
        try:
            if self._on_new_message is not None:
                self._on_new_message()
        except Exception:
            logger.debug("on_new_message callback error", exc_info=True)

    @staticmethod
    def _signal_flush(flush_event) -> None:
        """Wake a thread blocked in flush_pending_sync(), swallowing errors.  Every loop path
        that consumed a ``_FLUSH`` barrier (incl. early ``continue``) must call this; a
        missed set stalls the caller for the full timeout."""
        if flush_event is not None:
            with contextlib.suppress(Exception):
                flush_event.set()

    def _reset_segment_state(self, *, preserve_no_edit: bool = False) -> None:
        if preserve_no_edit and self._message_id == "__no_edit__":
            return
        # Retain the finalized visible text of the current segment before
        # clearing ``_last_sent_text``, so ``has_delivered_text`` can still
        # match it after a segment break. (#65919 review)
        if self._last_sent_text:
            finalized = self._clean_for_display(self._last_sent_text).strip()
            if finalized:
                self._delivered_segment_texts.append(finalized)
        self._message_id = None
        self._message_created_ts = None
        self._accumulated = ""
        self._stream_ledger = ""
        self._last_sent_text = ""
        self._fallback_final_send = False
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False
        self._segment_preview_message_ids = set()
        # #29346: a tool/segment boundary means what we delivered was an interim
        # preamble, not the final answer — clear the flags so a premature setter
        # can't fool the gateway. Safe: got_done returns before any reset, and
        # run.py reads these only after the consumer task exits.
        self._final_response_sent = False
        self._final_content_delivered = False
        self._delivered_final_text = None
        self._turn_split_delivery = False
        # Native draft streaming: bump the draft_id so the next text segment
        # animates as a fresh preview below the tool-progress bubbles, not
        # over the prior segment's already-finalized draft.  This is how
        # we avoid the "inter-tool-call text leak" failure mode openclaw
        # documented in their issue #32535 — each text block becomes its
        # own visible message via the finalize, then a new draft animates
        # for the next one.
        if self._use_draft_streaming:
            # Finding #4 (live canary, Alice): for stream-is-the-message
            # adapters (relay Slack native streaming), a draft_id bump opens
            # a brand-new platform stream per tool boundary — the user saw
            # one frozen message per segment (each stuck with the streaming
            # cursor, never sealed) plus the real final. Those adapters keep
            # ONE stream per turn: tool progress lives in the native task
            # card, and the connector's suffix-delta logic appends each new
            # segment cleanly (prefix mismatch → whole-segment append).
            # Telegram-shaped drafts (clear + separate final) keep the bump.
            if not self._stream_is_message():
                type(self)._draft_id_counter += 1
                self._draft_id = type(self)._draft_id_counter

    def on_delta(self, text: str) -> None:
        """Thread-safe callback from the agent's worker thread.  ``None`` signals a tool
        boundary: the current message is finalized and subsequent text goes out as a new
        message below any tool-progress messages."""
        if text:
            self._queue.put(text)
        elif text is None:
            self.on_segment_break()

    def finish(self, final_text: Optional[str] = None) -> None:
        """Signal that the stream is complete.

        ``final_text``, when provided, is the AUTHORITATIVE completed
        ``final_response`` — including post-stream augmentation the
        accumulator never saw (file-mutation verifier footer,
        turn-completion explainer, plugin transforms).  The drain loop
        adopts it as the finalize payload so the sealed/edited message IS
        the true final and no separate corrective send is needed
        (live finding #11).  Callers that cannot know the final yet
        (interrupt/error paths) call ``finish()`` bare — legacy behavior.
        """
        if final_text is not None:
            self._queue.put((_FINAL_TEXT, final_text))
        self._queue.put(_DONE)

    async def run(self) -> None:
        """Async task that drains the queue and edits the platform message."""
        self._len_fn, self._safe_limit = self._resolve_length_budget()
        await self._start_transports()
        try:
            while True:
                # Session reset (/new, /stop): abandon rather than deliver stale deltas.
                if not self._run_still_current():
                    await self._abandon_native_stream()
                    return
                tick = self._drain_queue()

                # Drain all available items from the queue
                got_done = False
                got_segment_break = False
                got_flush = False
                flush_event = None
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
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _FINAL_TEXT:
                            # Authoritative turn-final payload (see finish()).
                            # Adopt it as the finalize content so the seal /
                            # final edit carries the TRUE final — including
                            # post-stream augmentation (verifier footer,
                            # completion explainer) the accumulator never saw.
                            # Only when this consumer actually streamed
                            # something this turn: a no-stream turn keeps the
                            # gateway's normal final-send path (adopting here
                            # would move delivery ownership for every
                            # non-streaming model). Skip on a multi-message
                            # split delivery: heads are already sealed on
                            # screen, so adopting the full final would repeat
                            # them inside the tail (#78541 shape).
                            _streamed_something = bool(
                                self._accumulated
                                or self._message_id
                                or self._last_sent_text
                            )
                            if _streamed_something and not self._turn_split_delivery:
                                _final_payload = self._clean_for_display(item[1])
                                _visible = self._clean_for_display(self._accumulated)
                                if _final_payload and _final_payload != _visible:
                                    self._accumulated = item[1]
                                    self._stream_ledger = item[1]
                            elif _streamed_something and self._turn_split_delivery:
                                # Split delivery + authoritative final (review
                                # r2, finding 3): wholesale adoption would
                                # repeat sealed heads inside the tail (#78541),
                                # but REFUSING entirely re-creates the #11
                                # duplicate one level up — a post-split footer
                                # never enters the ledger, delivered_final_
                                # matches reports a mismatch, and the gateway
                                # resends the ENTIRE body+footer. When the
                                # authoritative final strictly prefix-extends
                                # the split ledger, the missing suffix is the
                                # only undelivered content: append it to the
                                # live tail and the ledger, so the finalize
                                # carries it and the recorded payload
                                # reconciles. Non-prefix rewrites keep the
                                # full-resend fallback (can't patch a rewrite).
                                _final_raw = item[1]
                                _ledger = self._stream_ledger
                                if (
                                    _ledger
                                    and _final_raw.startswith(_ledger)
                                    and len(_final_raw) > len(_ledger)
                                ):
                                    _suffix = _final_raw[len(_ledger):]
                                    self._accumulated += _suffix
                                    self._stream_ledger = _final_raw
                            continue
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _COMMENTARY:
                            commentary_text = item[1]
                            break
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _FLUSH:
                            # Flush barrier: finalize the current segment like a
                            # tool boundary, then signal the waiting thread once
                            # delivery for this iteration has completed (below).
                            got_flush = True
                            got_segment_break = True
                            flush_event = item[1]
                            break
                        self._filter_and_accumulate(item)
                    except queue.Empty:
                        break

                if tick.got_done:
                    self._flush_think_buffer()
                    # A bare intentional-silence marker (NO_REPLY / [SILENT]): the
                    # gateway's whole-response filter runs too late for a streamed
                    # preview, so retract it here instead of finalizing.
                    if _is_intentional_silence_response(self._clean_for_display(self._accumulated)):
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
                # Whether the got_done update below was delivered as a FRESH
                # persistent send through the native-draft transport (drafts
                # have no message id, so the finalize tick is a brand-new
                # send that already carried finalize=True).  Distinguishes
                # that case from an EDIT issued while draft streaming is
                # active, which must keep the legacy explicit-finalize pass
                # for REQUIRES_EDIT_FINALIZE adapters.
                draft_final_fresh_send = False
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
                    # Overflow split.  Native streaming bypasses this: the adapter
                    # truncates against the stream protocol's own limit.
                    if not self._use_native_streaming and self._first_send_overflows():
                        if await self._split_first_send(tick):
                            return
                        continue
                    await self._seal_overflow_heads()
                    await self._push_update(tick)

                    display_text = self._accumulated
                    if not got_done and not got_segment_break and commentary_text is None:
                        display_text += self.cfg.cursor

                    # Segment break: finalize the current message so platforms
                    # that need explicit closure (e.g. DingTalk AI Cards) don't
                    # leave the previous segment stuck in a loading state when
                    # the next segment (tool progress, next chunk) creates a
                    # new message below it.  got_done has its own finalize
                    # path below so we don't finalize here for it.
                    draft_final_fresh_send = (
                        got_done
                        and self._use_draft_streaming
                        and self._message_id is None
                    )
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
                            self._record_turn_final_payload(self._accumulated)
                        elif (
                            current_update_visible
                            and (
                                not self._adapter_requires_finalize
                                or self._last_edit_overflowed
                                or draft_final_fresh_send
                            )
                        ):
                            # The update above already delivered the final
                            # accumulated content.  Native drafts have no
                            # message id, so their got_done update is a fresh,
                            # persistent send with finalize=True; running the
                            # adapter's explicit finalize hook immediately
                            # afterward would edit that already-final message
                            # a second time.  This is especially harmful for
                            # Telegram, where a successful sendRichMessage was
                            # being followed by editMessageText and could fall
                            # back to the legacy table-to-bullets formatter.
                            #
                            # Also skip the redundant final edit for adapters
                            # that don't need an explicit finalize signal, and
                            # for any adapter when the update split-and-
                            # delivered across continuations: that update
                            # carried finalize=True itself, and re-finalizing
                            # with the full text would overflow-split again into
                            # the adopted continuation, duplicating chunks.
                            #
                            # Delivery is recorded via the shared helper so
                            # the recorded payload is the last ACKED edit,
                            # not the accumulated text (frozen-preview
                            # incident class; see _mark_skip_redundant_finalize).
                            self._mark_skip_redundant_finalize()
                        elif self._message_id:
                            # Either the mid-stream edit didn't run (no
                            # visible update this tick) OR the adapter needs
                            # explicit finalize=True to close the stream.
                            self._final_response_sent = await self._send_or_edit(
                                self._accumulated, finalize=True,
                            )
                            if self._final_response_sent:
                                self._final_content_delivered = True
                                self._record_turn_final_payload(self._accumulated)
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
                            # Turn-final retry after the finalize tick above
                            # failed (transport error, seal exception).
                            # finalize=True so a stream-is-the-message adapter
                            # can never route this through the draft-frame
                            # branch: its no-op dedupe compares against the
                            # last UNSEALED frame and would report success
                            # without any transport call, recording a final
                            # the user never received (silent-loss class).
                            self._final_response_sent = await self._send_or_edit(
                                self._accumulated, finalize=True,
                            )
                            if self._final_response_sent:
                                self._final_content_delivered = True
                                self._record_turn_final_payload(self._accumulated)
                    return

                if commentary_text is not None:
                    # Stream-is-the-message adapters: commentary posts as its
                    # own message (no notify → no seal-interception), and the
                    # native stream continues cumulatively. Resetting here
                    # would break the append-only invariant the connector's
                    # delta computation depends on (whole-snapshot re-append).
                    _stream_is_msg_c = self._stream_is_message()
                    if _stream_is_msg_c and self._use_draft_streaming:
                        await self._send_commentary(commentary_text)
                        self._last_edit_time = time.monotonic()
                    else:
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
                    # Stream-is-the-message adapters keep one cumulative native
                    # stream for the whole turn. Clearing _accumulated here makes
                    # the next frame a non-prefix snapshot, so the connector's
                    # append fallback repeats the entire answer at every tool
                    # boundary. Preserve all stream state; only non-native draft
                    # and edit-based transports start a new segment.
                    # ``is True`` + _use_draft_streaming: MagicMock adapters
                    # return truthy auto-attributes, and an edit-based run on a
                    # stream-capable adapter still needs the legacy reset.
                    if (
                        self._stream_is_message()
                        and self._use_draft_streaming
                    ):
                        pass
                    else:
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

                # Flush barrier satisfied: the buffered segment (if any) has now
                # been finalized and delivered above, so wake the thread blocked
                # in flush_pending_sync().  Done last so the waiter only unblocks
                # once everything queued before the barrier is on screen.
                if got_flush:
                    self._signal_flush(flush_event)

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
            elif self._message_id is None:
                # Native draft path deliberately keeps _message_id=None, so
                # the best-effort edit above never runs for it — the stream
                # stayed visibly live (streaming indicator) forever and the
                # adapter kept armed interception state for the next turn
                # to inherit (review B8). Seal in place with what's already
                # on screen; sets no delivery flags.
                await self._abandon_native_stream()
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
                self._record_turn_final_payload(self._accumulated)
        except Exception as e:
            logger.error("Stream consumer error: %s", e)
        finally:
            self._wake_flush_waiters()

    # ── run() collaborators ─────────────────────────────────────────────

    def _resolve_length_budget(self) -> "tuple[Callable[[str], int], int]":
        """Per-chat length function (relay adapters differ per chat, e.g. utf16) + budget.
        isinstance gate: MagicMock auto-attributes aren't callables; test doubles use len."""
        len_fn = (self.adapter.message_len_fn_for_chat(self.chat_id)
                  if isinstance(self.adapter, _BasePlatformAdapter) else len)
        return len_fn, max(500, self._raw_message_limit() - len_fn(self.cfg.cursor) - 100)

    async def _start_transports(self) -> None:
        """Resolve native/draft transport; native wins (adapters declaring it can't edit).
        The empty seed frame shows "typing" before the first token; on failure → edit path."""
        self._use_native_streaming = self._resolve_native_streaming()
        if self._use_native_streaming:
            logger.debug("Stream consumer using native-stream transport (chat=%s)", self.chat_id)
            if await self._try_seed_frame("Native streaming seed frame raised; disabling native",
                                          exc_info=True):
                self._native_stream_opened = True
                self._use_draft_streaming = False
                return
            self._use_native_streaming = False
        self._use_draft_streaming = self._resolve_draft_streaming()
        # Native draft streaming: bump the draft_id so the next text segment animates as a fresh preview
        # below the tool-progress bubbles, not over the prior segment's already-finalized draft. This is how
        # we avoid the "inter-tool-call text leak" failure mode openclaw documented in their issue #32535 —
        # each text block becomes its own visible message via the finalize, then a new draft animates for
        # the next one.
        if self._use_draft_streaming:
            self._bump_draft_id()
            logger.debug("Stream consumer using native-draft transport (chat=%s draft_id=%s)",
                         self.chat_id, self._draft_id)

    def _drain_queue(self) -> "_Tick":
        """Drain everything queued so far into one tick.  Control sentinels stop the drain
        (they take effect this tick); _FINAL_TEXT / _TOOL_PROGRESS / text deltas fold into
        state so simultaneous items batch."""
        tick = _Tick()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return tick
            if item is _DONE:
                tick.got_done = True
                return tick
            if item is _NEW_SEGMENT:
                tick.got_segment_break = True
                return tick
            if item is _REOPEN_SEED:
                tick.got_reopen_seed = True
                return tick
            kind = item[0] if isinstance(item, tuple) and item else None
            if kind is _FINAL_TEXT:
                self._adopt_final_text(item[1])
            elif kind is _TOOL_PROGRESS:  # keep draining to batch simultaneous lines
                if self._use_native_streaming:
                    self._tool_progress_lines.append(item[1])
                    self._tool_progress_active = True
            elif kind is _APPROVAL_BOUNDARY:
                tick.approval_boundary = (item[1], item[2])
                return tick
            elif kind is _COMMENTARY:
                tick.commentary_text = item[1]
                return tick
            elif kind is _FLUSH:
                # Barrier: finalize like a tool boundary, signal at the end of the tick.
                tick.got_flush = tick.got_segment_break = True
                tick.flush_event = item[1]
                return tick
            else:
                self._filter_and_accumulate(item)

    def _adopt_final_text(self, final_raw: str) -> None:
        """Adopt the authoritative final (see finish()) as the finalize content — only if this
        consumer streamed something (a no-stream turn keeps the gateway's final-send
        ownership).  Split delivery: wholesale adoption would repeat sealed heads, refusing
        makes the gateway resend the ENTIRE body — so append only the suffix when the final
        strictly prefix-extends the ledger."""
        if not (self._accumulated or self._message_id or self._last_sent_text):
            return
        if not self._turn_split_delivery:
            final_payload = self._clean_for_display(final_raw)
            if final_payload and final_payload != self._clean_for_display(self._accumulated):
                self._accumulated = final_raw
                self._stream_ledger = final_raw
            return
        ledger = self._stream_ledger
        if ledger and final_raw.startswith(ledger) and len(final_raw) > len(ledger):
            self._accumulated += final_raw[len(ledger):]
            self._stream_ledger = final_raw

    async def _eager_reopen_seed(self) -> None:
        """Eager re-seed after a clarify answer (gate re-checked: state may have advanced).
        Trade-off: WeCom's ~6-minute stream limit (errcode 846608, from the FIRST frame)
        now starts at the reply instant; on expiry we degrade to send()."""
        if not self._reopen_seed_pending():
            return
        if await self._try_seed_frame("Eager reopen seed raised, disabling native: %s"):
            self._native_stream_opened = True
            self._native_last_pushed_len = 0
            self._awaiting_reopen_after_boundary = False
            self._reopen_seeded_eagerly = True
            logger.info("[latency] Eager re-seed after clarify answer "
                        "(typing bubble reopened immediately, turn=%s)", self._turn_id)
        else:
            # Degrade to a single buffered send(), like the approval path.
            self._degrade_native_to_buffered_send()

    def _should_edit(self, tick: "_Tick") -> bool:
        """Decide whether this tick flushes an edit/frame."""
        if not tick.is_interim:
            return True
        if self.cfg.buffer_only:
            return False
        if self._use_native_streaming:
            # No platform edit-rate limit: push every delta immediately.
            should_edit = bool(self._accumulated) or self._tool_progress_active
        else:
            elapsed = time.monotonic() - self._last_edit_time
            # buffer_threshold is a codepoint debounce heuristic, not a
            # platform-limit check (_len_fn is for overflow).
            should_edit = bool((elapsed >= self._current_edit_interval and self._accumulated)
                               or len(self._accumulated) >= self.cfg.buffer_threshold)
        # Defer mid-stream edits while the buffer could still resolve to a silence
        # marker ("NO"→"NO_REPLY"); got_done always resolves the buffer.
        return should_edit and not _is_partial_silence_marker(
            self._clean_for_display(self._accumulated))

    async def _split_first_send(self, tick: "_Tick") -> bool:
        """No message to edit yet and the buffer overflows: seal only the head chunks; the
        tail stays in _accumulated as the active preview later deltas edit in place.
        True when the turn finished here (the run loop returns)."""
        chunks = self._truncate_for_stream(self._accumulated, self._safe_limit, self._len_fn)
        if len(chunks) <= 1:
            # Malformed/legacy adapter result must still be splittable.
            chunks = self._split_text_chunks(self._accumulated, self._safe_limit, self._len_fn)
        reply_to = self._initial_reply_to_id
        heads_delivered = len(chunks) > 1
        for chunk in chunks[:-1]:
            new_id = await self._send_new_chunk(chunk, reply_to, final=tick.got_done)
            if new_id is None or new_id == reply_to:
                heads_delivered = False  # keep the full text intact for the gateway fallback
                break
            reply_to = new_id

        if heads_delivered:
            self._accumulated = chunks[-1]
            # Flag BEFORE the tail send: fresh-final replaces every tracked preview
            # with one message, which is only valid while the active message holds
            # the whole answer — deleting sealed heads drops delivered text.
            self._turn_split_delivery = True
        # Heads are sealed (or a later head failed): never edit a sealed message with
        # the unsplit payload — the tail is sent fresh, or the fallback path retries.
        self._message_id = None
        self._message_created_ts = None
        self._last_sent_text = ""
        self._last_edit_time = time.monotonic()
        if tick.got_done:
            tail_delivered = (not self._accumulated
                              or await self._send_or_edit(self._accumulated, finalize=True))
            # ``_already_sent`` may be True from prior state — only heads + tail count.
            self._final_response_sent = heads_delivered and tail_delivered
            if self._final_response_sent:
                self._turn_split_delivery = True
                self._mark_final_delivered(record=self._accumulated)
            return True
        if tick.got_segment_break:
            self._fallback_final_send = False
            self._fallback_prefix = ""
            if not self._accumulated:
                return False
        # Early `continue` skips the bottom-of-loop flush signal.
        if tick.got_flush:
            self._signal_flush(tick.flush_event)
        return False

    def _overflows(self) -> bool:
        return self._len_fn(self._accumulated) > self._safe_limit

    def _first_send_overflows(self) -> bool:
        return self._message_id is None and self._overflows()

    async def _seal_overflow_heads(self) -> None:
        """Existing message overflowing: seal it with the head, start a new message for the rest."""
        while self._overflows() and self._message_id is not None and self._edit_supported:
            cp_budget = _custom_unit_to_cp(self._accumulated, self._safe_limit, self._len_fn)
            split_at = self._accumulated.rfind("\n", 0, cp_budget)
            if split_at < cp_budget // 2:
                split_at = cp_budget
            chunk = self._accumulated[:split_at]
            # finalize=True: the sealed chunk is never edited again, so it needs its
            # rich-text pass now.  is_turn_final=False: a split head is not the
            # answer, so fresh-final must not mark the turn delivered on it.
            ok = await self._send_or_edit(chunk, finalize=True, is_turn_final=False)
            if self._fallback_final_send or not ok:
                break  # keep the full text intact for the fallback final send
            self._accumulated = self._accumulated[split_at:].lstrip("\n")
            self._message_id = None
            self._last_sent_text = ""
            self._turn_split_delivery = True

    async def _push_update(self, tick: "_Tick") -> None:
        """Send/edit this tick's visible text (cursor-suffixed unless finalizing)."""
        display_text = self._accumulated
        if tick.is_interim:
            if self._use_native_streaming:
                display_text = self._compose_frame_content()
                if display_text and self.cfg.cursor:
                    display_text += self.cfg.cursor
            else:
                display_text += self.cfg.cursor

        # A got_done FRESH send via the draft transport already carries finalize=True,
        # unlike an EDIT, which REQUIRES_EDIT_FINALIZE adapters still need a pass for.
        tick.draft_final_fresh_send = (tick.got_done and self._use_draft_streaming
                                       and self._message_id is None)
        # Segment break finalizes so platforms needing explicit closure (DingTalk AI
        # Cards) don't leave the segment stuck loading; it closes a preamble, not the
        # answer.
        tick.update_visible = await self._send_or_edit(
            display_text, finalize=tick.got_done or tick.got_segment_break,
            is_turn_final=tick.got_done)
        self._last_edit_time = time.monotonic()
        # Lines stay in _tool_progress_lines for the next compose.
        self._tool_progress_active = False

    async def _finalize_turn(self, tick: "_Tick") -> None:
        """got_done: final edit without cursor, or one continuation send if edits failed."""
        if self._accumulated or self._message_id is not None or self._already_sent:
            await self._notify_before_finalize()
        if self._reopen_seed_pending() and not self._accumulated:
            # Lazy reopen, no post-prompt content: nothing is open on screen, so
            # don't re-seed just to emit a lone "✅".
            logger.debug("Clarify reopen boundary with no post-prompt content "
                         "— skipping lone-placeholder finalize (turn=%s)", self._turn_id)
        elif (self._reopen_seeded_eagerly and self._native_stream_opened
              and not self._accumulated and not tick.update_visible):
            # Eager seed, no content: the typing bubble IS on screen and would hang
            # forever — close it with an empty finalize.  Delivery flags untouched.
            await self._close_empty_native_bubble("Eager-seed empty finalize failed: %s")
            logger.debug("Eager reopen seed but no post-answer content — "
                         "closed empty typing bubble (turn=%s)", self._turn_id)
        elif self._use_native_streaming:
            # Native streams MUST close with finish=true even when empty (tool-only
            # turns) — placeholder if needed.
            if not tick.update_visible:
                await self._finalize_edit(self._accumulated or "✅", record=False)
            else:
                self._mark_final_delivered()
        elif self._accumulated:
            await self._finalize_edit_path(tick)

    async def _finalize_edit_path(self, tick: "_Tick") -> None:
        """Edit-transport finalize (the non-native got_done branches, in priority order)."""
        if self._fallback_final_send:
            await self._send_fallback_final(self._accumulated)
        elif self._final_response_sent:
            # Fresh-final already delivered; a second finalize would duplicate.
            self._mark_final_delivered(record=self._accumulated)
        elif tick.update_visible and (not self._adapter_requires_finalize
                                      or self._last_edit_overflowed or tick.draft_final_fresh_send):
            # The update already delivered the final.  A second finalize would re-edit
            # it (Telegram: editMessageText after sendRichMessage falls back to the
            # legacy formatter) or overflow-split again, duplicating chunks.
            self._mark_skip_redundant_finalize()
        elif self._message_id:
            # No visible update this tick, or the adapter needs explicit finalize=True.
            # The edit may exhaust flood strikes → fallback mode: send the unsent tail.
            if not await self._finalize_edit(self._accumulated) and self._fallback_final_send:
                await self._send_fallback_final(self._accumulated)
        elif not self._already_sent:
            # Retry after the finalize tick failed.  finalize=True keeps stream-is-the-
            # message adapters out of the draft-frame branch, whose dedupe against the
            # last UNSEALED frame would report success with no transport call.
            await self._finalize_edit(self._accumulated)

    async def _finalize_edit(self, text: str, *, record: bool = True) -> bool:
        """finalize=True send_or_edit; on success mark the turn delivered (+ record payload)."""
        self._final_response_sent = await self._send_or_edit(text, finalize=True)
        if self._final_response_sent:
            self._mark_final_delivered(record=text if record else None)
        return self._final_response_sent

    def _cumulative_transport(self) -> bool:
        """Stream-is-the-message drafts and WeCom native: one append-only stream per turn."""
        stream_draft = self._stream_is_message() and self._use_draft_streaming
        return stream_draft or self._use_native_streaming

    async def _deliver_commentary(self, commentary_text: str) -> None:
        """Post commentary as its own message.  Cumulative transports keep the stream going —
        resetting _accumulated would break the append-only invariant / lose text."""
        cumulative = self._cumulative_transport()
        if not cumulative:
            self._reset_segment_state()
        await self._send_commentary(commentary_text)
        self._last_edit_time = time.monotonic()
        if not cumulative:
            self._reset_segment_state()

    async def _end_segment(self, tick: "_Tick") -> None:
        """Tool boundary: edit-based transports reset so the next chunk is a fresh message.
        Cumulative transports must NOT reset — clearing _accumulated makes the next frame a
        non-prefix snapshot and the connector re-appends the whole answer.  preserve_no_edit:
        "__no_edit__" (platform never returned a real id — Signal, github_comment webhook)
        must keep its sentinel or every tool boundary posts a new message; the
        continuation goes out once via _send_fallback_final."""
        if self._cumulative_transport():
            return
        # If the segment-break edit didn't land (flood control / fallback mode),
        # _accumulated holds unseen pre-boundary text — flush it before the reset.
        if (self._accumulated and not tick.update_visible and self._message_id
                and self._message_id != "__no_edit__"):
            await self._flush_segment_tail_on_edit_failure()
        self._reset_segment_state(preserve_no_edit=True)

    async def _on_cancelled(self) -> None:
        """Best-effort final edit on task cancel: finalize=True so REQUIRES_EDIT_FINALIZE
        platforms apply formatting; is_turn_final=False because this handler owns the flags.
        Only a successful edit confirms delivery — a partial send may be just "Let me
        search…", not the answer."""
        best_effort_ok = False
        if self._accumulated and self._message_id:
            with contextlib.suppress(Exception):
                best_effort_ok = bool(await self._send_or_edit(
                    self._accumulated, finalize=True, is_turn_final=False))
        elif self._message_id is None:
            # Draft path keeps _message_id=None; seal in place (else the stream stays
            # visibly live and the adapter keeps armed interception state).
            await self._abandon_native_stream()
        if best_effort_ok and not self._final_response_sent:
            self._mark_final_delivered(record=self._accumulated)

    def _wake_flush_waiters(self) -> None:
        """Wake still-queued _FLUSH waiters so a consumer dying mid-flush
        doesn't stall flush_pending_sync() for its full timeout."""
        with contextlib.suppress(Exception):
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 2 and item[0] is _FLUSH:
                    self._signal_flush(item[1])

    @staticmethod
    # Strip MEDIA:<path> tags before display. Uses the shared anchored MEDIA_TAG_CLEANUP_RE from
    # gateway/platforms/base.py — only tags whose path ends in a deliverable extension are removed, so an
    # unknown-extension path stays visible instead of being silently dropped (issue #34517). Streaming and
    # non-streaming paths share the same regex, so a tag is treated identically whichever path delivered the
    # text.
    def _clean_for_display(text: str) -> str:
        """Hide MEDIA:<path> / [[audio_as_voice]] directives; media is delivered post-stream."""
        return _BasePlatformAdapter.strip_media_directives_for_display(text)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'MEDIA_TAG_CLEANUP_RE': ('gateway.platforms.base', 'MEDIA_TAG_CLEANUP_RE'),
    'escape_code_fences_for_display': ('gateway.stream_consumer_fences', 'escape_code_fences_for_display'),
}


        Thin delegate to the shared fence-chunker core in
        :mod:`gateway.platforms.helpers` (``balance_fences_across_chunks``);
        kept as a method for the existing call sites and tests.
        """
        from gateway.platforms.helpers import balance_fences_across_chunks

        return balance_fences_across_chunks(chunks)

    @staticmethod
    def _split_text_chunks(
        text: str,
        limit: int,
        len_fn: "Callable[[str], int]" = len,
    ) -> list[str]:
        """Split text into reasonably sized chunks for fallback sends.

        Chunks are fence-balanced: a split inside a ``` code block closes the
        fence on the head chunk and reopens it on the tail, so no chunk leaves
        the rest of a message rendering as one giant code block.

        Delegates to the shared fence-chunker core
        (:func:`gateway.platforms.helpers.split_text_fence_aware`) with this
        consumer's knobs: newline-preferred splitting + fence balancing.
        """
        from gateway.platforms.helpers import split_text_fence_aware

        return split_text_fence_aware(
            text,
            limit,
            len_fn,
            prefer_paragraphs=False,
            balance_fences=True,
        )

    def _truncate_for_stream(
        self,
        text: str,
        limit: int,
        len_fn: "Callable[[str], int]",
    ) -> list[str]:
        """Use the adapter's canonical splitter for streaming overflow.

        Platform adapters may add word-boundary, code-fence, table, or
        platform-specific formatting rules.  The consumer must not replace
        those rules with newline-only slicing.  Non-base test doubles and
        legacy adapters retain the historical two-argument call shape.
        """
        truncate = getattr(self.adapter, "truncate_message", None)
        if not callable(truncate):
            return self._split_text_chunks(text, limit, len_fn)

        if isinstance(self.adapter, _BasePlatformAdapter):
            chunks = truncate(text, limit, len_fn=len_fn)
        else:
            chunks = truncate(text, limit)
        if not isinstance(chunks, (list, tuple)) or not all(
            isinstance(chunk, str) for chunk in chunks
        ):
            return self._split_text_chunks(text, limit, len_fn)
        return list(chunks)

    async def _send_fallback_final(self, text: str) -> None:
        """Send the final continuation after streaming edits stop working.

        Retries each chunk once on flood-control failures with a short delay.
        """
        final_text = self._clean_for_display(text)
        # Ensure balanced code fences before computing continuation,
        # so the closing fence reaches the user even when the fallback
        # only delivers the tail after mid-stream edits failed.
        final_text = ensure_closed_code_fences(final_text)
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
                # The visible partial equals the complete final text (#71643).
                # Route through the recorder so a split turn records the full
                # ledger rather than this tail-only payload — an unrecorded or
                # tail-only split now reads as a mismatch and would re-send
                # text the user already has (#78541).
                self._record_turn_final_payload(final_text)
                return

        raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        _len_fn: "Callable[[str], int]" = (
            self.adapter.message_len_fn
            if isinstance(self.adapter, _BasePlatformAdapter)
            else len
        )
        # Per-chat resolution (relay adapter fronting N platforms): the cap and
        # length unit follow the chat's underlying platform, not the adapter
        # scalar. Native adapters return their scalar/property unchanged.
        if isinstance(self.adapter, _BasePlatformAdapter):
            try:
                raw_limit = self.adapter.max_message_length_for_chat(self.chat_id)
                _len_fn = self.adapter.message_len_fn_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("per-chat limit resolution failed: %s", e)
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
                retry_delay = self._fallback_flood_retry_delay(result)
                if attempt == 0 and retry_delay is not None:
                    logger.debug(
                        "Flood control on fallback send, retrying in %.1fs",
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    break  # non-flood error, long flood wait, or second failure

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
        # The fallback delivered the complete ``final_text`` (as one message
        # or prefix + continuation chunks that union to it), so record it as
        # the turn-final payload for the gateway's reconciliation (#71643).
        # On a split turn ``final_text`` is only the tail — the recorder
        # substitutes the unsplit ledger so the sealed heads count as
        # delivered too (#78541).
        self._record_turn_final_payload(final_text)
        self._last_sent_text = chunks[-1]
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False

    async def _send_empty_fallback_final(self, final_text: str) -> str:
        """Commit a completed answer after Telegram finalization fails.

        Returns ``delivered`` on confirmed success, ``failed`` when the
        gateway can safely retry, and ``ambiguous`` when a timeout may have
        reached the platform already.
        """
        # Tool/segment boundaries intentionally preserve the run-wide preview
        # IDs for normal fresh-final cleanup.  This recovery replaces only the
        # active final segment, so never delete an earlier finalized preamble.
        stale_ids = set(self._segment_preview_message_ids)
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
            retry_delay = self._fallback_flood_retry_delay(result)
            if attempt == 0 and retry_delay is not None:
                logger.debug(
                    "Flood control on empty fallback final send; retrying in %.1fs",
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
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

        self._segment_preview_message_ids = set()
        self._message_id = new_message_id or "__no_edit__"
        self._already_sent = True
        self._final_response_sent = True
        self._final_content_delivered = True
        # Fresh commit of the complete answer after a failed finalize (#71643).
        #
        # Record ``final_text`` VERBATIM -- do not route through
        # _record_turn_final_payload here.  This recovery deleted the sealed
        # segment previews just above, so the only thing left on screen is the
        # message we just sent.  On a split turn the ledger holds the sealed
        # heads too, and recording it would claim delivery for text this path
        # just removed -- the gateway would then suppress and the user would be
        # left with a fraction of the answer (the #78541 swallow, reintroduced).
        self._delivered_final_text = ensure_closed_code_fences(
            self._clean_for_display(final_text or "")
        ).strip()
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

    def _fallback_flood_retry_delay(self, result: Any) -> float | None:
        """Return a bounded retry delay for a fallback send, if safe to retry."""
        if not self._is_flood_error(result):
            return None
        try:
            delay = float(getattr(result, "retry_after", None) or 3.0)
        except (TypeError, ValueError):
            delay = 3.0
        if delay > self._max_fallback_flood_retry_seconds:
            logger.debug(
                "Flood control requests %.1fs; leaving final delivery to the gateway",
                delay,
            )
            return None
        return max(0.0, delay)

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
            try:
                # Per-chat capability (review r2, finding 2): multi-platform
                # relay adapters resolve draft support through the CHAT's
                # negotiated descriptor, not the primary identity's. Older
                # adapters without the kwarg keep the legacy probe.
                supported = self.adapter.supports_draft_streaming(
                    chat_type=self.cfg.chat_type or None,
                    metadata=self.metadata,
                    chat_id=self.chat_id,
                )
            except TypeError:
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
        # Carry the per-turn identity on EVERY frame (review B2): the
        # turn-final send goes out via _metadata_for_send, which stamps
        # reply_to_message_id — the relay adapter keys draft/seal state on
        # that identity, so frames must carry the same one or the final
        # cannot find the open stream (flat DMs have no thread metadata
        # at all and would otherwise key on the bare chat).
        _md = dict(self.metadata) if self.metadata else {}
        if self._initial_reply_to_id:
            _md.setdefault("reply_to_message_id", self._initial_reply_to_id)
        try:
            result = await self.adapter.send_draft(
                chat_id=self.chat_id,
                draft_id=self._draft_id,
                content=text,
                metadata=_md or None,
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

    async def _abandon_native_stream(self) -> None:
        """Close an orphaned native draft stream on turn death (review B8).

        Stale-generation exits and cancellations previously returned with
        the stream still open: the platform message kept its live
        streaming indicator forever, and the adapter's armed interception
        state survived into the next turn. Seal in place with the last
        delivered frame (adds nothing new on screen), via the adapter's
        best-effort ``abandon_open_draft``. Never sets delivery flags —
        an abandoned turn's text was partial, and the gateway's normal
        paths still own whatever happens next.
        """
        if not self._use_draft_streaming:
            return
        abandon = getattr(type(self.adapter), "abandon_open_draft", None)
        if abandon is None:
            return
        try:
            _md = dict(self.metadata) if self.metadata else {}
            if self._initial_reply_to_id:
                _md.setdefault("reply_to_message_id", self._initial_reply_to_id)
            await self.adapter.abandon_open_draft(
                self.chat_id,
                self._last_sent_text or self._clean_for_display(self._accumulated),
                metadata=_md or None,
            )
        except Exception as e:
            logger.debug("abandon_open_draft failed (best-effort): %s", e)

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
            # Interim declaration: this tail is pre-boundary text, not the
            # turn-final — never let it seal a native stream (see
            # _send_commentary).
            _md = dict(self.metadata) if self.metadata else {}
            _md["_interim_send"] = True
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=tail,
                metadata=_md,
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
            # Declare interim intent: this send is NOT the turn-final. A
            # stream-is-the-message adapter (relay Slack native streaming)
            # must not let its seal-interception convert this into
            # draft(final=true) — that would seal the live stream with
            # interim text and orphan the true final into a plain-send
            # duplicate (live finding, 2026-08-16 canary).
            _md = dict(self.metadata) if self.metadata else {}
            _md["_interim_send"] = True
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                metadata=_md,
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

        Resolved PER-CHAT via ``max_message_length_for_chat`` — a relay adapter
        fronting N platforms has a different cap per chat (Discord 2000 vs
        Telegram 4096 vs Slack 39000); native adapters return their scalar
        ``MAX_MESSAGE_LENGTH`` unchanged. Adapters with a richer send/draft
        path (e.g. Telegram rich messages) can raise this above the base via
        ``streaming_overflow_limit`` so a reply that fits one rich message isn't
        fragmented at the legacy edit limit.  Falls back to
        ``MAX_MESSAGE_LENGTH`` (4096 default) for everyone else.
        """
        base = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        # isinstance gate: MagicMock adapters return mock objects (truthy, not
        # ints) for arbitrary attribute access — keep them on the base limit.
        if isinstance(self.adapter, _BasePlatformAdapter):
            try:
                base = self.adapter.max_message_length_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("max_message_length_for_chat failed: %s", e)
            try:
                cap = self.adapter.streaming_overflow_limit()
            except Exception as e:
                logger.debug("streaming_overflow_limit check failed: %s", e)
                cap = None
            if isinstance(cap, int) and cap > base:
                return cap
        return base

    def _track_preview_id(self, message_id: Optional[str]) -> None:
        """Record a real preview message id for finalization cleanup."""
        if message_id and message_id != "__no_edit__":
            message_id = str(message_id)
            self._preview_message_ids.add(message_id)
            self._segment_preview_message_ids.add(message_id)

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
        #
        # That replacement is only sound while ``text`` holds the whole answer.
        # On a multi-message split the head chunks were sealed and dropped out
        # of ``_accumulated``, so ``text`` is just the tail — deleting the
        # sealed heads would erase text the user already received and leave the
        # complete reply nowhere on screen (#78541).  Keep the sealed messages
        # and take the normal edit path instead.
        if self._turn_split_delivery:
            return False
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
        self._stream_ledger = ""
        self._last_sent_text = ""
        self._already_sent = False
        self._final_response_sent = False
        self._final_content_delivered = False
        self._delivered_final_text = None
        self._turn_split_delivery = False
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
        # Preserve the pre-fence-closed form for stream-is-the-message draft
        # frames: appending a closing ``` to a mid-code-block frame makes
        # frame N not a prefix of frame N+1, so the connector's append-only
        # delta computation falls back to a whole-snapshot re-append (the
        # stacked-copies class). Native streams render unclosed fences
        # progressively; the finalize path below still fence-closes the
        # real final message.
        _pre_fence_text = text
        # Ensure code fences are balanced before send/edit.  Model output
        # truncated mid-code-block (e.g. finish_reason="length") leaves an
        # orphaned ``` which, on Discord/Slack/Matrix, causes the entire
        # remaining output to render as a single code block.  This covers
        # the streaming edit path (G2) and first-send path alike.
        text = ensure_closed_code_fences(text)
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
        # Stream-is-the-message exception (finding #5, live canary): for
        # adapters like relay Slack native streaming, a segment-break
        # finalize must NOT become a real send — the adapter's seal
        # interception would convert it to draft(final=true), sealing the
        # stream at EVERY tool boundary (one frozen cumulative message per
        # segment; only the turn-final seal belongs). Those adapters keep
        # ONE stream per turn: mid-turn boundaries just emit another
        # cumulative frame; only got_done (is_turn_final) seals.
        _stream_is_msg = self._stream_is_message()
        if (
            self._use_draft_streaming
            and self._message_id is None
            and (not finalize or (_stream_is_msg and not is_turn_final))
        ):
            # Stream-is-the-message frames must stay prefix-stable: use the
            # pre-fence-closed text (see _pre_fence_text above). The turn
            # final still goes through the fence-closed path below.
            _frame_text = _pre_fence_text if _stream_is_msg else text
            # Finding #6 (live canary, the duplicate-content root cause):
            # strip the gateway's text cursor from draft frames. Native
            # streams render their own typing indicator, and a cursor-
            # suffixed frame breaks the connector's prefix-delta check on
            # EVERY tick ("...text▉" is never a prefix of "...text more▉"),
            # triggering its whole-text fallback append — the user saw each
            # cumulative snapshot stacked inside one message, ▉ included.
            if self.cfg.cursor and _frame_text.endswith(self.cfg.cursor):
                _frame_text = _frame_text[: -len(self.cfg.cursor)]
            # No-op skip: identical to the last frame we sent.
            if _frame_text == self._last_sent_text:
                return True
            ok = await self._send_draft_frame(_frame_text)
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
                            # Adapter adopted continuation messages — this
                            # turn is a multi-message delivery (#71643).
                            self._turn_split_delivery = True
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
                            # ``text`` is already cleaned/fence-closed here and
                            # equals the visible prefix — the on-screen content
                            # IS this finalize payload (#71643).  Record it on
                            # split turns too: post-#78541 an unrecorded split
                            # reads as a mismatch and would re-send this
                            # already-visible answer, reintroducing the
                            # duplicate #45517 fixed (#36965 / #25349).
                            self._record_turn_final_payload(text)
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
                        expect_edits=not finalize,
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
