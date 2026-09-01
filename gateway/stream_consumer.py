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
import concurrent.futures
import inspect
import logging
import queue
import secrets
import threading
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
_NEW_SEGMENT = object()
_COMMENTARY = object()
# Sentinel for tool-progress lines injected into the native stream bubble.
# Enqueued as ``(_TOOL_PROGRESS, line_text)`` by ``on_tool_progress()``.
_TOOL_PROGRESS = object()
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

# Sentinel to signal an interaction boundary (approval prompt OR clarify
# decision prompt) — finalize the current stream, disable native streaming,
# and let post-interaction output go via send().
_APPROVAL_BOUNDARY = object()

# Sentinel to request an EAGER native re-seed after a clarify-reopen boundary.
# Posted the moment the user answers a clarify (before the LLM produces any
# post-answer delta), so the WeCom typing bubble reappears immediately instead
# of waiting for the first token.  On WeCom, typing is driven by the stream
# seed frame (send_typing is a no-op), and the reopen path otherwise re-seeds
# lazily on the first delta — measured 48s of dead air in one turn.  Handled
# serially in run(); see request_reopen_seed() and the run-loop handler.
_REOPEN_SEED = object()

# Default finalize text shown at an interaction boundary when no content has
# accumulated yet.  Callers may override per-boundary (e.g. clarify passes its
# own) via close_for_approval_prompt(placeholder=...).
_DEFAULT_BOUNDARY_PLACEHOLDER = "⏸ 等待审批中..."


def escape_code_fences_for_display(text: str) -> str:
    """Escape triple-backtick markers so text can be safely wrapped
    inside an outer ``` code block without breaking the fence.

    When reasoning content contains ``` (e.g. the model quotes code
    in its thinking), wrapping it in an outer ``` for display causes
    the inner fence to break the outer block.  Solution: replace each
    `` ``` `` with `` \\`\\`\\` `` before wrapping.

    Returns:
        The input text with each `` ``` `` replaced by `` \\`\\`\\` ``,
        or the input unchanged if no triple-backticks are present.
    """
    if not isinstance(text, str) or "```" not in text:
        return text
    return text.replace("```", "\\`\\`\\`")


def ensure_closed_code_fences(text: str) -> str:
    """Append a closing `` ``` `` fence and/or `` ` `` if the text has
    orphaned code-block or inline-code markers.

    When model output is truncated mid-code-block (e.g. by token limits
    or a finish_reason="length"), the resulting message has an unclosed
    code fence.  On Discord, Slack, and other platforms this causes
    everything after the orphaned fence to render as a single code block.
    The same problem applies to inline-code spans closed by a single
    backtick: an orphaned `` ` `` makes the remainder of the message
    render as inline code.

    Triple-backtick: count `` ``` `` occurrences.  If odd, append a
    closing fence on its own line.  This is safe because nested
    triple-backtick fences (e.g. a literal `` ``` `` inside a code block)
    are exceedingly rare in model output and, when they do appear, the
    extra closing fence just creates a brief empty code block at the end
    of the message — far less harmful than the entire message being one
    giant code block.

    Single backtick: after balancing triple-backtick fences, strip all
    complete `` ```…``` `` regions and count remaining standalone `` ` ``.
    If odd, append a closing inline-code backtick.  Same trade-off: a
    stray closing backtick may produce a brief empty inline-code span,
    which is far less harmful than the rest of the message being rendered
    as inline code.

    Returns:
        The input text with closing markers appended if needed, or the
        input text unchanged.
    """
    if not isinstance(text, str) or not text:
        return text

    # Step 1: fix triple-backtick code-block fences (existing logic)
    if text.count("```") % 2 == 1:
        text = text.rstrip("\n") + "\n```"

    # Step 2: fix single-backtick inline-code spans
    # Remove complete ```…``` regions so their internal backticks don't
    # pollute the standalone count.  Also remove any trailing unclosed
    # ``` that leaks through (defence in depth).
    import re
    without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    without_fences = re.sub(r"```[^`]*$", "", without_fences)

    if without_fences.count("`") % 2 == 1:
        text = text + "`"

    return text


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

        # Per-turn identifier: uniquely identifies this consumer's stream turn.
        # Passed to adapter.send_stream_frame() to prevent concurrent consumers
        # from interfering with each other (e.g., /background, parallel subagents).
        # Mirrors official wecom-openclaw-plugin's per-message streamId generation.
        import uuid
        self._turn_id = str(uuid.uuid4())

        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        # Full segment text mirror of ``_accumulated`` that is NOT truncated
        # when overflow splits seal head chunks.  Used to record a reconciliable
        # turn-final payload for multi-message deliveries (#78541).
        self._stream_ledger = ""
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
        # IDs from only the active text segment.  A tool boundary preserves
        # the run-wide set for fresh-final bookkeeping, but a failure recovery
        # must never delete an earlier finalized preamble/commentary message.
        self._segment_preview_message_ids: "set[str]" = set()
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
        # Keep fallback recovery responsive. Telegram's adapter already bounds
        # edit retries at five seconds; a final-delivery fallback must not hold
        # the stream task through a longer flood cooldown before retrying.
        self._max_fallback_flood_retry_seconds = 5.0
        self._flood_strikes = 0         # Consecutive flood-control edit failures
        self._current_edit_interval = self.cfg.edit_interval  # Adaptive backoff
        self._final_response_sent = False
        # Set when the final response content was sent to the user via
        # streaming, even if the final edit (cursor removal etc.)
        # subsequently failed.
        self._final_content_delivered = False
        # Exact cleaned payload of the turn-final delivery that set the flags
        # above.  The gateway compares this against the completed
        # ``final_response`` before trusting the flags: a *successful* finalize
        # edit that carried only a stale preview snapshot must not suppress the
        # complete send (#71643).  ``None`` means "no record" — legacy trust,
        # so paths that predate the record keep their behavior.
        self._delivered_final_text: Optional[str] = None
        # True when the current turn's answer was delivered across multiple
        # sealed messages (overflow split / adapter continuation adoption).
        # When a payload was recorded (via ``_stream_ledger`` /
        # ``_record_turn_final_payload``), ``delivered_final_matches`` can still
        # reconcile.  Payload-less split delivery must NOT inherit legacy trust
        # (#78541) — that combination was swallowing complete Telegram group
        # replies after an early/partial multi-message delivery.
        self._turn_split_delivery = False
        # True when a full-final send timed out in a way that MAY have reached
        # the platform (``_send_empty_fallback_final`` → "ambiguous").  The
        # only case where a payload-less delivery flag keeps legacy trust in
        # ``delivered_final_matches`` (#95382 tightening) — re-sending there
        # risks a duplicate rather than recovering a loss.
        self._delivery_ambiguous = False
        self._delivered_commentary_texts: list[str] = []
        # Retains the finalized visible text of each streaming segment so
        # ``has_delivered_text`` can still match after ``_reset_segment_state``
        # clears ``_last_sent_text``. Without this, a segment break (triggered
        # by ``on_segment_break`` or ``on_commentary``) erases the only record
        # of what was delivered, and the gateway's final-send suppression
        # can't recognize an already-delivered response. (#65919 review)
        self._delivered_segment_texts: list[str] = []
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
        # Native streaming transport (e.g. WeCom msgtype: "stream"). Unlike
        # drafts, native streaming is the *only* delivery channel for the
        # turn — first frame, mid-stream updates, and the final answer all
        # flow through ``adapter.send_stream_frame()`` and the adapter
        # manages the stream lifecycle (init → cumulative updates →
        # finish=true). Resolved at the start of run() and disabled on
        # any failure so the consumer falls back to edit/send.
        self._use_native_streaming = False
        # Tracks whether the native stream bubble has been opened (seed frame sent).
        # Used in fallback logic to decide if we need to finalize the stream before
        # falling back to send(). Set to True after seed frame succeeds, even though
        # seed has zero visible content.
        self._native_stream_opened = False
        # Number of visible characters last successfully pushed to the
        # native stream. Used for "send only when enough new content has
        # accumulated" throttling so we don't spam frames at WeCom's
        # 30 frames/min rate ceiling.
        self._native_last_pushed_len = 0
        # Finalize text used at an interaction boundary (approval/clarify) when
        # no content has accumulated yet.  Set by close_for_approval_prompt();
        # defaults to the approval wording for backward compatibility.
        self._boundary_placeholder = _DEFAULT_BOUNDARY_PLACEHOLDER
        # Human-readable label for the current interaction boundary, used only
        # for log prefixes so a clarify boundary doesn't log as "Approval".
        # Set by close_for_approval_prompt(); race-free because boundaries are
        # processed serially.
        self._boundary_reason = "Approval"
        # When True, the interaction boundary finalizes the current stream but
        # KEEPS native streaming enabled so post-prompt output re-opens a fresh
        # native stream (via the lazy re-seed in _send_or_edit) instead of
        # degrading to a one-shot send().  Clarify sets this (short waits, low
        # stream-staleness risk); approval leaves it False (long, unbounded
        # waits — the stream may go stale, so send() is safer).  Set by
        # close_for_approval_prompt(); race-free (boundaries are serial).
        self._boundary_reopen = False
        # Marks that a boundary asked to reopen the native stream but no
        # post-prompt content has re-seeded it yet.  Guards got_done from
        # re-seeding a fresh stream just to emit a lone "✅" placeholder when
        # the agent produced nothing after the prompt.
        self._awaiting_reopen_after_boundary = False
        # Marks that an EAGER re-seed (via _REOPEN_SEED) already opened a fresh
        # native stream after a clarify answer, BEFORE any post-answer content.
        # Unlike the lazy path, the typing bubble is already on screen, so
        # got_done must actively finalize it (not silently skip) when the agent
        # produces no content — otherwise a blank typing bubble hangs forever.
        self._reopen_seeded_eagerly = False

        # Tool-progress overlay state (native streaming only).
        # Lines are injected via on_tool_progress() and displayed as a
        # temporary overlay in the stream bubble until real text arrives.
        self._tool_progress_lines: list[str] = []
        self._tool_progress_active: bool = False


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

    @property
    def accepts_tool_progress(self) -> bool:
        """Whether this consumer can absorb tool progress into its stream.

        True only when native streaming is resolved and active. Callers use
        this to decide the progress routing path (in-stream vs progress_queue).
        """
        return self._use_native_streaming

    def on_tool_progress(self, line: str) -> None:
        """Inject a tool-progress status line into the native stream bubble.

        Thread-safe (called from agent worker thread via queue.Queue). Only
        meaningful when native streaming is active — callers should gate on
        ``accepts_tool_progress``.

        The line is displayed as an overlay until the next text delta arrives,
        at which point real content overwrites the tool-progress lines.
        """
        if line:
            self._queue.put((_TOOL_PROGRESS, line))

    def _compose_frame_content(self) -> str:
        """Compose the current frame content for native streaming.

        Strategy B: when both accumulated text and tool-progress lines exist,
        append tool lines below the text separated by a horizontal rule.
        On finalize, only accumulated text is sent (no tool lines).
        """
        if self._accumulated and self._tool_progress_lines:
            # Text + active tool status at the bottom
            return self._accumulated + "\n\n---\n" + "\n".join(self._tool_progress_lines)
        elif self._accumulated:
            return self._accumulated
        elif self._tool_progress_lines:
            return "\n".join(self._tool_progress_lines)
        return ""

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
        if self._initial_reply_to_id:
            meta["reply_to_message_id"] = self._initial_reply_to_id
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

    def _append_accumulated(self, text: str) -> None:
        """Append to the live buffer and the split-stable stream ledger."""
        if not text:
            return
        # New text delta arriving: clear tool-progress overlay so the next
        # frame shows real content (Strategy B: text overwrites tool lines).
        if self._tool_progress_lines:
            self._tool_progress_lines.clear()
            self._tool_progress_active = False
        self._accumulated += text
        self._stream_ledger += text

    def _mark_skip_redundant_finalize(self) -> None:
        """Mark the turn final as delivered by a prior mid-stream edit.

        Used by the run loop when the final accumulated content was already
        delivered by the last visible edit and the explicit finalize edit is
        skipped. Records what was actually ACKED on the wire, not what was
        accumulated: a throttled edit-transport stream can reach this state
        with the last acked edit still holding an earlier preview snapshot
        (cursor-suffixed). Recording ``_accumulated`` would let a frozen
        preview reconcile as the delivered final and suppress the corrective
        send, leaving the user a cut-off message. The multi-message split
        path keeps its ledger substitution inside
        ``_record_turn_final_payload``.
        """
        self._final_response_sent = True
        self._final_content_delivered = True
        acked = self._last_sent_text or self._accumulated
        if self.cfg.cursor and acked.endswith(self.cfg.cursor):
            acked = acked[: -len(self.cfg.cursor)]
        self._record_turn_final_payload(acked)

    def _record_turn_final_payload(self, text: str) -> None:
        """Record what the user has actually seen as this turn's final answer.

        Normalized the same way ``_send_or_edit`` normalizes outgoing text
        (media-directive strip + fence closing) so the gateway can compare it
        against the completed ``final_response`` (#71643).

        ``text`` is what the *calling* path just delivered. On a multi-message
        split that is only the trailing chunk — the overflow paths truncate
        ``_accumulated`` once head chunks are sealed — so ``_stream_ledger``
        (the un-truncated segment text) is preferred there and ``text`` is
        ignored. Without that substitution a split turn records a tail-only
        payload, which the gateway reads as a mismatch and re-sends on top of
        an answer the user already received (#78541).
        """
        source = text or ""
        if self._turn_split_delivery and self._stream_ledger:
            source = self._stream_ledger
        self._delivered_final_text = ensure_closed_code_fences(
            self._clean_for_display(source)
        ).strip()

    def delivered_final_matches(self, final_text: str) -> Optional[bool]:
        """Reconcile the recorded turn-final payload against ``final_text``.

        Returns a tri-state verdict for the gateway's suppression decision
        (#71643 — a *successful* finalize edit can still carry only a stale
        preview snapshot, so call success alone must not confirm delivery):

        - ``True``  — the recorded turn-final payload (or a previously
          delivered segment/commentary) matches ``final_text``; suppressing
          the normal final send is safe.
        - ``False`` — a turn-final delivery was recorded but its payload
          demonstrably differs from ``final_text``, OR this was a
          payload-less multi-message split delivery (#78541) whose flag
          alone must not suppress the normal final send.
        - ``None``  — no payload comparison is possible on a non-split
          legacy/uncertain path that recorded nothing. The caller keeps
          the pre-existing flag-trusting behavior so ambiguous-timeout
          dedup is not regressed.
        """
        target = ensure_closed_code_fences(
            self._clean_for_display(final_text or "")
        ).strip()
        if not target:
            return None
        if self._delivered_final_text is None:
            if self._turn_split_delivery:
                # #78541: refuse legacy trust for payload-less split delivery.
                return False
            # #95382 / #98552 class fix: a delivery flag with NO recorded
            # payload must still be judged against the FINAL content, not
            # trusted blindly. Every internal flag-setting site records a
            # payload; a record-less consumer whose visible/streamed text
            # does not contain the completed response has demonstrably NOT
            # delivered it (first-edit prefix, mid-stream truncation) — the
            # flag alone must not suppress the corrective send.
            if self.has_delivered_text(final_text):
                return True
            # The one legitimately ambiguous case keeps legacy trust: a
            # timed-out full-final send may have reached the platform
            # (``_send_empty_fallback_final`` → "ambiguous"), so re-sending
            # risks a duplicate. That site marks itself explicitly.
            if self._delivery_ambiguous:
                return None
            return False
        if self._delivered_final_text.strip() == target:
            return True
        # A segment break / commentary may have delivered the final text
        # earlier in the turn under a different record.
        if self.has_delivered_text(final_text):
            return True
        return False

    def has_delivered_text(self, text: str) -> bool:
        """Return True if *text* was already delivered as visible chat content."""
        target = self._clean_for_display(text or "").strip()
        if not target:
            return False
        visible_prefix = self._visible_prefix().strip()
        if visible_prefix == target:
            return True
        return any(
            sent.strip() == target
            for sent in (*self._delivered_commentary_texts, *self._delivered_segment_texts)
        )

    def on_segment_break(self) -> None:
        """Finalize the current stream segment and start a fresh message."""
        self._queue.put(_NEW_SEGMENT)

    def close_for_approval_prompt(
        self,
        placeholder: str | None = None,
        reason: str = "Approval",
        reopen: bool = False,
    ) -> asyncio.Future:
        """Signal an interaction boundary — finalize stream, then either disable
        native (approval) or keep it for a fresh re-opened stream (clarify).

        Used for any mid-stream interaction that must not keep updating the
        current native-stream bubble: a dangerous-command approval prompt or a
        clarify decision prompt.  Queues a boundary signal that the consumer
        processes serially: finalize the current stream with accumulated text
        (creating a stable message for pre-prompt content), then handle
        post-prompt output per ``reopen``.

        ``placeholder`` is the finalize text used only when there is no
        accumulated content yet (the prompt fired as the agent's first action).
        Defaults to the approval placeholder; clarify passes its own so the
        finalized bubble doesn't read "waiting for approval" for a question.

        ``reason`` is a human-readable label ("Approval"/"Clarify") used only
        for the boundary handler's log prefixes so a clarify boundary doesn't
        surface as an "Approval boundary" failure during troubleshooting.

        ``reopen`` controls post-prompt delivery.  False (approval): disable
        native streaming and buffer post-prompt output into a single reliable
        send() — approval waits are long and unbounded, so the stream may go
        stale.  True (clarify): keep native streaming enabled so post-prompt
        output re-opens a fresh native stream via the existing lazy re-seed,
        restoring the typing-bubble experience; if the re-seed later fails the
        consumer degrades to send() automatically.

        Returns a (Future, cancelled_flag) tuple. The Future resolves True
        when the boundary has been processed. cancelled_flag is included
        for backward compatibility with callers that set it on timeout;
        the boundary handler no longer reads it (finalize always runs).

        For platforms without native streaming this is a no-op (returns
        an immediately-resolved Future).

        Called from sync context (agent/approval thread). The boundary
        is processed by the consumer's async run() task, ensuring no
        race conditions with pending deltas or other queue items.
        """
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if not self._use_native_streaming:
            # No native stream to close — return resolved future
            f = asyncio.Future() if loop else concurrent.futures.Future()
            f.set_result(True)
            return f

        # Stash the empty-content placeholder, log label, and reopen mode for
        # the serial boundary handler.  Boundaries are processed one at a time,
        # so instance attributes are race-free and keep the queue signal shape
        # unchanged.
        self._boundary_placeholder = placeholder or _DEFAULT_BOUNDARY_PLACEHOLDER
        self._boundary_reason = reason or "Approval"
        self._boundary_reopen = bool(reopen)

        # Create a future that run() will resolve after processing.
        # cancelled_flag is retained for backward compatibility with callers
        # (run.py sets it on timeout) but the handler always finalizes regardless.
        if loop:
            boundary_future = loop.create_future()
        else:
            boundary_future = concurrent.futures.Future()

        cancelled_flag = {"cancelled": False}
        self._queue.put((_APPROVAL_BOUNDARY, boundary_future, cancelled_flag))
        return boundary_future, cancelled_flag

    def on_commentary(self, text: str) -> None:
        """Queue a completed interim assistant commentary message."""
        if text:
            self._queue.put((_COMMENTARY, text))

    def flush_pending_sync(self, timeout: float = 5.0) -> bool:
        """Block the calling (agent worker) thread until everything queued
        before this point has been finalized and delivered to the platform.

        Enqueues a ``(_FLUSH, Event)`` barrier behind any pending deltas /
        commentary / segment breaks.  The async ``run()`` task processes those
        first (FIFO), then handles the barrier — finalizing the current segment
        and setting the event.  Returns True if the flush completed within
        ``timeout``, False on timeout (so the caller continues rather than
        hanging if the consumer task is not running / already finished).

        This is the ordering barrier used before sending a blocking interactive
        prompt (clarify poll): without it, the poll — sent on a separate,
        agent-thread-blocking path — races ahead of buffered prose that is still
        sitting in this queue, so the question lands ABOVE its own explanation.
        """
        evt = threading.Event()
        try:
            self._queue.put((_FLUSH, evt))
        except Exception:
            return False
        return evt.wait(timeout=max(0.0, float(timeout)))

    def request_reopen_seed(self) -> None:
        """Request an EAGER native re-seed after a clarify-reopen boundary.

        Called (thread-safe, like on_commentary / close_for_approval_prompt)
        the instant the user answers a clarify — BEFORE the LLM emits any
        post-answer delta. Posts _REOPEN_SEED so run() immediately sends an
        empty seed frame, which is what makes the WeCom typing bubble reappear
        without waiting for the first token (measured 48s of dead air otherwise).

        No-op unless we're in the reopen-pending state on a native stream: only
        after a clarify boundary (`_awaiting_reopen_after_boundary`) with native
        still enabled and no stream currently open. This keeps a stray call from
        opening a spurious bubble mid-stream or on the approval path.
        """
        if (
            self._use_native_streaming
            and self._awaiting_reopen_after_boundary
            and not self._native_stream_opened
        ):
            self._queue.put(_REOPEN_SEED)

    def _notify_new_message(self) -> None:
        """Fire the on_new_message callback, swallowing any errors."""
        cb = self._on_new_message
        if cb is None:
            return
        try:
            cb()
        except Exception:
            logger.debug("on_new_message callback error", exc_info=True)

    @staticmethod
    def _signal_flush(flush_event) -> None:
        """Wake a thread blocked in flush_pending_sync(), swallowing errors.

        Centralised so every loop exit path that consumed a ``_FLUSH`` barrier
        (the normal bottom-of-iteration path AND early ``continue`` paths such
        as the oversized-prose overflow split) reliably sets the event. Missing
        a set is not a deadlock — the caller uses a bounded timeout — but it
        would make the caller stall the full timeout before its blocking send.
        """
        if flush_event is None:
            return
        try:
            flush_event.set()
        except Exception:
            pass

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
        # Tool-progress overlay: clear on segment reset so a new segment
        # starts clean.
        self._tool_progress_lines = []
        self._tool_progress_active = False
        # #29346: a tool/segment boundary means what we delivered was an interim
        # preamble, not the final answer — clear the flags so a premature setter
        # can't fool the gateway. Safe: got_done returns before any reset, and
        # run.py reads these only after the consumer task exits.
        self._final_response_sent = False
        self._final_content_delivered = False
        self._delivered_final_text = None
        self._delivery_ambiguous = False
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

    async def _handle_approval_boundary(self, boundary_future, cancelled_flag=None) -> None:
        """Process an approval boundary: finalize stream, disable native for post-approval.

        This method is called serially from run() when _APPROVAL_BOUNDARY is dequeued.

        Strategy: finalize the current stream with accumulated text (creating a
        stable message for pre-approval content), then disable native streaming
        so post-approval output goes through the reliable send() path.

        Why not keep the stream open across approval:
        - WeCom stream finalize ack only confirms server receipt, not client render.
        - Approval waits introduce an idle gap where the stream may become stale
          on the client side (no server-side 846608, but client stops tracking it).
        - If content_delivered=True but the client didn't render, the normal
          final send is suppressed → user sees nothing.
        - Approval is a natural interaction boundary; "pre-approval preamble" +
          "post-approval result" as two messages is acceptable UX.

        Post-approval output uses regular send() which is unconditionally reliable.
        """
        # Log label ("Approval"/"Clarify") so a clarify boundary failure doesn't
        # surface as an "Approval boundary" error during troubleshooting.
        _reason = getattr(self, "_boundary_reason", "Approval") or "Approval"
        delivery_failed = False
        try:
            if self._native_stream_opened:
                # Finalize current stream with accumulated content.
                # This converts the typing bubble into a stable message.
                finalize_text = self._accumulated or self._boundary_placeholder
                finalize_ok = False
                try:
                    result = await self.adapter.send_stream_frame(
                        finalize_text,
                        finalize=True,
                        chat_id=self.chat_id,
                        reply_to=self._initial_reply_to_id,
                        turn_id=self._turn_id,
                    )
                    finalize_ok = bool(result)
                except Exception as e:
                    logger.warning("%s boundary: finalize failed: %s", _reason, e)

                if not finalize_ok:
                    # Stream finalize didn't land — the typing bubble may still
                    # be showing partial content. Fallback: deliver the pre-prompt
                    # text via reliable send() so the user at least sees it.
                    logger.warning(
                        "%s boundary: finalize not confirmed, "
                        "falling back to send() for pre-prompt text (chat=%s)",
                        _reason, self.chat_id,
                    )
                    fallback_ok = False
                    try:
                        send_result = await self.adapter.send(
                            self.chat_id, finalize_text,
                        )
                        fallback_ok = getattr(send_result, "success", False)
                    except Exception as send_err:
                        logger.warning(
                            "%s boundary: fallback send also failed: %s",
                            _reason, send_err,
                        )
                    if not fallback_ok:
                        # Both finalize and fallback failed — pre-prompt text
                        # may be lost. Mark boundary as failed so the caller knows.
                        logger.error(
                            "%s boundary: both finalize and fallback send failed "
                            "(chat=%s) — pre-prompt text may not have been delivered",
                            _reason, self.chat_id,
                        )
                        delivery_failed = True
                else:
                    logger.debug(
                        "%s boundary: finalized stream (chat=%s, turn=%s)",
                        _reason, self.chat_id, self._turn_id,
                    )

            if self._boundary_reopen:
                # Clarify boundary: KEEP native streaming enabled.  The current
                # stream was finalized above (pre-prompt content is now a stable
                # bubble); marking it closed makes the next post-prompt delta
                # re-open a fresh native stream via the lazy re-seed in
                # _send_or_edit, restoring the typing-bubble experience.  Do NOT
                # set buffer_only — post-prompt output should stream, not batch.
                # If the re-seed later fails, the consumer degrades to send()
                # on its own.  _awaiting_reopen_after_boundary guards got_done
                # from re-seeding a stream just to emit a lone "✅" when the
                # agent produced no post-prompt content.
                self._native_stream_opened = False
                self._native_last_pushed_len = 0
                self._awaiting_reopen_after_boundary = True
                self._reset_segment_state()
                # INFO (temporary latency probe): boundary finalize is done and
                # the old bubble is closed.  From here the consumer waits for
                # the LLM's first post-answer delta before re-seeding the C
                # bubble — so the gap between THIS line and the
                # "Re-opened native stream" INFO below is exactly the
                # "typing slow to reappear after clarify" delay.
                logger.info(
                    "[latency] Clarify boundary finalized, awaiting first "
                    "post-answer delta to re-seed (chat=%s, turn=%s)",
                    self.chat_id, self._turn_id,
                )
            else:
                # Approval boundary: disable native streaming for post-approval
                # output, which goes through regular send() (unconditionally
                # reliable, no client-side stream state dependency).  Set
                # buffer_only=True so the consumer accumulates all post-approval
                # text and delivers it in one shot on got_done, avoiding
                # mid-stream flushes that would create multiple messages on
                # non-editable platforms like WeCom.
                self._use_native_streaming = False
                self._native_stream_opened = False
                self._native_last_pushed_len = 0
                self.cfg.buffer_only = True

                # Reset segment state so post-approval output starts fresh via send().
                self._reset_segment_state()

            boundary_ok = not delivery_failed

        except Exception as e:
            logger.warning("%s boundary processing failed: %s", _reason, e)
            boundary_ok = False
        finally:
            # Resolve future so approval callback knows the result
            if boundary_future is not None:
                try:
                    if isinstance(boundary_future, asyncio.Future):
                        if not boundary_future.done():
                            boundary_future.set_result(boundary_ok)
                    elif isinstance(boundary_future, concurrent.futures.Future):
                        if not boundary_future.done():
                            boundary_future.set_result(boundary_ok)
                except Exception:
                    pass

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
                    self._append_accumulated(buf[:best_idx])
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
                        self._append_accumulated(buf[:-held_back])
                        self._think_buffer = buf[-held_back:]
                    else:
                        # No (partial) open tag — but the model may have
                        # emitted an orphan close tag like </think> on its
                        # own (e.g. when a thinking-mode toggle drops the
                        # matched open, or when upstream stripping is
                        # incomplete). Strip those before accumulating so
                        # they never reach the user.
                        self._append_accumulated(self._strip_orphan_close_tags(buf))
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
            self._append_accumulated(self._strip_orphan_close_tags(self._think_buffer))
            self._think_buffer = ""

    async def run(self) -> None:
        """Async task that drains the queue and edits the platform message."""
        # Platform message length limit — leave room for cursor + formatting.
        # Use the adapter's length function (e.g. utf16_len for Telegram) so
        # overflow detection matches what the platform actually enforces.
        # Both resolve PER-CHAT (max_message_length_for_chat): a relay adapter
        # fronting N platforms has different caps per chat (Discord 2000 vs
        # Telegram 4096); native adapters return their scalar unchanged.
        # Gate on isinstance(BasePlatformAdapter) so test MagicMocks (whose
        # auto-attributes return mock objects, not callables) fall back to len.
        _len_fn: "Callable[[str], int]" = (
            self.adapter.message_len_fn_for_chat(self.chat_id)
            if isinstance(self.adapter, _BasePlatformAdapter)
            else len
        )
        # Rich-capable adapters (Telegram rich messages) raise this above the
        # legacy per-message limit so a reply that fits one rich send/draft
        # isn't fragmented at 4096 while streaming.  See _raw_message_limit.
        _raw_limit = self._raw_message_limit()
        _safe_limit = max(500, _raw_limit - _len_fn(self.cfg.cursor) - 100)

        # Resolve transport once per run. Native streaming wins over draft
        # because the only adapters that declare it (WeCom) cannot edit
        # messages at all — there is no edit path to fall back to mid-turn.
        # When native is selected we send an empty seed frame immediately so
        # the user sees the platform's "typing" indicator before the LLM
        # produces any tokens; if that seed fails (no req_id, transport
        # error) we disable native and let the consumer take the regular
        # edit path (which will in turn refuse and fall back to fallback
        # send via the gateway, since SUPPORTS_MESSAGE_EDITING=False).
        self._use_native_streaming = self._resolve_native_streaming()
        if self._use_native_streaming:
            logger.debug(
                "Stream consumer using native-stream transport (chat=%s)",
                self.chat_id,
            )
            try:
                seed_ok = await self.adapter.send_stream_frame(
                    "",
                    chat_id=self.chat_id,
                    reply_to=self._initial_reply_to_id,
                    turn_id=self._turn_id,
                )
                if seed_ok:
                    # Mark stream as opened so fallback knows to finalize
                    self._native_stream_opened = True
            except Exception:
                logger.debug(
                    "Native streaming seed frame raised; disabling native",
                    exc_info=True,
                )
                seed_ok = False
            if not seed_ok:
                self._use_native_streaming = False

        # Resolve native draft streaming (Telegram drafts) only when native
        # streaming is not in use — they target the same first-frame slot.
        if self._use_native_streaming:
            self._use_draft_streaming = False
        else:
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
                    await self._abandon_native_stream()
                    return

                # Drain all available items from the queue
                got_done = False
                got_segment_break = False
                got_flush = False
                flush_event = None
                got_approval_boundary = False
                got_reopen_seed = False
                approval_boundary_future = None
                approval_boundary_cancelled = None
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
                        if item is _REOPEN_SEED:
                            got_reopen_seed = True
                            break
                        if isinstance(item, tuple) and len(item) == 3 and item[0] is _APPROVAL_BOUNDARY:
                            got_approval_boundary = True
                            approval_boundary_future = item[1]
                            approval_boundary_cancelled = item[2]
                            break
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
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _TOOL_PROGRESS:
                            # Tool-progress overlay: accumulate the status line.
                            # Only effective in native-streaming mode (callers
                            # gate before enqueue via accepts_tool_progress).
                            if self._use_native_streaming:
                                self._tool_progress_lines.append(item[1])
                                self._tool_progress_active = True
                            continue  # continue draining to batch simultaneous progress lines
                        self._filter_and_accumulate(item)
                    except queue.Empty:
                        break

                # Handle approval boundary: close current stream, reset for new turn.
                # Must happen before got_done/segment_break processing since it
                # produces its own finalize and resets state.
                if got_approval_boundary:
                    await self._handle_approval_boundary(
                        approval_boundary_future, approval_boundary_cancelled
                    )
                    continue

                # Handle eager re-seed: the user just answered a clarify prompt.
                # Open a fresh native stream NOW (empty seed frame) so the WeCom
                # typing bubble reappears immediately, without waiting for the
                # LLM's first post-answer delta.  Only meaningful in the
                # reopen-pending state with native still live and no stream open;
                # request_reopen_seed() already gates on that, and we re-check
                # here because state may have advanced between put and dequeue.
                #
                # TRADE-OFF: this moves the start of WeCom's ~6-minute stream
                # session limit (STREAM_EXPIRED_ERRCODE 846608, counted from the
                # FIRST frame, not renewed by intermediate frames) forward from
                # the first post-answer delta to the user-reply instant — the
                # effective window shrinks by however long the LLM takes to
                # produce its first token. A first token >5min is very rare, and
                # if the stream does expire send_stream_frame returns False and
                # the else branch below degrades to send(), so the answer still
                # lands (only the streaming animation is lost). Acceptable.
                if got_reopen_seed:
                    if (
                        self._use_native_streaming
                        and self._awaiting_reopen_after_boundary
                        and not self._native_stream_opened
                    ):
                        try:
                            seed_ok = await self.adapter.send_stream_frame(
                                "",
                                chat_id=self.chat_id,
                                reply_to=self._initial_reply_to_id,
                                turn_id=self._turn_id,
                            )
                        except Exception as e:
                            logger.debug(
                                "Eager reopen seed raised, disabling native: %s", e,
                            )
                            seed_ok = False
                        if seed_ok:
                            self._native_stream_opened = True
                            self._native_last_pushed_len = 0
                            self._awaiting_reopen_after_boundary = False
                            self._reopen_seeded_eagerly = True
                            logger.info(
                                "[latency] Eager re-seed after clarify answer "
                                "(typing bubble reopened immediately, turn=%s)",
                                self._turn_id,
                            )
                        else:
                            # Seed failed — degrade to a single buffered send()
                            # so the post-answer content still lands as one
                            # bubble (not per-tick fragments on a non-editable
                            # platform).  Mirrors the approval-boundary degrade.
                            self._use_native_streaming = False
                            self._native_stream_opened = False
                            self._native_last_pushed_len = 0
                            self.cfg.buffer_only = True
                    continue

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
                    if self._use_native_streaming:
                        # Fire-and-forget: native streaming has no platform
                        # edit-rate limit — push every accumulated delta
                        # immediately. The only gate is "have we accumulated
                        # anything new at all".
                        should_edit = should_edit or bool(self._accumulated) or self._tool_progress_active
                    else:
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
                    should_edit = False
                if should_edit and (self._accumulated or (self._use_native_streaming and self._tool_progress_active)):
                    # Split overflow: if accumulated text exceeds the platform
                    # limit, split into properly sized chunks.
                    # Native streaming bypasses this entirely — the adapter's
                    # send_stream_frame handles byte-level truncation against
                    # the stream protocol's larger limit (e.g. WeCom's 20480
                    # bytes vs. MAX_MESSAGE_LENGTH's 4000 codepoints).
                    if (
                        not self._use_native_streaming
                        and _len_fn(self._accumulated) > _safe_limit
                        and self._message_id is None
                    ):
                        # No existing message to edit (first message or after a
                        # segment break).  Seal only the overflowing head chunks
                        # as fixed messages, then keep the trailing chunk in
                        # _accumulated so the normal send/edit path below makes
                        # it the active preview.  That lets chunk 2, 3, ... keep
                        # updating in-place as later streamed deltas arrive
                        # instead of posting every split as an immutable message.
                        chunks = self._truncate_for_stream(
                            self._accumulated, _safe_limit, _len_fn,
                        )
                        if len(chunks) <= 1:
                            # A malformed/legacy adapter result must not leave
                            # this overflow branch with an unsplittable payload.
                            chunks = self._split_text_chunks(
                                self._accumulated, _safe_limit, _len_fn,
                            )
                        chunks_delivered = False
                        reply_to = self._initial_reply_to_id
                        all_heads_delivered = len(chunks) > 1
                        for chunk in chunks[:-1]:
                            new_id = await self._send_new_chunk(
                                chunk,
                                reply_to,
                                final=got_done,
                            )
                            if new_id is None or new_id == reply_to:
                                # Failed to deliver a sealed head; keep the
                                # full accumulated text intact so the gateway's
                                # fallback path can still deliver it completely.
                                all_heads_delivered = False
                                chunks_delivered = False
                                break
                            chunks_delivered = True
                            reply_to = new_id

                        if all_heads_delivered:
                            self._accumulated = chunks[-1]
                            # The head chunks are sealed.  Clear the edit target
                            # so the remaining tail is sent as a fresh active
                            # chunk, then edited by subsequent deltas.
                            self._message_id = None
                            self._message_created_ts = None
                            self._last_sent_text = ""
                        else:
                            # A prior head may have landed before a later head
                            # failed.  Do not edit that sealed message with the
                            # unsplit full payload; let the fallback path retry.
                            self._message_id = None
                            self._message_created_ts = None
                            self._last_sent_text = ""

                        if chunks_delivered:
                            # A sealed head is on screen, so this turn is now a
                            # multi-message delivery.  Flag it BEFORE the tail
                            # send below: the fresh-final route replaces every
                            # tracked preview with one message, which is only
                            # valid while the active message holds the whole
                            # answer.  Once heads are sealed it does not, and
                            # deleting them would drop delivered text (#78541).
                            self._turn_split_delivery = True

                        self._last_edit_time = time.monotonic()
                        if got_done:
                            tail_delivered = True
                            if self._accumulated:
                                tail_delivered = await self._send_or_edit(
                                    self._accumulated, finalize=True,
                                )
                            # Only claim final delivery if the sealed chunks and
                            # final tail actually landed.  ``_already_sent`` may
                            # be True from prior progress/fallback state (#10748).
                            self._final_response_sent = chunks_delivered and tail_delivered
                            if self._final_response_sent:
                                self._final_content_delivered = True
                                # Multi-message split delivery — record the
                                # unsplit ledger payload so the gateway can
                                # still reconcile against final_response
                                # (#71643, #78541).
                                self._turn_split_delivery = True
                                self._record_turn_final_payload(self._accumulated)
                            return
                        if got_segment_break:
                            self._message_id = None
                            self._fallback_final_send = False
                            self._fallback_prefix = ""
                            if not self._accumulated:
                                continue

                        # This iteration consumed a _FLUSH barrier and delivered
                        # the buffered prose via the chunk loop above, then takes
                        # an early `continue` that skips the bottom-of-loop set.
                        # Signal here so flush_pending_sync() doesn't stall the
                        # full timeout waiting on already-delivered content.
                        if got_flush:
                            self._signal_flush(flush_event)
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
                        if split_at < _cp_budget // 2:
                            split_at = _cp_budget
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
                        # Sealed head chunk delivered — this turn is now a
                        # multi-message delivery (#71643 record semantics).
                        self._turn_split_delivery = True

                    display_text = self._accumulated
                    if not got_done and not got_segment_break and commentary_text is None:
                        # Native streaming with tool-progress: compose frame
                        # content that includes tool status overlay. The cursor
                        # is appended to the composed content for consistency.
                        if self._use_native_streaming:
                            display_text = self._compose_frame_content()
                            if display_text and self.cfg.cursor:
                                display_text += self.cfg.cursor
                        else:
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
                    # Reset tool_progress_active flag after frame delivery —
                    # the lines are still in _tool_progress_lines (for the next
                    # frame's compose) but we don't need to trigger another
                    # should_edit until new progress arrives.
                    if self._tool_progress_active:
                        self._tool_progress_active = False

                if got_done:
                    if self._accumulated or self._message_id is not None or self._already_sent:
                        await self._notify_before_finalize()
                    # Final edit without cursor. If progressive editing failed
                    # mid-stream, send a single continuation/fallback message
                    # here instead of letting the base gateway path send the
                    # full response again.
                    if (
                        self._awaiting_reopen_after_boundary
                        and not self._native_stream_opened
                        and not self._accumulated
                    ):
                        # Clarify reopen boundary (LAZY path), but the agent
                        # produced no post-prompt content.  The pre-prompt stream
                        # was already finalized into a stable bubble at the
                        # boundary, and no fresh stream was ever re-seeded, so
                        # there is nothing on screen to close.  Do NOT re-seed a
                        # fresh stream just to emit a lone "✅" placeholder — that
                        # would leave a meaningless empty bubble below the
                        # question.  Close quietly; the finalized bubble stands.
                        logger.debug(
                            "Clarify reopen boundary with no post-prompt content "
                            "— skipping lone-placeholder finalize (turn=%s)",
                            self._turn_id,
                        )
                    elif (
                        self._reopen_seeded_eagerly
                        and self._native_stream_opened
                        and not self._accumulated
                        and not current_update_visible
                    ):
                        # EAGER-seed path: the typing bubble is ALREADY on screen
                        # (opened the instant the user answered), but the agent
                        # then produced no content.  Unlike the lazy case we
                        # cannot skip — an open, empty typing bubble would hang
                        # forever.  Close it with an empty finalize (NOT a lone
                        # "✅", which would be a meaningless bubble below the
                        # question).  Leave the delivery flags as-is: nothing
                        # substantive was delivered, so the gateway's own
                        # whole-response filter still governs any fallback.
                        try:
                            await self.adapter.send_stream_frame(
                                "",
                                finalize=True,
                                chat_id=self.chat_id,
                                reply_to=self._initial_reply_to_id,
                                turn_id=self._turn_id,
                            )
                        except Exception as e:
                            logger.debug(
                                "Eager-seed empty finalize failed: %s", e,
                            )
                        self._native_stream_opened = False
                        self._native_last_pushed_len = 0
                        # Reset for symmetry with _suppress_silence_marker; the
                        # consumer is per-turn today so this is defensive, but it
                        # keeps the flag from leaking if a consumer is ever reused
                        # across turns.
                        self._reopen_seeded_eagerly = False
                        logger.debug(
                            "Eager reopen seed but no post-answer content — "
                            "closed empty typing bubble (turn=%s)",
                            self._turn_id,
                        )
                    elif self._use_native_streaming:
                        # Native streaming MUST always close the stream with
                        # finish=true — even when _accumulated is empty (e.g.
                        # tool-only turns with no text output). Mirror OpenClaw's
                        # finishThinkingStream: use a placeholder if needed.
                        if not current_update_visible:
                            close_text = self._accumulated or "✅"
                            self._final_response_sent = await self._send_or_edit(
                                close_text, finalize=True,
                            )
                            if self._final_response_sent:
                                self._final_content_delivered = True
                        else:
                            self._final_response_sent = True
                            self._final_content_delivered = True
                    elif self._accumulated:
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
                    elif self._use_native_streaming:
                        # Native streaming (WeCom): commentary is sent as an
                        # independent message via adapter.send(), but we must
                        # NOT reset _accumulated — the native stream is
                        # cumulative and a reset would lose all pre-commentary
                        # text. Subsequent frames must still carry the full
                        # accumulated content. Same rationale as segment-break
                        # no-op for native streaming.
                        await self._send_commentary(commentary_text)
                        self._last_edit_time = time.monotonic()
                    else:
                        self._reset_segment_state()
                        await self._send_commentary(commentary_text)
                        self._last_edit_time = time.monotonic()
                        self._reset_segment_state()

                # Tool boundary: for edit-based platforms, reset message state
                # so the next text chunk creates a fresh message below tool-progress.
                # For WeCom native streaming: NO reset — stream uses cumulative text,
                # so resetting would lose pre-boundary content in subsequent frames.
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
                    # WeCom native streaming also uses cumulative text — each
                    # frame must carry the full content so far, so a segment
                    # break must NOT reset accumulated state or subsequent
                    # frames lose the pre-boundary text.
                    if (
                        self._stream_is_message()
                        and self._use_draft_streaming
                    ) or self._use_native_streaming:
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
            # Safety net: if run() exits (normal return, cancellation, or
            # exception) while a _FLUSH barrier is still queued or was consumed
            # but not yet signaled, wake any waiters now. Without this a caller
            # blocked in flush_pending_sync() would stall the full timeout when
            # the consumer dies mid-flush. Bounded either way, but this makes
            # the common case instant instead of timeout-delayed.
            try:
                while True:
                    item = self._queue.get_nowait()
                    if (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and item[0] is _FLUSH
                    ):
                        self._signal_flush(item[1])
            except queue.Empty:
                pass
            except Exception:
                pass

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
                metadata=self._metadata_for_send(
                    final=final,
                    expect_edits=not final,
                ),
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
    def _balance_fences_across_chunks(chunks: "list[str]") -> "list[str]":
        """Close orphaned ``` fences at each chunk boundary and reopen on the next.

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
                    self._delivery_ambiguous = True
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

    def _resolve_native_streaming(self) -> bool:
        """Decide whether this run should use the native-streaming transport.

        Native streaming (e.g. WeCom's ``msgtype: "stream"``) routes ALL
        frames — first send, mid-stream updates, and the final ``finish=true``
        — through ``adapter.send_stream_frame()``. It takes precedence over
        both edit and draft transports because it provides the best client
        experience on platforms whose API is built for it (the WeCom client,
        for example, renders cumulative content updates in-place with a
        built-in typing animation while the stream stays open).

        Adapter eligibility:
          1. Must subclass :class:`BasePlatformAdapter` (MagicMock test
             adapters fall back to edit).
          2. Must declare ``SUPPORTS_NATIVE_STREAMING = True`` at the class
             level.
          3. Must provide ``supports_native_streaming(chat_type, metadata)``
             returning truthy for this chat.
        """
        if not isinstance(self.adapter, _BasePlatformAdapter):
            return False
        if not getattr(type(self.adapter), "SUPPORTS_NATIVE_STREAMING", False):
            return False
        probe = getattr(self.adapter, "supports_native_streaming", None)
        if probe is None:
            return False
        try:
            supported = probe(
                chat_type=self.cfg.chat_type or None,
                metadata=self.metadata,
            )
        except Exception:
            logger.debug(
                "supports_native_streaming probe raised", exc_info=True,
            )
            return False
        return bool(supported)

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
            _md = self._metadata_for_send(final=False) or {}
            _md["_interim_send"] = True
            # Only pass reply_to for platforms that use reply-anchoring for
            # threading. Discord/Telegram use native thread_id in metadata;
            # passing reply_to on every commentary creates reply spam.
            _plat = getattr(getattr(self.adapter, "platform", None), "value", None)
            _platform_name = str(_plat or getattr(self.adapter, "name", "")).lower()
            _needs_reply_anchor = _platform_name in ("buzz", "slack", "mattermost", "feishu")
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                reply_to=self._initial_reply_to_id if _needs_reply_anchor else None,
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
                # Pass the chat id so multi-platform adapters (relay) resolve
                # the decision through THIS chat's negotiated platform, not
                # the primary identity's.  Without it a Slack-primary relay
                # with unfurl force-on misroutes a fronted Telegram/Discord
                # chat's final through the fresh-send lane (duplicate
                # delivery: those descriptors advertise no ``delete`` op),
                # and the mirror posture leaves fronted Slack chats dark.
                result = fn(text, metadata=self.metadata, chat_id=self.chat_id)
            except TypeError:
                try:
                    # Single-platform hook signature (Telegram, base class):
                    # (content, metadata=None) — no chat_id keyword.
                    result = fn(text, metadata=self.metadata)
                except TypeError:
                    # Adapter / test double whose hook doesn't accept the
                    # metadata keyword — fall back to the positional-only form.
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
            # Fresh send carried exactly ``text`` — record it so the gateway
            # can reconcile the flag against the completed response
            # (#71643/#95382 content-vs-flag contract).
            self._record_turn_final_payload(text)
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
        # Native-stream bubbles (e.g. WeCom) are NOT deletable messages — they
        # are an open stream closed by a finalize frame, not delete_message.
        # If a stream is open (notably one opened by an EAGER re-seed after a
        # clarify answer, where the typing bubble is already on screen with no
        # content), close it with an empty finalize so it doesn't hang forever.
        # Do this before the delete loop; keep the delivery flags False below.
        if self._native_stream_opened:
            try:
                await self.adapter.send_stream_frame(
                    "",
                    finalize=True,
                    chat_id=self.chat_id,
                    reply_to=self._initial_reply_to_id,
                    turn_id=self._turn_id,
                )
            except Exception as e:
                logger.debug(
                    "Silence-marker native stream close failed: %s", e,
                )
            self._native_stream_opened = False
            self._native_last_pushed_len = 0
            self._reopen_seeded_eagerly = False

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
        self._delivery_ambiguous = False
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
            # For native streaming: even when the display text is empty (e.g.
            # MEDIA-only response cleaned away), we MUST send a finalize frame
            # to close the thinking bubble. Use placeholder text.
            if finalize and self._use_native_streaming and self._native_stream_opened:
                try:
                    ok = await self.adapter.send_stream_frame(
                        "✅",
                        finalize=True,
                        chat_id=self.chat_id,
                        reply_to=self._initial_reply_to_id,
                        turn_id=self._turn_id,
                    )
                    if ok:
                        self._final_response_sent = True
                        self._final_content_delivered = True
                except Exception as e:
                    logger.debug("Finalize empty stream failed: %s", e)
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

        # Native streaming transport (e.g. WeCom): every frame — first send,
        # mid-stream updates, and the final answer — flows through
        # adapter.send_stream_frame(), which manages the underlying stream
        # lifecycle (init seed → cumulative updates → finish=true). The
        # adapter's send/edit_message paths are NOT touched in this mode.
        #
        # Throttling: WeCom AI Bot caps replies at ~30 frames/min per chat.
        # With 15 concurrent users, we need ≤2 frames per turn on average
        # to stay under the limit. 60 chars ≈ one short sentence, which
        # produces 3-5 frames per turn — close to OpenClaw's block-level cadence.
        if self._use_native_streaming:
            # Re-seed if stream was closed (e.g., by approval boundary)
            # and we have new content to send.
            if not self._native_stream_opened and text:
                try:
                    seed_ok = await self.adapter.send_stream_frame(
                        "",
                        chat_id=self.chat_id,
                        reply_to=self._initial_reply_to_id,
                        turn_id=self._turn_id,
                    )
                    if seed_ok:
                        self._native_stream_opened = True
                        # A fresh stream is open — post-prompt content will
                        # stream into it, so got_done no longer needs the
                        # lone-placeholder guard for this turn.
                        self._awaiting_reopen_after_boundary = False
                        # INFO (temporary latency probe): this is the moment the
                        # C bubble / typing animation first becomes visible after
                        # a clarify answer.  Comparing this timestamp to the
                        # boundary-finalize log below quantifies the "typing is
                        # slow to reappear" delay the user reported.
                        logger.info(
                            "[latency] Re-opened native stream after boundary "
                            "(turn=%s, waited for first delta)",
                            self._turn_id,
                        )
                    else:
                        self._use_native_streaming = False
                except Exception as e:
                    logger.debug("Re-seed failed, disabling native streaming: %s", e)
                    self._use_native_streaming = False

        if self._use_native_streaming:
            # For WeCom native streaming: segment breaks should NOT finalize
            # the stream. WeCom renders each finalize as a separate message bubble.
            # Only turn-final (got_done) and approval boundary should close the stream.
            # Tool boundary segment breaks just continue accumulating in the same stream.
            if finalize and not is_turn_final:
                finalize = False

            # Fire-and-forget: send immediately when content differs from
            # the last pushed frame. No buffering / throttle — WeCom long-
            # connection mode has no polling cadence, so every cumulative
            # update is pushed as soon as it arrives.
            if not finalize and text == self._last_sent_text:
                return True  # unchanged — skip

            # B2 — timeout-inversion race fix. For a finalize frame, mark
            # delivery OPTIMISTICALLY, before send_stream_frame blocks on the
            # ack. The finalize frame's bytes are written to the wire by an
            # independent control-worker task *before* the ack wait begins, and
            # for WeCom a frame on the wire is already rendered by the client
            # (the same premise the ack-timeout-as-success path already relies
            # on). Setting the flag here means a gateway join-cancel during the
            # ack wait — the timeout inversion between run.py's stream_task join
            # and adapter._REPLY_ACK_TIMEOUT — can no longer strand
            # final_content_delivered=False while WeCom has already shown the
            # message, which is what produced the duplicate normal send
            # (see tests/gateway/test_wecom_double_send.py and
            # docs/rca-wecom-stream-final-ack-timeout-duplicate.md).
            #
            # A DEFINITIVE dispatch failure (ok is False below: stream never
            # opened, 846608 expired, errcode 6000, or the call raised) rolls
            # the mark back so the edit/send fallback still delivers exactly
            # once. Residual window: if the consumer is cancelled between this
            # optimistic mark and the control worker actually writing the bytes
            # (queue latency, sub-ms in practice), the message could be
            # suppressed without being sent — far rarer than the guaranteed
            # duplicate this replaces, and the send-path idempotency guard
            # cannot help there (nothing was sent). Accepted trade-off.
            _optimistic_finalize = bool(finalize)
            if _optimistic_finalize:
                self._final_response_sent = True
                self._final_content_delivered = True
                # Record what this finalize frame carries so the gateway's
                # content reconciliation (#71643/#95382) can judge the flag:
                # a frame holding only a stale/partial snapshot must not
                # suppress the corrective send of the complete response.
                self._record_turn_final_payload(text)

            ok = False
            try:
                ok = await self.adapter.send_stream_frame(
                    text,
                    finalize=finalize,
                    chat_id=self.chat_id,
                    reply_to=self._initial_reply_to_id,
                    turn_id=self._turn_id,
                )
            except Exception as e:
                logger.debug(
                    "send_stream_frame raised, disabling native streaming: %s", e,
                )
                ok = False

            if ok:
                self._already_sent = True
                self._last_sent_text = text
                self._native_last_pushed_len = len(text)
                if finalize:
                    self._final_response_sent = True
                    self._final_content_delivered = True
                return True

            # Dispatch failed definitively — roll back the optimistic finalize
            # mark so the edit/send fallback below delivers the content once.
            if _optimistic_finalize:
                self._final_response_sent = False
                self._final_content_delivered = False
                # Roll back the recorded payload too — nothing was delivered.
                self._delivered_final_text = None

            # Native streaming refused / failed — switch off so this and
            # subsequent frames take the edit/send fallback path below.
            # The adapter is responsible for marking the chat as expired
            # so it doesn't keep retrying the dead stream session.
            self._use_native_streaming = False

            # If the stream bubble was opened (seed frame succeeded), try
            # best-effort finalize to close it before falling back to send().
            # This prevents leaving an unclosed thinking stream visible to the
            # user. Check _native_stream_opened (not _native_last_pushed_len)
            # because the seed frame has zero length but still opens the bubble.
            if self._native_stream_opened:
                try:
                    await self.adapter.send_stream_frame(
                        text,
                        finalize=True,
                        chat_id=self.chat_id,
                        reply_to=self._initial_reply_to_id,
                        turn_id=self._turn_id,
                    )
                    logger.debug("Native fallback: finalized stream (best-effort close)")
                    # DO NOT mark _final_content_delivered here.
                    # The finalize frame closes the typing bubble, but WeCom may
                    # not actually render the content (e.g., errcode 6000 race).
                    # Let the fallback send() path deliver the content reliably.
                except Exception as e:
                    logger.debug(
                        "Native fallback: failed to finalize stream: %s", e,
                    )
            # Fall through to the edit/send paths so any accumulated text
            # still reaches the user as a one-shot proactive markdown send.


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
