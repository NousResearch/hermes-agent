"""Slack-native thinking-card rail for ``TurnRunner`` (Option A, rendering variant A1/D).

One per-turn card carries the agent's narration (assistant text between tool calls) as a
cumulative feed, replacing the one-thread-reply-per-segment interim lane. Rendering contract
(proven live against Slack 2026-10-15): ``plan_update`` title plus ONE ``task_update`` whose
``details`` holds the whole feed, updated in place as bursts arrive -- collapsed, the thread
shows a single row.

REV-1 guard: bursts pass through ``visible_commentary`` (think-block strip + secret redaction)
at this sink, and tag-only bursts are dropped, so no reasoning content reaches the card even if
an upstream path fails to strip it.

This is a topical sibling of ``gateway/run_turn_runner.py``; it duck-types the TurnRunner
(``self._ctx``, ``self._runner``, ``self._agent_interrupted``) and never imports the facade.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, List, Optional

from agent.i18n import t

logger = logging.getLogger(__name__)

BURST_SEPARATOR = "\n\n"
# Defensive ceiling: Slack's documented chunk cap is unenforced (2.6k verified live), but a
# runaway turn must not ship an unbounded API body. Above this the feed keeps its head and says
# so -- visible truncation, never silent (acceptance criterion 2).
MAX_FEED_CHARS = 24_000
# A reasoning block gets its "Thinking…" row once it has run this long; shorter blocks never show.
REASONING_CARD_AFTER_S = 1.5
REASONING_TAIL_CHARS = 200
_TRUNCATION_ROW = f"[... narration truncated at {MAX_FEED_CHARS} chars]"
_REASONING_EVENTS = frozenset({"reasoning", "reasoning.end"})


def sanitize_thinking_burst(text: Any) -> Optional[str]:
    """Think-stripped, redacted narration text (REV-1), or None when nothing visible remains."""
    if not isinstance(text, str) or not text.strip():
        return None
    from agent.history_commentary import visible_commentary

    from agent.redact import redact_for_egress

    # The card leaves the process for a chat platform: egress redaction applies even when
    # ``security.redact_secrets`` is off for local logs.
    cleaned = redact_for_egress(visible_commentary(text)).strip()
    return cleaned or None


def reasoning_tail(notes: str, limit: int = REASONING_TAIL_CHARS) -> str:
    """The last ``limit`` characters of a reasoning block on one line, redacted for egress."""
    from agent.redact import redact_for_egress

    text = " ".join(redact_for_egress(notes).split())
    return text if len(text) <= limit else "…" + text[-limit:]


@dataclasses.dataclass
class _ReasoningBlock:
    """One live reasoning block: when it started, its notes, and whether its row is on the card."""

    row_id: str
    started: float
    notes: str = ""
    shown: bool = False


@dataclasses.dataclass
class _ThinkingCardState:
    """Per-turn thinking-card rail state for ``send_thinking_card_messages``."""

    adapter: Any
    feed_parts: list[str] = dataclasses.field(default_factory=list)
    fallback_msg_id: Optional[str] = None
    native_failed: bool = False
    # Same terminal semantics as the tool card: an egress-guard DECLINE ends delivery for the
    # turn. Re-sending the feed as plain text would launder an authorization refusal.
    publication_suppressed: bool = False
    truncated: bool = False
    block: Optional[_ReasoningBlock] = None
    blocks_seen: int = 0
    last_notes: str = ""

    @property
    def feed_text(self) -> str:
        return BURST_SEPARATOR.join(self.feed_parts)

    def append(self, burst: str) -> bool:
        """Queue one burst as its own row. Visible truncation, never silent (acceptance
        criterion 2): the burst is cut to fit, then a marker ROW lands beside it.
        Returns False once the feed is capped (further bursts dropped)."""
        if self.truncated:
            return False
        room = MAX_FEED_CHARS - sum(len(p) for p in self.feed_parts)
        if len(burst) <= room:
            self.feed_parts.append(burst)
            return True
        self.feed_parts.append(burst[:max(room, 0)])
        self.feed_parts.append(_TRUNCATION_ROW)
        self.truncated = True
        return True


class TurnRunnerThinkingMixin:
    """Thinking-card drain, mixed into ``TurnRunner``."""

    def _route_interim_narration(self, text: Any, *, already_streamed: bool = False) -> bool:
        """Thinking-card sink: queue narration for the card instead of per-segment replies.

        Returns True when the thinking card consumed the event (caller must not fall through to
        the interim-message lane). ``already_streamed`` text belongs to the answer stream, not
        narration, and is left to the existing segment-break path.
        """
        ctx = self._ctx
        if not (ctx._native_slack_thinking_cards and ctx.thinking_queue is not None):
            return False
        if not already_streamed and str(text or "").strip():
            ctx.thinking_queue.put({"type": "thinking", "text": text})
        return True

    async def send_progress_messages(self):
        """The progress drain, with the thinking card's drain beside it for the same turn."""
        if not getattr(self._ctx, "_native_slack_thinking_cards", False):
            return await super().send_progress_messages()
        thinking = asyncio.create_task(self.send_thinking_card_messages())
        try:
            await super().send_progress_messages()
        finally:
            thinking.cancel()
            with suppress(asyncio.CancelledError):
                await thinking

    def _init_thinking_cards(self) -> None:
        """Turn the per-turn thinking card on when the Slack adapter opted in (``thinking_cards``).

        Resolved per turn from the delivering adapter; the card shares the tool card's opt-in.
        """
        ctx = self._ctx
        adapter = None
        if getattr(ctx, "_native_slack_task_cards", False) and self._runner is not None:
            adapter = self._runner._delivery_adapter_for(ctx.source)
        enabled = getattr(adapter, "thinking_cards_enabled", None)
        if callable(enabled) and enabled():
            ctx._native_slack_thinking_cards = True
            ctx.thinking_queue = queue.Queue()

    def _thinking_reasoning_callback(self):
        """``agent.reasoning_callback`` for "Thinking…" rows: thinking card AND show_reasoning on."""
        ctx = self._ctx
        resolve = getattr(self._runner, "_show_reasoning_for_source", None)
        if not (ctx._native_slack_thinking_cards and ctx.thinking_queue is not None and callable(resolve)
                and resolve(ctx.source)):
            return None

        def _reasoning(text):
            if text and ctx._run_still_current() and not self._agent_interrupted():
                ctx.thinking_queue.put({"type": "reasoning", "text": str(text), "at": time.monotonic()})
        return _reasoning

    async def _reasoning_row(self, st: _ThinkingCardState, *, done: bool) -> None:
        block = st.block
        send = getattr(st.adapter, "send_reasoning_card_row", None)
        if block is None or not callable(send) or st.native_failed or st.publication_suppressed:
            return
        ctx = self._ctx
        if done:
            seconds = max(1, round(time.monotonic() - block.started))
            title, status, details = t("gateway.progress.reasoning_card_done", seconds=seconds), "complete", (
                reasoning_tail(block.notes) or None)
        else:
            title, status, details = t("gateway.progress.reasoning_card_live"), "in_progress", None
        result = await send(ctx.source.chat_id, block.row_id, title=title, status=status, details=details,
                            reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata)
        block.shown = bool(getattr(result, "success", False)) or block.shown

    async def _thinking_card_reasoning_event(self, st: _ThinkingCardState, raw: dict) -> None:
        """Track reasoning blocks: a row once a block passes 1.5 s, "Thought for Ns" when it ends."""
        if raw.get("type") == "reasoning":
            text = str(raw.get("text") or "")
            if st.block is None and text.strip() and text in st.last_notes:
                return  # the provider's whole-block re-delivery after the block already ended
            if st.block is None:
                st.blocks_seen += 1
                st.block = _ReasoningBlock(f"reasoning-{st.blocks_seen}", float(raw.get("at") or time.monotonic()))
            # A provider may re-deliver the whole block after its deltas: keep it once.
            st.block.notes = text if text.startswith(st.block.notes) and st.block.notes else st.block.notes + text
            return
        if st.block is not None:
            if st.block.shown:
                await self._reasoning_row(st, done=True)
            st.last_notes = st.block.notes
            st.block = None

    async def _thinking_card_tick(self, st: _ThinkingCardState) -> None:
        block = st.block
        if block is not None and not block.shown and time.monotonic() - block.started >= REASONING_CARD_AFTER_S:
            await self._reasoning_row(st, done=False)

    if TYPE_CHECKING:  # borrowed from TurnRunner at mix-in time
        _ctx: Any
        _runner: Any

        def _agent_interrupted(self) -> bool: ...
        async def _send_progress_text(self, st: Any, text: str) -> Any: ...

    async def _thinking_card_send_or_edit_fallback(self, st: _ThinkingCardState) -> None:
        """One editable text message carrying the feed (tool-card fallback ladder)."""
        ctx = self._ctx
        if st.publication_suppressed:
            return
        text = st.feed_text
        if st.fallback_msg_id:
            result = await st.adapter.edit_message(
                chat_id=ctx.source.chat_id, message_id=st.fallback_msg_id,
                content=text, metadata=ctx._progress_metadata,
            )
            if getattr(result, "success", False):
                return
            from gateway.relay.egress import declined_send

            if declined_send(result):
                logger.warning(
                    "Thinking-card fallback edit DECLINED by the connector's egress guard; "
                    "suppressing narration delivery for the rest of this turn"
                )
                st.publication_suppressed = True
                return
        result = await self._send_progress_text(st, text)
        if getattr(result, "success", False) and getattr(result, "message_id", None):
            st.fallback_msg_id = str(result.message_id)

    async def _thinking_card_publish(self, st: _ThinkingCardState) -> None:
        ctx = self._ctx
        if not st.feed_parts or st.publication_suppressed:
            return
        if not st.native_failed:
            result = await st.adapter.send_thinking_card_progress(
                chat_id=ctx.source.chat_id, bursts=list(st.feed_parts),
                reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata,
                fallback_text=st.feed_text,
            )
            if getattr(result, "success", False):
                return
            from gateway.relay.egress import declined_send

            if declined_send(result):
                st.publication_suppressed = True
                st.native_failed = True
                logger.warning(
                    "Slack thinking-card progress DECLINED by the connector's egress guard "
                    "-- suppressing delivery for the rest of this turn"
                )
                return
            st.native_failed = True
            logger.warning(
                "Slack thinking-card progress failed; falling back to an editable text "
                "update: %s", getattr(result, "error", "unknown error"),
            )
        await self._thinking_card_send_or_edit_fallback(st)

    def _thinking_card_absorb(self, raw: Any) -> Optional[str]:
        """Sanitize one queue payload into a burst (REV-1); None when it must not render."""
        text = raw.get("text") if isinstance(raw, dict) else raw
        return sanitize_thinking_burst(text)

    async def send_thinking_card_messages(self) -> None:
        """Drain narration bursts into one per-turn thinking card.

        See plan-option-a.md (Option A). Cancellation drains the queue once more so the last
        bursts land before the finalizing ``stop_thinking_card_progress``.
        """
        ctx = self._ctx
        adapter = self._runner._delivery_adapter_for(ctx.source) if ctx.thinking_queue else None
        if not adapter or not hasattr(adapter, "send_thinking_card_progress"):
            return
        st = _ThinkingCardState(adapter)
        try:
            while ctx._run_still_current():
                try:
                    raw = ctx.thinking_queue.get_nowait()
                except queue.Empty:
                    if not self._agent_interrupted():
                        await self._thinking_card_tick(st)
                    await asyncio.sleep(0.3)
                    continue
                if self._agent_interrupted():
                    continue
                if isinstance(raw, dict) and raw.get("type") in _REASONING_EVENTS:
                    await self._thinking_card_reasoning_event(st, raw)
                    continue
                burst = self._thinking_card_absorb(raw)
                if not burst:
                    continue
                st.append(burst)
                await self._thinking_card_publish(st)
        except asyncio.CancelledError:
            with suppress(Exception):
                while True:
                    try:
                        raw = ctx.thinking_queue.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(raw, dict) and raw.get("type") in _REASONING_EVENTS:
                        if raw.get("type") == "reasoning" and st.block is not None:
                            st.block.notes += str(raw.get("text") or "")
                        continue
                    burst = self._thinking_card_absorb(raw)
                    if burst:
                        st.append(burst)
            if st.feed_parts and ctx._run_still_current() and not self._agent_interrupted():
                await self._thinking_card_publish(st)
        finally:
            if st.block is not None and st.block.shown:
                # Terminal title before the close, also on /stop: never a frozen "Thinking…" row.
                with suppress(Exception):
                    await self._reasoning_row(st, done=True)
            if hasattr(adapter, "stop_thinking_card_progress"):
                try:
                    await adapter.stop_thinking_card_progress(
                        ctx.source.chat_id,
                        reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("thinking-card stop failed during turn cleanup", exc_info=True)
