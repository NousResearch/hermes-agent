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
from contextlib import suppress
from typing import TYPE_CHECKING, Any, List, Optional

logger = logging.getLogger(__name__)

BURST_SEPARATOR = "\n\n"
# Defensive ceiling: Slack's documented chunk cap is unenforced (2.6k verified live), but a
# runaway turn must not ship an unbounded API body. Above this the feed keeps its head and says
# so -- visible truncation, never silent (acceptance criterion 2).
MAX_FEED_CHARS = 24_000
_TRUNCATION_ROW = "[... narration truncated at %d chars]" % MAX_FEED_CHARS


def sanitize_thinking_burst(text: Any) -> Optional[str]:
    """Think-stripped, redacted narration text (REV-1), or None when nothing visible remains."""
    if not isinstance(text, str) or not text.strip():
        return None
    from agent.history_commentary import visible_commentary

    cleaned = visible_commentary(text).strip()
    return cleaned or None


@dataclasses.dataclass
class _ThinkingCardState:
    """Per-turn thinking-card rail state for ``send_thinking_card_messages``."""

    adapter: Any
    feed_parts: List[str] = dataclasses.field(default_factory=list)
    fallback_msg_id: Optional[str] = None
    native_failed: bool = False
    # Same terminal semantics as the tool card: an egress-guard DECLINE ends delivery for the
    # turn. Re-sending the feed as plain text would launder an authorization refusal.
    publication_suppressed: bool = False
    truncated: bool = False

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
                    await asyncio.sleep(0.3)
                    continue
                if self._agent_interrupted():
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
                    burst = self._thinking_card_absorb(raw)
                    if burst:
                        st.append(burst)
            if st.feed_parts and ctx._run_still_current() and not self._agent_interrupted():
                await self._thinking_card_publish(st)
        finally:
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
