"""ChatCardEngine — per-chat CardKit streaming-card session engine.

Sessions are keyed by **chat** (plus thread for Feishu topics): draft frames
from the gateway stream consumer carry no message identity, and each frame's
content is a **full snapshot**, so the ANSWER segment is replaced wholesale
rather than appended — replaying snapshots must not duplicate text.

Segment model (v1): at most one REASONING and one ANSWER segment per turn
(merged across the whole turn); a TOOL panel is appended on the first tool
event and refreshed in place. Building blocks: SegmentState / ToolUseTracker /
FlushController / the CardKit v2 builder / the executor-integrated client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .builder import (
    HEARTBEAT_ELEMENT_ID,
    MAX_FAVORITE_MODELS,
    build_complete_card,
    build_cron_card,
    build_streaming_card_v2,
    cap_reasoning_text,
)
from .markdown import optimize_markdown_style
from .flush import FlushController
from .segment_helper import (
    build_add_segment_action,
    build_reasoning_finalized_action,
    build_tool_update_action,
    tool_segment_end,
)
from .segments import Segment, SegmentState, SegmentType
from .text import strip_reasoning_tags
from .tooluse import ToolUseTracker

# Standard module logger; the engine logs lifecycle decisions at INFO.
logger = logging.getLogger(__name__)

# Cap for heartbeat/interim text before it enters the status line (bounds card size).
_HEARTBEAT_MAX_LEN = 300


@dataclass
class ChatSession:
    """Streaming-card session for one chat (one card per turn)."""

    chat_id: str
    reply_to: str | None = None  # reply anchor (user message id); None = send flat to the chat
    created_at: float = field(default_factory=time.time)
    state: str = "creating"  # creating → streaming → completed / failed
    card_id: str | None = None
    card_msg_id: str | None = None
    answer_seg: Segment | None = None
    thread_id: str | None = None  # topic thread (Feishu topic/reply chain); None = main chat
    redirected: bool = False  # diagnostics: this turn was restarted by a redirect (seal/replace logs)
    model_switch: dict[str, str] | None = None  # footer switch-button data (carried through NOTICE re-renders)
    redirect_anchor: str | None = None  # new card anchor after redirect (user correction message id, from the ack reply_to)
    # Straggler guard for redirects: snapshots squeezed out before the old model
    # request is cancelled are supersets of the old turn's content — drop them on
    # a prefix match so the old answer never flashes into the new card; cleared
    # once the first unrelated content passes.
    straggler_guard: str = ""
    # Draft anchors accepted as this turn's own (followup boundary detection). A
    # redirected turn legitimately carries two anchors (the correction message
    # plus the old turn's message); both count as the same turn.
    accepted_anchors: set[str] = field(default_factory=set)
    _reasoning_logged: bool = False
    tool_seg: Segment | None = None
    heartbeat_text: str = ""
    sequence: int = 0
    card_create_task: asyncio.Task | None = None
    segment_state: SegmentState = field(default_factory=SegmentState)
    tool_tracker: ToolUseTracker = field(default_factory=ToolUseTracker)
    flush: FlushController | None = None

    completed_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in ("completed", "failed")


class ChatCardEngine:
    """Per-chat streaming-card engine, driven by the Feishu adapter.

    Lifecycle: the probe opens the session and creates the card as a background
    task (never blocking the transport) → each frame replaces the answer text
    wholesale → the turn's final state renders the complete card via
    :meth:`complete`.
    """

    def __init__(self, client: Any, *, body_text_size: str = "normal_v2",
                 show_tool_use: bool = True, header_enabled: bool = False,
                 width_mode: str = "default",
                 footer_fields: list[list[str]] | None = None,
                 footer_show_label: bool = False,
                 footer_enabled: bool = True,
                 model_cycle: list[str] | None = None,
                 tool_panel_expanded: bool = False,
                 reasoning_panel_expanded: bool = False) -> None:
        self._client = client
        self._body_text_size = body_text_size
        self._show_tool_use = show_tool_use
        self._header_enabled = header_enabled
        self._width_mode = width_mode
        # Complete-card panel expansion (streaming.tool_panel_expanded /
        # reasoning_panel_expanded, falling back to the legacy panel_expanded key);
        # always collapsed while streaming — the title-row action tag suffices.
        self._tool_panel_expanded = tool_panel_expanded
        self._reasoning_panel_expanded = reasoning_panel_expanded
        # Footer styling reads the streaming.footer section of the gateway config so
        # every deployment renders one consistent shape.
        self._footer_fields = footer_fields or [["elapsed", "model", "context"]]
        self._footer_show_label = footer_show_label
        self._footer_enabled = footer_enabled
        # Rotation list for the footer model-switch button (streaming.footer.model_cycle,
        # derived from model.default + fallback_providers when unset) — no button
        # when the list has <2 entries or the current model is not on it. The
        # favorite-models picker persists its own JSON and takes precedence.
        self._model_cycle = [m.strip() for m in (model_cycle or []) if str(m).strip()]
        self._model_cycle_fallback = list(self._model_cycle)
        favorites = self._load_model_favorites()
        if favorites:
            self._model_cycle = favorites
        # All models Hermes can serve (management-card candidate pool), cached in-process.
        self._candidates_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._sessions: dict[str, ChatSession] = {}
        # Session key = (chat, thread): Feishu topics are separate Hermes sessions
        # (dm:oc_x:omt_y); card sessions must isolate at the same granularity or
        # topic and main-chat turns overwrite each other's cards.
        # chat → per-turn accumulated usage (grouped by session id, consumed by complete).
        self._usage: dict[str, dict[str, Any]] = {}
        # chat → model name from the latest usage report (drives the picker's current marker).
        self._chat_models: dict[str, str] = {}
        self._last_model = ""
        # chat → inbound message ids seen (registered on processing start; drained
        # followups register too) — the followup split gate requires the new
        # anchor to be a real inbound message.
        self._inbound_seen: dict[str, dict[str, float]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Any = None

    # ── usage aggregation (post_api_request → footer tokens / t/s) ──

    @staticmethod
    def _chat_of_session(session_id: str) -> str:
        """agent session key (agent:main:feishu:dm:<chat_id>) → chat_id; "" when unparseable."""
        m = re.search(r"oc_[0-9a-f]{16,40}", str(session_id or ""))
        return m.group(0) if m else ""

    def record_usage(self, session_id: str, usage: dict[str, Any] | None, model: str = "") -> None:
        """Accumulate one API call's usage into the owning chat's turn bucket."""
        if not usage:
            return
        chat_id = self._chat_of_session(session_id)
        bucket = self._usage.setdefault(
            chat_id, {"input": 0, "output": 0, "model": "", "context_max": 0,
                      "first_started": None, "last_ended": None})
        # Input tokens take the last value, not a sum: every request's prompt
        # carries the full history, so summing inflates it by orders of magnitude
        # (observed 880k for a 2k-token turn).
        bucket["input"] = max(bucket["input"], int(usage.get("prompt_tokens") or 0))
        bucket["output"] += int(usage.get("completion_tokens")
                                or usage.get("output_tokens") or usage.get("total_tokens") or 0)
        if model:
            bucket["model"] = model
        if usage.get("context_length"):
            bucket["context_max"] = int(usage["context_length"])
            bucket["context_used"] = int(usage.get("prompt_tokens") or 0)
        # API wall-clock span (including queueing/tool gaps) is the t/s denominator.
        if usage.get("started_at"):
            if bucket["first_started"] is None:
                bucket["first_started"] = usage["started_at"]
            bucket["last_ended"] = usage.get("ended_at") or bucket["last_ended"]
        if model:
            self._chat_models[chat_id or ""] = model
            self._last_model = model

    def _pop_usage(self, chat_id: str) -> dict[str, Any] | None:
        bucket = self._usage.pop(chat_id, None) or self._usage.pop("", None)
        if not bucket or not (bucket["input"] or bucket["output"]):
            return None
        return bucket

    @staticmethod
    def _skey(chat_id: str, thread_id: str | None) -> str:
        """Card session key: (chat, thread) composite; empty thread = main chat."""
        return f"{chat_id}::{thread_id}" if thread_id else chat_id

    def on_turn_started(self, chat_id: str, thread_id: str | None = None) -> None:
        """Turn start (probe time) → create the card immediately.

        Tool-heavy turns produce no draft until every tool has run; without an
        eager card the user sees nothing meanwhile. Create on the probe (flat
        send when no anchor yet); tool events and body text join the card as
        they arrive.
        """
        self._capture_loop()
        key = self._skey(chat_id, thread_id)
        existing = self._sessions.get(key)
        if (existing is not None and existing.is_terminal
                and time.time() - existing.completed_at < 5.0):
            # The probe is a 2s heartbeat loop: a trailing call right after a turn
            # completes is not a new turn — recreating would strand a loading card.
            return
        existing = self._sessions.get(key)
        if existing is None or existing.is_terminal:
            self._ensure_session(chat_id, thread_id)
            logger.info("turn started: chat=%s thread=%s",
                         chat_id[:12], (thread_id or "-")[:16])
        # Repeated heartbeat calls stay silent while the session lives.

    def note_inbound(self, chat_id: str, message_id: str) -> None:
        """Register an inbound message id (called on processing start; drained
        followups register at drain start too) — the data source for the
        followup split gate: a draft anchor may only split the card if it is a
        real inbound message. Consumer re-anchoring at tool boundaries is NOT
        an inbound message and must not split (observed: a redirected turn's
        post-file-write re-anchor split into an empty "Done." card)."""
        if not chat_id or not message_id:
            return
        bucket = self._inbound_seen.setdefault(chat_id, {})
        bucket[message_id] = time.time()
        while len(bucket) > 50:  # keep the most recent 50 per chat (bounded memory)
            bucket.pop(next(iter(bucket)))

    def _inbound_verified(self, chat_id: str, message_id: str | None) -> bool:
        """Whether the anchor is an inbound message this engine has seen."""
        if not message_id:
            return False
        return message_id in self._inbound_seen.get(chat_id, {})

    def mark_redirect(self, chat_id: str, anchor: str | None = None,
                      thread_id: str | None = None) -> None:
        """Redirect ack (user correction / interrupt mode) → seal the old card and
        open a new one immediately.

        A Hermes redirect re-anchors and continues the SAME run (the model
        request is cancelled, the correction injected, the loop retried) — the
        draft anchor stays put, so anchor-change detection cannot catch it.
        Splitting on the ack instead of the first content-bearing draft means
        reasoning and tool phases no longer render into the old card. The old
        card gets a red-flagged NOTICE closure; the new card anchors on the
        correction message and streams from the first millisecond.

        ``anchor`` is the ack's reply_to (the user's correction message id);
        the new card replies to it, falling back to the old anchor when absent.
        Straggler snapshots from the cancelled request are prefix-dropped in
        on_draft."""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return
        self._capture_loop()
        session.redirected = True  # diagnostics: mark the restarted turn in seal/replace logs
        if anchor:
            session.redirect_anchor = anchor
        logger.info("redirect boundary at ack: chat=%s anchor=%s",
                     chat_id[:12], anchor or "-")
        new = ChatSession(
            chat_id=chat_id, thread_id=thread_id,
            reply_to=session.redirect_anchor or session.reply_to,
            straggler_guard=(session.answer_seg.text if session.answer_seg else ""))
        # A redirected turn's draft anchor swings back to the old message id mid-turn
        # (consumer re-anchoring at tool boundaries) — both anchors count as this
        # turn, so the followup boundary cannot split it in half.
        new.accepted_anchors = {a for a in (session.redirect_anchor,
                                            session.reply_to) if a}
        # Terminate synchronously: sealing is async (close+update cross the network);
        # a still-streaming old session would attract the new turn's reasoning.
        session.state = "completed"
        self._sessions[self._skey(chat_id, thread_id)] = new
        self._ensure_session(chat_id, thread_id)  # create the new card now, not on the first draft
        self._seal_session(session, notice="↪ Restarted with the new instruction — see the new card below")

    # ── session queries ──

    def session_for(self, chat_id: str, thread_id: str | None = None) -> ChatSession | None:
        return self._sessions.get(self._skey(chat_id, thread_id))

    def active_session(self, chat_id: str, thread_id: str | None = None) -> ChatSession | None:
        """Active (non-terminal) session — the entry point for send() completion routing."""
        session = self._sessions.get(self._skey(chat_id, thread_id))
        if session is not None and not session.is_terminal:
            return session
        return None

    def latest_session_for_chat(self, chat_id: str) -> ChatSession | None:
        """Most recently created session for this chat across threads — background
        notices merge into the newest card (thread_id marks provenance, not the
        merge target)."""
        candidates = [s for s in self._sessions.values() if s.chat_id == chat_id]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.created_at)

    def streaming_sessions(self) -> list[ChatSession]:
        return [s for s in self._sessions.values() if s.state == "streaming"]

    def open_sessions(self) -> list[ChatSession]:
        """Non-terminal sessions (creating/streaming) — the "turn in flight" signal
        for usage routing (post_api_request lands mid-turn; before the first
        draft the session is still creating)."""
        return [s for s in self._sessions.values() if not s.is_terminal]

    def last_card_msg_id(self, chat_id: str, thread_id: str | None = None) -> str | None:
        """Newest card message id (reply anchor for document delivery; terminal counts)."""
        session = self._sessions.get(self._skey(chat_id, thread_id))
        return session.card_msg_id if session else None

    # ── streaming input ──

    def on_draft(self, chat_id: str, content: str, reply_to: str | None = None,
                 thread_id: str | None = None) -> None:
        """Draft frame (full snapshot) — ensure the session and card, replace the
        answer text wholesale.

        Called from async send_draft: the engine event loop is captured here
        (format_tool_event may later arrive synchronously from an agent worker
        thread and needs the call_soon_threadsafe path). reply_to is the user
        message anchor supplied by the transport draft metadata (recorded on
        the first frame, idempotent afterwards)."""
        self._capture_loop()
        session = self._ensure_session(chat_id, thread_id)
        guard = session.straggler_guard
        if guard and content and (content.startswith(guard) or guard.startswith(content)):
            # Redirect straggler: a snapshot squeezed out before the old request's
            # cancellation (a superset of the old turn) — drop it or the new card
            # flashes the old answer at the top.
            logger.info("redirect straggler draft dropped: len=%d",
                         len(content))
            return
        if guard:
            session.straggler_guard = ""  # first genuinely new content passed; stop intercepting
        accepted = session.accepted_anchors or (
            {session.reply_to} if session.reply_to else set())
        if (session.state != "creating" and reply_to and session.reply_to
                and reply_to not in accepted):
            if not self._inbound_verified(chat_id, reply_to):
                # A new anchor that is not an inbound message = consumer re-anchoring at a
                # tool boundary (same turn continues) — re-anchor in place, do not
                # split. The split gate must be "new anchor = new inbound message"
                # (queued followups drain with processing_start firing at drain
                # start, before the first frame).
                logger.info("anchor swing (not an inbound msg), "
                             "re-anchor in place: %s -> %s", session.reply_to, reply_to)
                session.reply_to = reply_to
                session.accepted_anchors.add(reply_to)
            else:
                # Draft anchor became a new inbound message = a new turn started (queued
                # followup drained): seal the old card with whatever it has (green
                # completed state) and open a new card for the new turn.
                logger.info(
                    "followup boundary: anchor %s -> %s (inbound), "
                    "sealing old card", session.reply_to, reply_to)
                old_session = session
                new = ChatSession(chat_id=chat_id, thread_id=thread_id, reply_to=reply_to)
                new.accepted_anchors = {reply_to}
                self._sessions[self._skey(chat_id, thread_id)] = new
                session = self._ensure_session(chat_id, thread_id)  # backfill the card task for the new session
                self._seal_session(old_session)
        if session.reply_to is None and reply_to:
            session.reply_to = reply_to
            session.accepted_anchors = {reply_to}
        # Topic session: the anchor has arrived (first frame carries it) → create now.
        self._maybe_start_card_task(session)
        if session.answer_seg is None:
            # Empty text goes through on_answer_delta: create the empty ANSWER segment in
            # the right position (after reasoning).
            session.segment_state.on_answer_delta("")
            session.answer_seg = session.segment_state.segments[-1]
        # Strip <thinking>/<thought> tag-shaped reasoning leakage (bare-text
        # fragments cannot be stripped here).
        session.answer_seg.text = strip_reasoning_tags(content)
        session.answer_seg.dirty = True
        self._schedule(session)

    def on_tool_start(self, tool_name: str, detail: str = "",
                      anchor: tuple[str, str | None] | None = None) -> None:
        """Sink for format_tool_event(ToolCallChunk) — record the step and refresh
        the tool panel.

        Tool events may precede any draft (the norm for tool-heavy turns): the
        card is created even without a session so the panel rolls from step one.
        ``anchor`` = (chat, thread) recorded on the adapter for the current turn
        (format_tool_event lands on the turn's own adapter, which knows whom it
        serves) — concurrent sessions route by anchor so chat B's tool names and
        arguments never render into chat A's card."""
        session = None
        if anchor is not None:
            session = self.active_session(anchor[0], anchor[1])
        if session is None:
            session = self._any_active_session()
        if session is None:
            # Tool event before the probe: capture the loop on this thread and create a
            # placeholder session (relocated once the probe arrives); drop the
            # event if no loop is reachable cross-thread.
            try:
                self._capture_loop()
            except RuntimeError:
                return
            session = self._ensure_session("")
        session.tool_tracker.record_start(tool_name, detail)
        if session.tool_seg is None:
            session.segment_state.on_tool_event(len(session.tool_tracker.build_display_steps()))
            session.tool_seg = session.segment_state.segments[-1]
            session.tool_seg.created = False  # element creation deferred to flush
        session.tool_seg.dirty = True
        self._schedule(session)

    def on_tool_end(self, tool_name: str, *, error: str = "", output: str = "") -> None:
        session = self._any_active_session()
        if session is None:
            return
        session.tool_tracker.record_end(tool_name, error=error, output=output)
        if session.tool_seg is not None:
            session.tool_seg.dirty = True
        self._schedule(session)

    def on_reasoning(self, chat_id: str, text: str, *, strict: bool = False) -> None:
        """Reasoning delta — dropped under concurrent sessions (anti-cross-talk).

        ``strict`` (session-id routed): only chat_id's own session, dropped when
        the target is inactive — never fall back to the single-active heuristic
        (falling back would write B's reasoning into A's card)."""
        target = self.active_session(chat_id) if strict else self._resolve_reasoning_target(chat_id)
        if target is None:
            return
        if not target._reasoning_logged:
            target._reasoning_logged = True
            logger.info("reasoning streaming into card: chat=%s",
                         target.chat_id[:12])
        target.segment_state.on_reasoning_delta(text)
        self._schedule(target)

    def on_heartbeat(self, chat_id: str, text: str, thread_id: str | None = None) -> None:
        """Heartbeat / interim text → the status line at the bottom of the card."""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return
        session.heartbeat_text = text[:_HEARTBEAT_MAX_LEN]
        self._schedule(session)

    # ── completion ──

    async def complete(self, chat_id: str, final_text: str, *, is_error: bool = False,
                       duration: float | None = None, tokens: dict[str, int] | None = None,
                       model: str = "", thread_id: str | None = None) -> str | None:
        """Turn terminal state: render the complete card (with footer), return its message id."""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return None
        logger.info("complete: chat=%s state=%s segs=%d redirected=%s",
                     chat_id[:12], session.state, len(session.segment_state.segments),
                     session.redirected)
        if session.card_create_task is not None:
            await session.card_create_task
        if session.flush is not None:
            session.flush.mark_completed()
        if session.state == "failed" or session.card_id is None:
            return session.card_msg_id  # card creation failed: let native text deliver
        if final_text and session.answer_seg is not None:
            session.answer_seg.text = final_text
            session.answer_seg.dirty = False  # the complete card re-renders wholesale
        elif final_text:
            # A short answer may be sent whole with no draft frame (reasoning-only
            # segments) — create the ANSWER segment from the final text or the body
            # is lost and the card renders its "Done." placeholder.
            session.segment_state.on_answer_delta("")
            session.answer_seg = session.segment_state.segments[-1]
            session.answer_seg.text = strip_reasoning_tags(final_text)
            session.answer_seg.dirty = False
        session.segment_state.finalize_segments(
            len(session.tool_tracker.build_display_steps()))
        session.state = "failed" if is_error else "completed"
        session.completed_at = time.time()
        usage = self._pop_usage(chat_id)
        if (usage and usage.get("first_started") and usage.get("last_ended")
                and usage["last_ended"] > usage["first_started"] and duration is None):
            # True turn duration = first API start → last API end (tool time included);
            # session lifetime only covers the streaming tail and inflates t/s 10x.
            duration = usage["last_ended"] - usage["first_started"]
        if usage:
            logger.info(
                "footer usage: chat=%s in=%d out=%d model=%s",
                chat_id[:12], usage.get("input", 0), usage.get("output", 0),
                usage.get("model") or model or "?")
        if tokens:
            usage = {"input": tokens.get("input_tokens", 0),
                     "output": tokens.get("output_tokens", 0), "model": model}
        session.model_switch = self._model_switch_data(
            (usage or {}).get("model") or model)
        card = build_complete_card(
            segments=session.segment_state.segments,
            all_tool_steps=session.tool_tracker.build_display_steps(),
            footer_data=self._footer_data(session, duration, usage, model),
            footer_fields=self._footer_fields,
            footer_show_label=self._footer_show_label,
            footer_enabled=self._footer_enabled,
            is_error=is_error,
            header_enabled=self._header_enabled,
            body_text_size=self._body_text_size,
            show_tool_use=self._show_tool_use,
            width_mode=self._width_mode,
            model_switch=session.model_switch,
            tool_panel_expanded=self._tool_panel_expanded,
            reasoning_panel_expanded=self._reasoning_panel_expanded,
        )
        try:
            session.sequence += 1
            await self._client.cardkit_close_streaming(session.card_id, sequence=session.sequence)
            session.sequence += 1
            await self._client.cardkit_update(session.card_id, card, sequence=session.sequence)
        except Exception as e:
            logger.warning("plugin complete card update failed: chat=%s err=%s", chat_id, e)
        return session.card_msg_id

    async def append_notice(self, chat_id: str, text: str,
                            thread_id: str | None = None) -> str | None:
        """Append a background/system notice to the chat's newest card (cross-turn
        merge).

        The session may already be COMPLETED (background turns often arrive
        after the main turn finished) — append a NOTICE segment and re-render
        the complete card; an active session defers to flush for element
        creation. Returns the card message id, or None when no card is
        available (the caller falls back to native text)."""
        session = self._sessions.get(self._skey(chat_id, thread_id))
        if session is None or session.card_id is None:
            return None
        session.segment_state.add_notice(text)
        if session.is_terminal:
            card = build_complete_card(
                segments=session.segment_state.segments,
                all_tool_steps=session.tool_tracker.build_display_steps(),
                footer_data=self._footer_data(session, None, None, ""),
                footer_fields=self._footer_fields,
                footer_show_label=self._footer_show_label,
                footer_enabled=self._footer_enabled,
                header_enabled=self._header_enabled,
                body_text_size=self._body_text_size,
                show_tool_use=self._show_tool_use,
                width_mode=self._width_mode,
                model_switch=getattr(session, "model_switch", None),
                tool_panel_expanded=self._tool_panel_expanded,
                reasoning_panel_expanded=self._reasoning_panel_expanded,
            )
            session.sequence += 1
            try:
                await self._client.cardkit_update(session.card_id, card, sequence=session.sequence)
            except Exception as e:
                logger.warning("append notice update failed: %s", e)
                return None
        else:
            self._schedule(session)
        logger.info("notice merged into card: chat=%s len=%d",
                     chat_id[:12], len(text))
        return session.card_msg_id

    async def send_cron_card(self, chat_id: str, content: str, *, task_name: str = "",
                             job_id: str = "", run_time: str = "",
                             template: str = "blue") -> str | None:
        """One-shot cron result card (a static card sent flat, no streaming session).

        A side-channel send like :meth:`append_notice`: no session state, no
        follow-up edits; failures return None and the caller falls back to
        native text. Failure notices render template="red" for instant
        recognition."""
        card = build_cron_card(content, task_name=task_name, run_time=run_time,
                               template=template)
        try:
            card_id = await self._client.cardkit_create(card)
            # Cron delivery carries no reply anchor: send flat to the chat.
            msg_id: str | None = await self._client.send_card_to_chat(
                chat_id, {"type": "card", "data": {"card_id": card_id}})
            return msg_id
        except Exception as e:
            logger.warning("cron card send failed: chat=%s err=%s",
                            chat_id[:12], e)
            return None

    async def abandon(self, chat_id: str, thread_id: str | None = None) -> None:
        """Stream abandoned (interrupt/error) — close as an error rather than leave a spinner."""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return
        await self.complete(chat_id, session.answer_seg.text if session.answer_seg else "",
                            is_error=True)

    # ── internals ──

    def _capture_loop(self) -> None:
        """Capture the engine event loop (first async entry; used for cross-thread scheduling)."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._thread = threading.current_thread()

    def _ensure_session(self, chat_id: str, thread_id: str | None = None) -> ChatSession:
        key = self._skey(chat_id, thread_id)
        session = self._sessions.get(key)
        if session is None or session.is_terminal:
            # The old (terminal) card stays in the chat history; the new turn opens a new session.
            if session is not None:
                logger.info(
                    "session replaced: chat=%s old_state=%s "
                    "completed_ago=%s redirected=%s",
                    chat_id[:12], session.state,
                    f"{time.time() - session.completed_at:.1f}s"
                    if session.completed_at else "never",
                    session.redirected)
            new = ChatSession(chat_id=chat_id, thread_id=thread_id)
            self._sessions[key] = new
            session = new
        self._maybe_start_card_task(session)
        return session

    def _maybe_start_card_task(self, session: ChatSession) -> None:
        """Start the card-creation task; topic sessions are the exception — creation
        is deferred to the first draft (once the reply anchor is known).

        An anchorless card lands at the top of the main chat; a topic turn's
        card must reply to a message inside the topic to join the thread
        (observed: probe-time anchorless creation left the card in the main
        chat and the topic an empty shell)."""
        if session.state != "creating" or session.card_create_task is not None:
            return
        if session.thread_id and session.reply_to is None:
            return  # topic session waits for its anchor
        assert self._loop is not None
        if session.flush is None:
            session.flush = FlushController(loop=self._loop)
        session.card_create_task = self._loop.create_task(self._do_create_card(session))

    def _seal_session(self, session: ChatSession, notice: str | None = None) -> None:
        """Seal an old session (followup boundary): render its complete card from whatever accumulated; failures only log."""
        assert self._loop is not None

        async def _seal() -> None:
            if session.card_create_task is not None:
                await session.card_create_task
            if session.state == "failed" or session.card_id is None:
                return
            session.completed_at = time.time()
            if notice:
                session.segment_state.add_notice(notice)
            session.segment_state.finalize_segments(
                len(session.tool_tracker.build_display_steps()))
            session.state = "completed"
            card = build_complete_card(
                segments=session.segment_state.segments,
                all_tool_steps=session.tool_tracker.build_display_steps(),
                footer_data=self._footer_data(session, None, None, ""),
                footer_fields=self._footer_fields,
                footer_show_label=self._footer_show_label,
                footer_enabled=self._footer_enabled,
                # Interrupted cards close red (user-facing decision): the redirected old card
                # is recognizable at a glance and the NOTICE explains where the result
                # went. The red mark lives in the header, force-shown for abnormal
                # states regardless of the header config default.
                is_aborted=bool(notice),
                header_enabled=self._header_enabled or bool(notice),
                body_text_size=self._body_text_size,
                show_tool_use=self._show_tool_use,
                width_mode=self._width_mode,
                tool_panel_expanded=self._tool_panel_expanded,
                reasoning_panel_expanded=self._reasoning_panel_expanded,
            )
            try:
                session.sequence += 1
                await self._client.cardkit_close_streaming(session.card_id, sequence=session.sequence)
                session.sequence += 1
                await self._client.cardkit_update(session.card_id, card, sequence=session.sequence)
            except Exception as e:
                logger.warning("seal old card failed: %s", e)

        self._loop.create_task(_seal())

    def _any_active_session(self) -> ChatSession | None:
        active = [s for s in self._sessions.values() if not s.is_terminal]
        return active[0] if active else None

    def _resolve_reasoning_target(self, chat_id: str) -> ChatSession | None:
        # Explicit chat wins when present; otherwise only a single session may claim
        # the delta (concurrency drops it, preventing cross-talk). The fallback
        # must include creating sessions — reasoning lost during the card-creation
        # window leaves the card spinning forever.
        session = self.active_session(chat_id)
        if session is not None:
            return session
        active = [s for s in self._sessions.values() if not s.is_terminal]
        return active[0] if len(active) == 1 else None

    def _schedule(self, session: ChatSession) -> None:
        if session.state == "creating" or session.flush is None or self._loop is None:
            return  # the first flush after card creation carries everything staged so far
        callback = lambda s=session: self._do_flush(s)  # noqa: E731
        if threading.current_thread() is self._thread:
            session.flush.schedule_update(callback)
        else:
            self._loop.call_soon_threadsafe(session.flush.schedule_update, callback)

    async def _do_create_card(self, session: ChatSession) -> None:
        card = build_streaming_card_v2(
            show_tool_use=False,
            show_reasoning=False,
            show_streaming_element=False,
            header_enabled=self._header_enabled,
            text_size=self._body_text_size,
            heartbeat_enabled=True,
            width_mode=self._width_mode,
        )
        try:
            card_id = await self._client.cardkit_create(card)
            if session.reply_to:
                # Anchored: the card lands under the user's message (inside the topic thread).
                card_msg_id = await self._client.reply_card_by_id(session.reply_to, card_id)
            else:
                # Anchorless: send flat (a chat_id is not a valid reply target; 230001).
                card_msg_id = await self._client.send_card_to_chat(
                    session.chat_id, {"type": "card", "data": {"card_id": card_id}})
        except Exception as e:
            logger.warning("plugin card create failed: chat=%s err=%s", session.chat_id, e)
            session.state = "failed"
            return
        session.card_id = card_id
        session.card_msg_id = card_msg_id
        if session.state == "creating":
            # The session may have been terminated synchronously by a redirect boundary
            # while the card was being created: never revive it to streaming or the
            # global reasoning routing would drop the new turn's thinking as noise.
            session.state = "streaming"
        if session.flush is not None:
            session.flush.set_card_message_ready(True)
        self._schedule(session)

    async def _do_flush(self, session: ChatSession) -> None:
        """Idempotent flush: create elements for new segments + stream dirty text + refresh panels/heartbeat."""
        if session.is_terminal or not session.card_id:
            return
        segments = session.segment_state.segments
        all_steps = session.tool_tracker.build_display_steps()
        actions: list[dict[str, Any]] = []
        for seg in segments:
            if not seg.created:
                actions.append(build_add_segment_action(
                    seg, all_steps, text_size=self._body_text_size))
                seg.created = True
            elif seg.type == SegmentType.TOOL and seg.dirty:
                # Update with the segment's own [offset, end) slice: passing the full list
                # makes a second tool panel re-render earlier segments' steps (a
                # single-panel-era leftover) and diverges from the complete card.
                seg_steps = all_steps[seg.tool_offset:tool_segment_end(seg, all_steps)]
                actions.append(build_tool_update_action(
                    element_id=seg.el_id, steps=seg_steps, step_offset=seg.tool_offset))
                seg.dirty = False
            elif (seg.type == SegmentType.REASONING and seg.elapsed_ms > 0
                  and not seg.reasoning_finalized):
                actions.append(build_reasoning_finalized_action(seg))
                seg.reasoning_finalized = True
        if actions:
            session.sequence += 1
            try:
                await self._client.cardkit_batch_update(
                    session.card_id, actions, sequence=session.sequence)
            except Exception as e:
                logger.warning("plugin batch update failed: chat=%s err=%s",
                                session.chat_id, e)
                return
        # Stream dirty text (answer / reasoning).
        for seg in segments:
            if not seg.created or not seg.dirty:
                continue
            # The reasoning panel streams the same excerpt the complete card renders
            # (cap_reasoning_text) — a full thinking stream would push the card
            # past Feishu's size limit; the answer body is the deliverable, never cut.
            raw = (cap_reasoning_text(seg.text)
                   if seg.type == SegmentType.REASONING else seg.text)
            content = optimize_markdown_style(raw) or " "
            session.sequence += 1
            try:
                await self._client.cardkit_stream_element(
                    session.card_id, seg.text_el_id or seg.el_id, content,
                    sequence=session.sequence)
                seg.dirty = False
            except Exception as e:
                logger.debug("plugin stream element failed: el=%s err=%s", seg.el_id, e)
        # Heartbeat status line.
        if session.heartbeat_text:
            session.sequence += 1
            try:
                await self._client.cardkit_stream_element(
                    session.card_id, HEARTBEAT_ELEMENT_ID,
                    optimize_markdown_style(session.heartbeat_text) or " ",
                    sequence=session.sequence)
                session.heartbeat_text = ""
            except Exception as e:
                logger.debug("plugin heartbeat update failed: %s", e)

    def _footer_data(self, session: ChatSession, duration: float | None,
                     usage: dict[str, Any] | None, model: str) -> dict[str, Any] | None:
        data: dict[str, Any] = {
            "duration": duration if duration is not None else time.time() - session.created_at,
        }
        if usage:
            data["input_tokens"] = usage.get("input", 0)
            data["output_tokens"] = usage.get("output", 0)
            if model:
                data["model"] = model
            elif usage.get("model"):
                data["model"] = usage["model"]
            # t/s = output tokens / turn duration.
            if data["output_tokens"] and data["duration"] > 0:
                data["tps"] = data["output_tokens"] / data["duration"]
            if usage.get("context_max"):
                data["context_max"] = usage["context_max"]
                data["context_used"] = usage.get("context_used", usage.get("input", 0))
        elif model:
            data["model"] = model
        return data

    def _model_switch_data(self, current: str) -> dict[str, str] | None:
        """Footer switch-button data: {"current"}; no button when the rotation list is <2 or the model is unknown."""
        current = (current or "").strip()
        if len(self._model_cycle) < 2 or not current:
            return None
        return {"current": current}

    def model_picker_data(self, chat_id: str) -> dict[str, Any] | None:
        """Picker-card data after a switch click: {"current", "models"}; None without a list."""
        if len(self._model_cycle) < 2:
            return None
        current = self._chat_models.get(chat_id) or self._chat_models.get("") \
            or self._last_model or self._model_cycle[0]
        return {"current": current, "models": list(self._model_cycle)}

    # ── Favorite models (picker ⚙ entry → management-card toggles, persisted across restarts) ──

    _CANDIDATES_TTL = 60.0

    def _model_cycle_file(self) -> Path | None:
        """Favorites store: <HERMES_HOME>/feishu_streaming_model_cycle.json.

        Sits next to config.yaml (profile-scoped by construction; deploying the
        plugin directory never overwrites it)."""
        try:
            from hermes_constants import get_hermes_home

            return get_hermes_home() / "feishu_streaming_model_cycle.json"
        except Exception:
            return None

    def _load_model_favorites(self) -> list[str]:
        path = self._model_cycle_file()
        if path is None or not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            models = raw.get("models") if isinstance(raw, dict) else raw
            out: list[str] = []
            for m in models or []:
                name = str(m).strip()
                if name and name not in out:
                    out.append(name)
            return out[:MAX_FAVORITE_MODELS]
        except Exception as e:
            logger.warning("model favorites load failed: %s", e)
            return []

    def _apply_model_favorites(self, models: list[str]) -> list[str]:
        """Persist the list and apply it to this engine (footer button and picker update immediately); empty falls back to derivation."""
        out: list[str] = []
        for m in models:
            name = str(m).strip()
            if name and name not in out:
                out.append(name)
        out = out[:MAX_FAVORITE_MODELS]
        self._model_cycle = out or list(self._model_cycle_fallback)
        path = self._model_cycle_file()
        if path is not None:
            try:
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(json.dumps({"models": out}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                tmp.replace(path)
            except Exception as e:
                logger.warning("model favorites save failed: %s", e)
        return out

    def toggle_model_favorite(self, target: str) -> tuple[list[str], str]:
        """Toggle one favorite model → (new list, notice). Adding past the cap is refused."""
        target = (target or "").strip()
        if not target:
            return self._load_model_favorites(), ""
        favorites = self._load_model_favorites()
        if target in favorites:
            favorites.remove(target)
        elif len(favorites) >= MAX_FAVORITE_MODELS:
            return favorites, f"Favorites are full ({MAX_FAVORITE_MODELS}); remove one before adding {target}"
        else:
            favorites.append(target)
        return self._apply_model_favorites(favorites), ""

    async def model_candidates(self) -> list[dict[str, Any]]:
        """Every model Hermes can serve, grouped by provider — the management card's
        candidate pool.

        Source: the same list_picker_providers the /model picker uses (built-in
        provider catalog + custom endpoints, non-blocking disk-cached reads).
        Sync IO goes through a worker thread; 60s cache. Any failure returns []
        (the card renders an empty-state hint; nothing else is affected)."""
        now = time.time()
        if self._candidates_cache and now - self._candidates_cache[0] < self._CANDIDATES_TTL:
            return self._candidates_cache[1]

        def _fetch() -> list[dict[str, Any]]:
            import yaml

            from hermes_constants import get_hermes_home

            cfg = yaml.safe_load((get_hermes_home() / "config.yaml").read_text(encoding="utf-8")) or {}
            model_cfg = cfg.get("model") or {}
            try:
                from hermes_cli.config import get_compatible_custom_providers

                custom = get_compatible_custom_providers(cfg)
            except Exception:
                custom = cfg.get("custom_providers")
            excluded = (cfg.get("model_catalog") or {}).get("excluded_providers")
            from hermes_cli.model_switch_providers import list_picker_providers

            providers: list[dict[str, Any]] = list(list_picker_providers(
                current_provider=str(model_cfg.get("provider") or "openrouter"),
                current_base_url=str(model_cfg.get("base_url") or ""),
                current_model=self._last_model or str(model_cfg.get("default") or ""),
                user_providers=cfg.get("providers"),
                custom_providers=custom,
                excluded_providers=excluded if isinstance(excluded, list) else [],
                non_blocking_catalogs=True, probe_custom_providers=False,
                probe_current_custom_provider=False, max_models=50, include_moa=False))
            return providers

        try:
            providers = await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.warning("model candidates fetch failed: %s", e)
            return []
        slim: list[dict[str, Any]] = [
            {"slug": p.get("slug"), "name": p.get("name"),
             "models": [str(m) for m in (p.get("models") or []) if str(m).strip()],
             "total_models": int(p.get("total_models") or 0)}
            for p in providers or [] if p.get("models")]
        for item in slim:
            item["total_models"] = item["total_models"] or len(item["models"])
        self._candidates_cache = (now, slim)
        logger.info("model candidates loaded: providers=%d models=%d",
                     len(slim), sum(len(p["models"]) for p in slim))
        return slim
