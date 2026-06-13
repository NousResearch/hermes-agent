"""Structured streaming events — the agent→gateway delivery contract.

Historically the agent drove gateway delivery through a fan of loosely-typed
callbacks (``stream_delta_callback(text)``, ``tool_progress_callback(event_type,
tool_name, preview, args)``, ``interim_assistant_callback(text)`` …) and each
gateway callback decided *both* what to render and how to send it.  That
coupling is why tool-progress bubbles and the streaming draft raced each other
on Telegram, and why tool-call formatting lived agent-side even though only the
gateway knows what a given platform can render.

This module defines a small, typed event vocabulary that names *what happened*
without prescribing *how it is delivered*.  The gateway's stream consumer
(``GatewayStreamConsumer``) is the single sink; the platform adapter decides how
to render each event (Telegram can stream a MarkdownV2 ```bash``` block as a
native draft; iMessage has no rich formatting and may collapse or drop tool
chrome).  Separation of concerns: smart agent emits structured data, smart
gateway decides delivery.

These are intentionally plain frozen dataclasses — no behavior, no platform
knowledge, no I/O.  They are cheap to construct on the agent's worker thread and
safe to hand across the thread/async boundary into the consumer queue.

Design constraints (see hermes-agent-dev skill — message-flow + cache
invariants):
  * Events describe *transport*, never *context*.  Nothing here is persisted to
    conversation history; what the gateway chooses to "eat" (e.g. tool chrome on
    a platform that can't render it) must never diverge from the bytes stored in
    the agent's message history.  History is owned by the agent; these events are
    a presentation-layer stream only.
  * Backward compatible by construction.  The gateway adapts its existing
    callbacks into these events at the boundary; adapters that don't opt into
    event-native rendering get identical behavior via the base-class default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


# ── Message (assistant text) events ──────────────────────────────────────────

@dataclass(frozen=True)
class MessageChunk:
    """A delta of streamed assistant text.

    ``text`` is the incremental content as it arrives from the model.  The
    consumer accumulates chunks and progressively renders them (native draft on
    Telegram DMs, edit-in-place elsewhere).  Reasoning/think-block content is
    filtered upstream and never arrives as a MessageChunk.
    """
    text: str


@dataclass(frozen=True)
class MessageStop:
    """The current assistant message segment is complete.

    Emitted when a contiguous run of assistant text ends — either the whole
    response finished, or a tool boundary interrupts the text so the next
    segment should render as a fresh message *below* any tool chrome.

    ``final`` is True only for the terminal stop of the whole turn; an
    intermediate stop (text → tool call → more text) carries ``final=False`` so
    the consumer finalizes the current bubble and prepares a new segment without
    treating the turn as done.
    """
    final: bool = False


@dataclass(frozen=True)
class Commentary:
    """A complete interim assistant message emitted between tool iterations.

    Example: the model says "I'll inspect the repo first." before issuing a tool
    call.  Unlike a MessageChunk this is already-complete text (not a delta); the
    consumer renders it as its own message so it reads as a distinct beat.
    """
    text: str


# ── Tool-call events ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCallChunk:
    """A tool invocation has started (or its in-progress state changed).

    Carries the raw facts about the call — name, a short argument ``preview``,
    and the full ``args`` dict — and lets the *gateway* decide presentation
    (emoji, truncation, verbose vs compact, or eat it entirely on platforms that
    don't show tool chrome).  Previously the agent's gateway callback baked the
    emoji + preview formatting in; that decision now belongs to the adapter.
    """
    tool_name: str
    preview: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    # Monotonic per-turn index, so the consumer can correlate a finish with its
    # start and so "new"-mode dedup (only report when the tool changes) works
    # without the consumer tracking call order itself.
    index: int = 0


@dataclass(frozen=True)
class ToolCallFinished:
    """A tool invocation completed.

    ``duration`` is wall-clock seconds.  ``ok`` reflects whether the tool
    returned without raising.  The gateway uses this to clear/settle a progress
    bubble and to drive one-time onboarding hints (e.g. suggest /verbose after a
    long tool run).  No tool *output* travels here — output is the agent's
    concern and is persisted to history, not streamed as presentation.
    """
    tool_name: str
    duration: float = 0.0
    ok: bool = True
    index: int = 0


# ── Gateway control / lifecycle events ───────────────────────────────────────

@dataclass(frozen=True)
class LongToolHint:
    """One-shot onboarding nudge when a tool runs longer than the threshold.

    The gateway gates this on platform capability (the /verbose command must be
    usable) and on the user not having seen the hint before.  Modeled as an
    event so the *gateway* owns the "should I surface this here?" decision rather
    than the agent.
    """
    tool_name: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class GatewayNotice:
    """A gateway-originated control message (restart, online, long-run notice).

    ``kind`` is a stable string the adapter can switch on
    (``"restart"`` / ``"online"`` / ``"long_run"`` / …).  ``text`` is the
    human-readable default the base class renders when an adapter has no
    platform-specific treatment.
    """
    kind: str
    text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# Union of every event the consumer's dispatcher accepts.  Kept explicit (rather
# than a marker base class) so a missing ``case`` in an exhaustive match is a
# visible type error rather than a silent fall-through.
StreamEvent = Union[
    MessageChunk,
    MessageStop,
    Commentary,
    ToolCallChunk,
    ToolCallFinished,
    LongToolHint,
    GatewayNotice,
]


__all__ = [
    "MessageChunk",
    "MessageStop",
    "Commentary",
    "ToolCallChunk",
    "ToolCallFinished",
    "LongToolHint",
    "GatewayNotice",
    "StreamEvent",
]
