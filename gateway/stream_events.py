"""Structured streaming events — the agent→gateway delivery contract.

Typed *what happened* events (frozen dataclasses, no I/O) emitted from the agent's
worker thread; ``GatewayStreamConsumer`` is the sink and the adapter decides rendering.
Events describe *transport*, never *context*: whatever the gateway "eats" must never
diverge from the agent-owned message history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class MessageChunk:
    """A delta of streamed assistant text (think-block content is filtered upstream)."""
    text: str


@dataclass(frozen=True)
class MessageStop:
    """Assistant text segment complete.  ``final`` only for the turn's terminal stop; an
    intermediate stop (text → tool → text) starts a fresh segment below tool chrome."""
    final: bool = False


@dataclass(frozen=True)
class Commentary:
    """A complete interim assistant message between tool iterations (not a delta)."""
    text: str


@dataclass(frozen=True)
class ToolCallChunk:
    """A tool invocation started. Raw facts only; the adapter decides presentation."""
    tool_name: str
    preview: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    index: int = 0  # monotonic per-turn index: correlates a finish with its start


@dataclass(frozen=True)
class ToolCallFinished:
    """A tool invocation completed (drives bubble settling + LongToolHint).  Tool *output*
    never travels here — it is history."""
    tool_name: str
    duration: float = 0.0  # wall-clock seconds
    ok: bool = True        # returned without raising
    index: int = 0


@dataclass(frozen=True)
class LongToolHint:
    """One-shot onboarding nudge for a long tool run; the gateway gates it on platform
    capability (/verbose usable) and first-time use."""
    tool_name: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class GatewayNotice:
    """Gateway-originated control message; ``kind`` is a stable string adapters switch on
    (``"restart"`` / ``"online"`` / ``"long_run"`` / …), ``text`` the default rendering."""
    kind: str
    text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# Explicit union (not a marker base class) so a missing ``case`` in an
# exhaustive match is a visible type error rather than a silent fall-through.
StreamEvent = Union[
    MessageChunk, MessageStop, Commentary,
    ToolCallChunk, ToolCallFinished, LongToolHint, GatewayNotice,
]

__all__ = [
    "MessageChunk", "MessageStop", "Commentary", "ToolCallChunk",
    "ToolCallFinished", "LongToolHint", "GatewayNotice", "StreamEvent",
]
