"""Carry finished background results into the running turn (``delegation.busy_result_mode: steer``).

A surface that owns a finished subagent result while its agent is mid-turn offers it here instead of
queueing a whole new turn. The result rides the newest tool result just before that result is
persisted, the same mutable slot the iteration-budget warning uses: it stays tool-channel evidence
(never a user-authority steer row), past context is never rewritten, and the prompt cache holds.
Offers no tool batch picked up are handed back by ``take_outcome`` so the surface can deliver them
the ordinary way, as the next turn.
"""

from __future__ import annotations

import threading
from typing import Any

_STATE_ATTR = "_background_result_carrier"
_STATE_LOCK = threading.Lock()

BACKGROUND_RESULT_MARKER = (
    "[BACKGROUND RESULT — a subagent you delegated finished while you were working. "
    "This is tool-channel evidence for the current task, not a new user request.]"
)


class _CarrierState:
    __slots__ = ("delivered", "lock", "pending")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pending: list[tuple[Any, str]] = []
        self.delivered: list[Any] = []


def _state(agent: Any) -> _CarrierState:
    with _STATE_LOCK:
        state = vars(agent).get(_STATE_ATTR)
        if state is None:
            state = _CarrierState()
            setattr(agent, _STATE_ATTR, state)
        return state


def offer(agent: Any, key: Any, text: str) -> None:
    """Queue ``text`` (identified by ``key``) for the next tool result this agent persists."""
    state = _state(agent)
    with state.lock:
        state.pending.append((key, text))


def attach_to_tool_tail(agent: Any, messages: list) -> int:
    """Append every pending offer to the newest tool result while it is still unpersisted.

    Returns how many offers were attached. Called right before the tool-progress flush, so the
    appended text is exactly what both the transcript and the next model call see."""
    state = vars(agent).get(_STATE_ATTR)
    if state is None:
        return 0
    from agent.context_compressor import _DB_PERSISTED_MARKER
    if (not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "tool"
            or messages[-1].get(_DB_PERSISTED_MARKER)):
        return 0
    message = messages[-1]
    content = message.get("content")
    if not (content is None or isinstance(content, (str, list))):
        return 0
    with state.lock:
        offers, state.pending = state.pending, []
        if not offers:
            return 0
        block = "\n\n".join(f"{BACKGROUND_RESULT_MARKER}\n{text}" for _key, text in offers)
        if isinstance(content, str):
            message["content"] = f"{content}\n\n{block}"
        else:
            message["content"] = [*(content or []), {"type": "text", "text": block}]
        state.delivered.extend(key for key, _text in offers)
    return len(offers)


def take_delivered(agent: Any) -> list[Any]:
    """Return and clear the keys attached since the last call, leaving pending offers in place."""
    state = vars(agent).get(_STATE_ATTR)
    if state is None:
        return []
    with state.lock:
        delivered, state.delivered = state.delivered, []
    return delivered


def take_outcome(agent: Any) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Return and clear ``(delivered_keys, undelivered_offers)``.

    Clearing under the lock means a worker thread that outlives its turn can no longer attach an
    offer the surface has already taken back."""
    state = vars(agent).get(_STATE_ATTR)
    if state is None:
        return [], []
    with state.lock:
        delivered, state.delivered = state.delivered, []
        undelivered, state.pending = state.pending, []
    return delivered, undelivered
