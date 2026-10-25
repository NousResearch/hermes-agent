"""Carry finished background results into the running turn (``busy_result_mode: steer``).

A surface that owns a finished subagent or background-process result while its agent is mid-turn
offers it here instead of queueing a whole new turn. The result rides the newest tool result just before that result is
persisted, the same mutable slot the iteration-budget warning uses: it stays tool-channel evidence
(never a user-authority steer row), past context is never rewritten, and the prompt cache holds.
Offers no tool batch picked up are handed back by ``take_outcome`` so the surface can deliver them
the ordinary way, as the next turn.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, NamedTuple, Optional

_STATE_ATTR = "_background_result_carrier"
_STATE_LOCK = threading.Lock()

BACKGROUND_RESULT_MARKER = (
    "[BACKGROUND RESULT — a subagent you delegated finished while you were working. "
    "This is tool-channel evidence for the current task, not a new user request.]"
)
PROCESS_RESULT_MARKER = (
    "[BACKGROUND RESULT — a background process you started finished while you were working. "
    "This is tool-channel evidence for the current task, not a new user request.]"
)


class _Offer(NamedTuple):
    key: Any
    text: str
    marker: str
    # True once the agent already read the result another way (process wait/log): the offer then
    # settles as delivered instead of showing the same output twice.
    already_read: Optional[Callable[[], bool]]

    def read(self) -> bool:
        return self.already_read is not None and bool(self.already_read())


class _CarrierState:
    __slots__ = ("delivered", "lock", "pending", "staged")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pending: list[_Offer] = []
        self.delivered: list[Any] = []
        # Attached to a tool result but not yet persisted: acknowledged only once the caller
        # confirms the flush that wrote that result.
        self.staged: list[_Offer] = []


def _state(agent: Any) -> _CarrierState:
    with _STATE_LOCK:
        state = vars(agent).get(_STATE_ATTR)
        if state is None:
            state = _CarrierState()
            setattr(agent, _STATE_ATTR, state)
        return state


def offer(agent: Any, key: Any, text: str, *, marker: str = BACKGROUND_RESULT_MARKER,
          already_read: Optional[Callable[[], bool]] = None) -> None:
    """Queue ``text`` (identified by ``key``) for the next tool result this agent persists."""
    state = _state(agent)
    with state.lock:
        state.pending.append(_Offer(key, text, marker, already_read))


def withdraw(agent: Any, key: Any) -> None:
    """Drop a pending offer whose delivery the surface abandoned (e.g. cancelled at shutdown)."""
    state = vars(agent).get(_STATE_ATTR)
    if state is not None:
        with state.lock:
            state.pending = [item for item in state.pending if item.key is not key]


def attach_to_tool_tail(agent: Any, messages: list) -> list[_Offer]:
    """Append every pending offer to the newest tool result while it is still unpersisted.

    Returns the offers it attached (empty when nothing was attached). They are STAGED, not
    acknowledged: a result counts as delivered only when ``confirm_attached`` sees the flush that
    persisted this tool row succeed, so a settlement tick can never complete a delivery that only
    ever existed in memory. Called right before the tool-progress flush, so the appended text is
    exactly what both the transcript and the next model call see."""
    state = vars(agent).get(_STATE_ATTR)
    if state is None:
        return []
    from agent.context_compressor import _DB_PERSISTED_MARKER
    if (not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "tool"
            or messages[-1].get(_DB_PERSISTED_MARKER)):
        return []
    message = messages[-1]
    content = message.get("content")
    if not (content is None or isinstance(content, (str, list))):
        return []
    with state.lock:
        offers, state.pending = state.pending, []
        read = [item.read() for item in offers]
        state.delivered.extend(item.key for item, was_read in zip(offers, read) if was_read)
        attached = [item for item, was_read in zip(offers, read) if not was_read]
        if not attached:
            return []
        block = "\n\n".join(f"{item.marker}\n{item.text}" for item in attached)
        if isinstance(content, str):
            message["content"] = f"{content}\n\n{block}"
        else:
            message["content"] = [*(content or []), {"type": "text", "text": block}]
        state.staged.extend(attached)
    return attached


def confirm_attached(agent: Any, attached: list[_Offer], persisted: bool) -> None:
    """Record the tool-progress flush outcome for offers ``attach_to_tool_tail`` staged.

    A persisted flush acknowledges them; a failed one keeps them staged, so ``take_outcome`` hands
    them back at turn end and the result arrives again instead of vanishing behind a settled claim."""
    if not attached:
        return
    state = vars(agent).get(_STATE_ATTR)
    if state is None:
        return
    ids = {id(item) for item in attached}
    with state.lock:
        if persisted:
            state.staged = [item for item in state.staged if id(item) not in ids]
            state.delivered.extend(item.key for item in attached)


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
    offer the surface has already taken back. Offers attached to a tool row whose flush never
    persisted are still here: handing them back duplicates a result that survives, which beats
    losing one behind a settled claim."""
    state = vars(agent).get(_STATE_ATTR)
    if state is None:
        return [], []
    with state.lock:
        delivered, state.delivered = state.delivered, []
        pending, state.pending = state.pending, []
        staged, state.staged = state.staged, []
    read = [item.read() for item in pending]
    delivered += [item.key for item, was_read in zip(pending, read) if was_read]
    undelivered = [*staged, *(item for item, was_read in zip(pending, read) if not was_read)]
    return delivered, [(item.key, item.text) for item in undelivered]
