"""The current turn's identity and working directory, as seen by LitKit tools.

The turn runner (:mod:`litco.hermes_runner`) binds a :class:`TurnIdentity` for the
duration of each turn. Hermes copies context variables into the threads that run tool
calls and delegated subagents, so every LitKit call made during the turn sees the same
acting user. Outside a turn (cron, unattended work) nothing is bound: calls carry no
user assertion and run under the Matter Agent user's own viewer role.

A turn in a lawyer's own private thread may also carry the app-minted turn grant
(FIRM_AGENT_HOST 6.3), which only ``litkit_cross_matter_search`` sends back.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class TurnIdentity:
    turn_id: Optional[str] = None
    matter_id: Optional[str] = None
    # Set only when the turn's user assertion verified; the tools assert this user to LitKit.
    acting_user: Optional[str] = None
    cwd: Optional[Path] = None
    # The slug of the LitKit matter channel the turn arrived in (``litkitChannel.slug``), if any.
    litkit_channel: Optional[str] = None
    # The turn's ``sessionId``: the LitKit thread the turn runs in (a review proposal posts its card there).
    thread_id: Optional[str] = None
    # The app's grant for cross-matter search; kept out of reprs and logs.
    turn_grant: Optional[str] = field(default=None, repr=False)


_TURN: ContextVar[Optional[TurnIdentity]] = ContextVar("LITCO_TURN_IDENTITY", default=None)


def bind_turn(identity: TurnIdentity) -> Token:
    return _TURN.set(identity)


def reset_turn(token: Token) -> None:
    try:
        _TURN.reset(token)
    except ValueError:  # reset from a different context (thread handoff); clear instead
        _TURN.set(None)


def current_turn() -> Optional[TurnIdentity]:
    return _TURN.get()


def current_acting_user() -> Optional[str]:
    turn = _TURN.get()
    return turn.acting_user if turn is not None else None


def current_thread_id() -> Optional[str]:
    turn = _TURN.get()
    return turn.thread_id if turn is not None else None


def current_turn_grant() -> Optional[str]:
    turn = _TURN.get()
    return turn.turn_grant if turn is not None else None


@contextlib.contextmanager
def turn_scope(identity: TurnIdentity) -> Iterator[TurnIdentity]:
    token = bind_turn(identity)
    try:
        yield identity
    finally:
        reset_turn(token)
