"""The current turn's identity and working directory, as seen by LitKit tools.

The turn runner (:mod:`litco.hermes_runner`) binds a :class:`TurnIdentity` for the
duration of each turn. Hermes copies context variables into the threads that run tool
calls and delegated subagents, so every LitKit call made during the turn sees the same
acting user. Outside a turn (cron, unattended work) nothing is bound: calls carry no
user assertion and run under the Matter Agent user's own viewer role.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class TurnIdentity:
    turn_id: Optional[str] = None
    matter_id: Optional[str] = None
    # Set only when the turn's user assertion verified; the tools assert this user to LitKit.
    acting_user: Optional[str] = None
    cwd: Optional[Path] = None


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


@contextlib.contextmanager
def turn_scope(identity: TurnIdentity) -> Iterator[TurnIdentity]:
    token = bind_turn(identity)
    try:
        yield identity
    finally:
        reset_turn(token)
