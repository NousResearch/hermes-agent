"""Local engine segment identity boundary, independent of gateway registry locks.

Guarded readers must not wait here while holding locks needed by engine callbacks.
Gateway activation should try the guard and refuse a settling transition instead.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Callable
import os
import threading
from functools import wraps


class SessionIdentityMixin:
    """Serialize segment assignments, retaining a monotonic revision across ABA changes.

    Ordinary attribute reads retain their historical behavior. A reader certifying
    identity must hold session_identity_guard across comparison and its action.
    """

    def session_identity_guard(self):
        # First assignment happens during single-threaded agent construction.
        lock = self.__dict__.get("_session_identity_lock")
        if lock is None:
            lock = self.__dict__["_session_identity_lock"] = threading.RLock()
        return lock

    @property
    def session_id(self):
        return self.__dict__.get("session_id")

    @session_id.setter
    def session_id(self, value):
        with self.session_identity_guard():
            if self.__dict__.get("session_id") != value:
                self.__dict__["_session_identity_revision"] = self.session_identity_revision + 1
            self.__dict__["session_id"] = value

    @property
    def session_identity_revision(self) -> int:
        return self.__dict__.get("_session_identity_revision", 0)


def identity_transaction(fn):
    """Guard DB publication/adoption as well as the eventual pointer assignment.

    Non-engine duck-typed callers keep their existing behavior. They cannot supply
    proof of this boundary to a conditional activation consumer.
    """
    @wraps(fn)
    def guarded(agent, *args, **kwargs):
        guard = agent.session_identity_guard() if isinstance(agent, SessionIdentityMixin) else contextlib.nullcontext()
        with guard:
            return fn(agent, *args, **kwargs)
    return guarded


def _rebind_session_context(session_id: str) -> None:
    """Point the worker thread's session ContextVar and log context at ``session_id``."""
    try:
        from gateway.session_context import set_current_session_id
        set_current_session_id(session_id)
    except Exception:
        os.environ["HERMES_SESSION_ID"] = session_id
    with contextlib.suppress(Exception):
        from hermes_logging import set_session_context
        set_session_context(session_id)


# The hosting adapter carries an immutable admission witness across its worker
# handoff and a potentially long durable lease wait. It is not a global engine pin:
# compression within a turn already admitted against this identity remains legal.

@dataclass(frozen=True)
class ExpectedTurnIdentity:
    engine: object
    database: object
    revision: int
    session_id: str
    runtime_is_current: Callable[[], bool]


_expected_turn_identity = contextvars.ContextVar("expected_turn_identity", default=None)


@contextlib.contextmanager
def expected_turn_identity(engine, database, revision: int, session_id: str, runtime_is_current):
    token = _expected_turn_identity.set(ExpectedTurnIdentity(engine, database, revision, session_id, runtime_is_current))
    try:
        yield
    finally:
        _expected_turn_identity.reset(token)


_UNREAD_TIP = object()


def check_expected_turn_identity(agent, db, latest_session_id=_UNREAD_TIP) -> None:
    expected = _expected_turn_identity.get()
    if expected is None:
        return
    if (agent is not expected.engine or db is not expected.database
            or agent.session_identity_revision != expected.revision
            or agent.session_id != expected.session_id
            or (latest_session_id is not _UNREAD_TIP and latest_session_id != expected.session_id)
            or not expected.runtime_is_current()):
        raise ValueError("Conditional turn identity changed before admission; input was not executed")


def conditional_turn_pending() -> bool:
    return _expected_turn_identity.get() is not None


def consume_expected_turn_identity() -> None:
    # Admission has finished. Delegated engines and later background work inherit
    # the profile context, but must not inherit this parent's one-turn witness.
    _expected_turn_identity.set(None)
