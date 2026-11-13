"""Messaging ``/retry`` under a session authority: the transcript rewrite commits in the re-sent
turn's own admission transaction (#139257 step 2, MSG-11).

``/retry`` drops the last exchange and re-sends its user text. Rewriting first and admitting
afterwards loses the turn whenever admission refuses (a drain or retirement that begins after the
pre-check, a full FIFO). Here the handler only PLANS the rewrite: it reads the transcript, pins the
active row ids it read, and binds the plan to the re-send. ``admit_native`` takes the binding and
runs the rewrite as a write guard after its own admission gate and queue cap, in the same
``BEGIN IMMEDIATE`` that inserts the row. A refusal rolls both back; a transcript that changed since
the read refuses both (``RetryRewriteRefused``).
"""
from __future__ import annotations

import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from hermes_state_runtime import RuntimeStoreError

_bound: ContextVar[Optional[RetryRewrite]] = ContextVar('retry_rewrite', default=None)


class RetryRewriteRefused(RuntimeError):
    """The planned rewrite no longer applies (the transcript changed since the read, a turn lease is
    live, the session closed): the re-send was refused with it and nothing was written."""


@dataclass
class RetryRewrite:
    """One ``/retry`` rewrite, applied at most once inside the re-send's admission write."""
    db: Any
    session_id: str
    source: Any  # the re-send's own SessionSource object: only that admission takes the binding
    write: Callable[[Any], None]  # (conn) -> None, inside the admission transaction
    on_commit: Callable[[], None]  # on the loop, once the admission that applied it committed
    applied: bool = False
    taken: bool = field(default=False, repr=False)

    def guard(self, conn):
        self.applied = False  # the writer may retry the whole transaction
        try:
            self.write(conn)
        except (RuntimeStoreError, sqlite3.OperationalError):
            raise  # gate refusals and writer contention keep their own handling
        except Exception as exc:
            raise RetryRewriteRefused(str(exc)) from exc
        self.applied = True


def bind(rewrite: RetryRewrite):
    return _bound.set(rewrite)


def unbind(token) -> None:
    _bound.reset(token)


def take(event, db) -> Optional[RetryRewrite]:
    """The bound rewrite for this native admission, once, and only for the re-send itself (the same
    source object on the same store). A drain task inheriting the context never takes it."""
    rewrite = _bound.get()
    if (rewrite is None or rewrite.taken or getattr(event, 'source', None) is not rewrite.source
            or rewrite.db is not db):
        return None
    rewrite.taken = True
    return rewrite


def compose(rewrite: Optional[RetryRewrite], authorize):
    """``authorize`` (the admission gate) first, then the rewrite, in one write transaction."""
    if rewrite is None:
        return authorize

    def guard(conn):
        authorize(conn)
        rewrite.guard(conn)
    return guard
