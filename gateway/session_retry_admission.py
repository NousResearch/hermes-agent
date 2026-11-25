"""Messaging ``/retry`` under a session authority: the transcript rewrite commits in the re-sent
turn's own admission transaction (#139257 step 2, MSG-11).

``/retry`` drops the last exchange and re-sends its user text. Rewriting first and admitting
afterwards loses the turn whenever admission refuses (a drain or retirement that begins after the
pre-check, a full FIFO). Here the handler only PLANS the rewrite: it reads the transcript, pins the
active row ids it read, and binds the plan to the re-send. ``admit_native`` takes the binding and
hands the rewrite to ``admissions.admit``, which runs it after its own admission gate and queue cap,
in the same write that inserts the row. A refusal rolls both back; a transcript that changed since
the read refuses both (``RetryRewriteRefused``).
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from persistence.authority.admissions import TranscriptRewrite

_bound: ContextVar[Optional[RetryRewrite]] = ContextVar('retry_rewrite', default=None)


@dataclass
class RetryRewrite:
    """One ``/retry`` rewrite, applied at most once by the re-send's admission write."""
    store: Any  # the authority's storage handle: only an admission on that store takes the binding
    rewrite: TranscriptRewrite
    source: Any  # the re-send's own SessionSource object: only that admission takes the binding
    on_commit: Callable[[], None]  # on the loop, once the admission that applied it committed
    applied: bool = False
    taken: bool = field(default=False, repr=False)


def bind(rewrite: RetryRewrite):
    return _bound.set(rewrite)


def unbind(token) -> None:
    _bound.reset(token)


def take(event, store) -> Optional[RetryRewrite]:
    """The bound rewrite for this native admission, once, and only for the re-send itself (the same
    source object on the same store). A drain task inheriting the context never takes it."""
    rewrite = _bound.get()
    if (rewrite is None or rewrite.taken or getattr(event, 'source', None) is not rewrite.source
            or rewrite.store is not store):
        return None
    rewrite.taken = True
    return rewrite
