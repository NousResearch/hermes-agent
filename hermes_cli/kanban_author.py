"""The Kanban author contract: who an author-attributing ``hermes kanban`` action is from.

One home for the resolution order — ``hermes_cli.kanban``, ``kanban_specify`` and
``kanban_decompose`` all delegate here so their mirrors cannot drift:

  1. an author bound by the calling surface for this call (:func:`bind_author` — the gateway
     passes the routed chat profile, its programmatic equivalent of ``--author``),
  2. ``HERMES_PROFILE_NAME``,
  3. ``HERMES_PROFILE``.

Anything else raises :class:`KanbanAuthorRequired`. There is deliberately NO home-derived
fallback: ``HERMES_HOME`` follows ``hermes profile use``, so an unpinned caller used to
attribute its comments and status moves to whichever lane was last made active, and a
comment that reads as a human operator's can be a stale sticky profile's.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from typing import Iterator, Optional

AUTHOR_REQUIRED_MESSAGE = "cannot determine author; pass --author (or set HERMES_PROFILE)"

_bound_author: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "kanban_bound_author", default=None
)


class KanbanAuthorRequired(RuntimeError):
    """No explicit author was supplied for an author-attributing action.

    A ``RuntimeError`` on purpose: the kanban CLI's dispatch already renders that class as
    ``kanban: <message>`` on stderr with a non-zero exit, so even a path that forgets to catch
    this still fails loudly instead of writing a row nobody can attribute.
    """

    def __init__(self, message: str = AUTHOR_REQUIRED_MESSAGE) -> None:
        super().__init__(message)


@contextlib.contextmanager
def bind_author(author: Optional[str]) -> Iterator[None]:
    """Bind *author* for calls made in this context (``None``/empty = bind nothing).

    A contextvar, not ``os.environ``: concurrent gateway chats must never see each other's
    identity, and a nested bind must not outlive its caller.
    """
    token = _bound_author.set((author or "").strip() or None)
    try:
        yield
    finally:
        _bound_author.reset(token)


def resolve_author() -> str:
    """Return the explicitly signalled author, or raise :class:`KanbanAuthorRequired`."""
    for candidate in (
        _bound_author.get(),
        os.environ.get("HERMES_PROFILE_NAME"),
        os.environ.get("HERMES_PROFILE"),
    ):
        name = (candidate or "").strip()
        if name:
            return name
    raise KanbanAuthorRequired()
