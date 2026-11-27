"""The kanban root: today's ``HERMES_KANBAN_HOME`` resolution plus the context-scoped override.

``kanban_home()`` stays in ``hermes_cli.kanban_db`` (the seam every caller patches); this sibling
owns the context-scoped pin the storage port's local backend needs: a store opens for one root
and must keep every operation inside it, whatever the ambient environment later says. With no
pin set, ``kanban_home()`` reads ``HERMES_KANBAN_HOME`` exactly as before.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

_KANBAN_HOME_OVERRIDE: ContextVar[str | None] = ContextVar(
    "hermes_kanban_home_override", default=None,
)


def scoped_home_override() -> str | None:
    """The context-scoped kanban root pin, when one is set (``None`` otherwise)."""
    return _KANBAN_HOME_OVERRIDE.get()


@contextlib.contextmanager
def scoped_kanban_home(home: str | Path) -> Iterator[None]:
    """Pin the kanban root for the current context only.

    The storage port's local backend opens a store for one root and must keep every operation
    inside it, whatever the ambient environment later says; tests flip roots within one process.
    With no pin set this context changes nothing (``kanban_home()`` reads ``HERMES_KANBAN_HOME``
    as before)."""
    token = _KANBAN_HOME_OVERRIDE.set(str(home))
    try:
        yield
    finally:
        _KANBAN_HOME_OVERRIDE.reset(token)
