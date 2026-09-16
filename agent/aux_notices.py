"""User-visible notices for auxiliary (side-task) route switches.

The main conversation loop already surfaces a provider/model switch through
``AIAgent._emit_pending_fallback_notice``. Auxiliary work — compression,
titles, vision, web extraction, session search — resolves its own client and
walks its own fallback chain with no such surface: a switch there only ever
produced a ``logger.info`` line, so a user could have every side task silently
served by a different provider with nothing on screen.

This module is the missing channel. It mirrors the ``agent.aux_accounting``
design (issue #23270): the agent loop publishes a sink for the duration of a
turn, auxiliary code calls :func:`emit_aux_notice`, and everything is
best-effort — a notice must never break the call it is reporting on.

Deduplication is per published scope (one turn / one background context), so a
retry storm inside one call produces a single notice while a genuinely new
switch in a later turn is still reported.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Callable, Optional, Set

logger = logging.getLogger(__name__)


class _NoticeScope:
    """One published notice scope: the sink plus what it already showed."""

    __slots__ = ("sink", "seen")

    def __init__(self, sink: Callable[[str], None]) -> None:
        self.sink = sink
        self.seen: Set[str] = set()


_scope: contextvars.ContextVar[Optional[_NoticeScope]] = contextvars.ContextVar(
    "aux_notice_scope", default=None
)


def set_notice_sink(sink: Callable[[str], None]):
    """Publish *sink* as the user-visible channel for this context.

    Returns the token to hand back to :func:`reset_notice_sink`. Each call
    starts a fresh dedupe scope.
    """
    return _scope.set(_NoticeScope(sink))


def reset_notice_sink(token) -> None:
    """Restore the previous notice scope (best-effort)."""
    try:
        _scope.reset(token)
    except Exception:
        _scope.set(None)


def emit_aux_notice(message: str) -> None:
    """Send one aux route-switch notice to the published sink.

    No-ops when nothing published a sink (an auxiliary call outside any agent
    turn), when the message is empty, or when this scope already showed the
    same line. Never raises.
    """
    scope = _scope.get()
    if scope is None or not message:
        return
    if message in scope.seen:
        return
    scope.seen.add(message)
    try:
        scope.sink(message)
    except Exception:
        logger.debug("Aux notice sink failed", exc_info=True)
