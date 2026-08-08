"""Shared active-host-parent binding for plugin-facing lifecycle APIs.

Hermes binds the currently running parent agent for the duration of an agent
turn (``run_agent.run_conversation``). Plugin-facing services that must bind a
registration to the *active* host parent — the subagent lifecycle API and the
durable external background-task API — resolve that parent through this module
instead of reading arbitrary platform/chat/session values from the caller.

The binding is deliberately host-owned: plugins never set it, and never pass a
parent/session identity into the service methods. See
``agent.subagent_lifecycle.bind_subagent_parent`` for the historical wrapper
that delegates here.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Optional


_ACTIVE_HOST_PARENT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "hermes_host_parent", default=None
)


@contextmanager
def bind_host_parent(parent_agent: Any):
    """Bind the host-owned parent for the current agent turn.

    Restores the previous value on exit. Safe to nest: each call saves and
    restores its own token.
    """
    token = _ACTIVE_HOST_PARENT.set(parent_agent)
    try:
        yield
    finally:
        _ACTIVE_HOST_PARENT.reset(token)


def get_active_host_parent() -> Optional[Any]:
    """Return the parent bound to this execution context, if any."""
    return _ACTIVE_HOST_PARENT.get()
