"""Host-owned cancellation for agents created deep inside synchronous work.

A host that runs a blocking command on a worker thread (Hermes Console) never sees
the ``AIAgent`` a CLI subcommand forks inside it, so it cannot call ``interrupt()``
when the user cancels. The host binds an :class:`InterruptScope` around the work;
every ``run_conversation()`` under that scope registers its agent, and
``scope.cancel()`` hard-interrupts them from any thread. Agents registering after
the cancel are interrupted immediately, so a cancel never loses the race with a
turn that has not started yet (#106179).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from agent.interrupt_compat import request_hard_interrupt

_ACTIVE_SCOPE: ContextVar[Optional["InterruptScope"]] = ContextVar("hermes_interrupt_scope", default=None)


class InterruptScope:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: list[Any] = []
        self.reason: Optional[str] = None

    def cancel(self, reason: str) -> None:
        """Latch ``reason`` and hard-interrupt every agent running under this scope."""
        with self._lock:
            self.reason = reason
            agents = list(self._agents)
        for agent in agents:
            request_hard_interrupt(agent, reason, tool_reason="host cancelled the command")

    @contextmanager
    def track(self, agent: Any) -> Iterator[None]:
        with self._lock:
            self._agents.append(agent)
            reason = self.reason
        if reason is not None:
            request_hard_interrupt(agent, reason, tool_reason="host cancelled the command")
        try:
            yield
        finally:
            with self._lock:
                self._agents.remove(agent)


@contextmanager
def bind_interrupt_scope(scope: Optional[InterruptScope]) -> Iterator[None]:
    """Make ``scope`` the owner of every agent turn started in this context."""
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield
    finally:
        _ACTIVE_SCOPE.reset(token)


def track_in_interrupt_scope(agent: Any):
    """Register ``agent`` with the bound scope for the duration of its turn (no-op without one)."""
    scope = _ACTIVE_SCOPE.get()
    return nullcontext() if scope is None else scope.track(agent)
