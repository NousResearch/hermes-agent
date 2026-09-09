"""Command-scoped lifecycle ownership for Console executions."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_CURRENT_CONSOLE_EXECUTION: ContextVar["ConsoleExecution | None"] = ContextVar(
    "hermes_console_execution", default=None,
)


def current_console_execution() -> "ConsoleExecution | None":
    return _CURRENT_CONSOLE_EXECUTION.get()


@contextmanager
def bind_console_execution(execution: "ConsoleExecution | None") -> Iterator[None]:
    token = _CURRENT_CONSOLE_EXECUTION.set(execution)
    try:
        yield
    finally:
        _CURRENT_CONSOLE_EXECUTION.reset(token)


class ConsoleExecution:
    """Own one Console command's executor future and active AIAgent.

    The concurrent future is retained separately from the asyncio wrapper so
    cancellation can distinguish queued work (which can be cancelled cheaply)
    from work that has started and must unwind cooperatively.
    """

    def __init__(self, command_id: int) -> None:
        self.command_id = command_id
        self._lock = threading.RLock()
        self._future: concurrent.futures.Future[Any] | None = None
        self._agent: Any | None = None
        self._cancelled = False
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def active_agent(self) -> Any | None:
        with self._lock:
            return self._agent

    def bind_future(self, future: concurrent.futures.Future[Any]) -> None:
        with self._lock:
            self._future = future
            cancelled = self._cancelled
        if cancelled:
            future.cancel()

    def bind_agent(self, agent: Any) -> None:
        with self._lock:
            if not self._cancelled:
                self._agent = agent
                return
            reason = self._reason or "Console command cancelled"
        self._interrupt(agent, reason)

    def unbind_agent(self, agent: Any) -> None:
        with self._lock:
            if self._agent is agent:
                self._agent = None

    def cancel(self, reason: str = "cancelled") -> bool:
        """Latch cancellation, interrupt the active agent, and cancel queued work."""
        message = reason if reason.startswith("Console command ") else f"Console command {reason}"
        with self._lock:
            first = not self._cancelled
            if first:
                self._cancelled = True
                self._reason = message
            agent = self._agent
            future = self._future
        if first and agent is not None:
            self._interrupt(agent, message)
        if future is not None:
            future.cancel()
        return first

    @staticmethod
    def _interrupt(agent: Any, reason: str) -> None:
        from agent.interrupt_compat import request_hard_interrupt

        request_hard_interrupt(agent, reason, tool_reason="console cancellation")

    async def wait(self) -> None:
        """Wait until owned executor work has completed or was queued-cancelled."""
        with self._lock:
            future = self._future
        if future is None:
            return
        try:
            await asyncio.shield(asyncio.wrap_future(future))
        except asyncio.CancelledError:
            if not future.cancelled():
                raise
        except Exception:
            if not self.cancelled:
                raise
