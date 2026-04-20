"""Runtime backends for digital employees.

Two implementations:

* :class:`SimulatedRuntime` — synthetic, deterministic, *no* LLM calls. Default
  on first launch (Story 4.10) so kids and demos never spend tokens.
* :class:`HermesRuntime` — wraps :class:`run_agent.AIAgent` running on a worker
  thread; emits real activity events.

Both implement the same :class:`Runtime` protocol below so the API server is
runtime-agnostic.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from ..models import ActivityEvent, Employee, Task

EventCallback = Callable[[ActivityEvent], Awaitable[None]]


class TaskResult:
    """Lightweight value-type returned by ``run_task``."""

    __slots__ = ("status", "summary", "tokens_in", "tokens_out", "error")

    def __init__(
        self,
        *,
        status: str = "done",
        summary: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: str | None = None,
    ) -> None:
        self.status = status
        self.summary = summary
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.error = error

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "summary": self.summary,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error": self.error,
        }


class Runtime(Protocol):
    name: str

    async def run_task(
        self,
        employee: Employee,
        task: Task,
        on_event: EventCallback,
    ) -> TaskResult:
        ...


def make_runtime(kind: str) -> Runtime:
    """Factory used by :func:`hermes_office.make_runtime`. Lazy imports keep
    the heavy ``run_agent`` module out of the import path until needed."""
    kind = (kind or "simulated").lower()
    if kind == "simulated":
        from .simulated import SimulatedRuntime

        return SimulatedRuntime()
    if kind == "hermes":
        from .hermes import HermesRuntime

        return HermesRuntime()
    raise ValueError(f"unknown runtime kind: {kind!r}")


__all__ = ["Runtime", "TaskResult", "EventCallback", "make_runtime"]
