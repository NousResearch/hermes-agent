"""Task types shared by the scheduler engine, wheels, and plugin registration.

Design: website/docs/developer-guide/scheduled-tasks.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

PERIODS = ("turn", "step", "cron")
PHASES = ("pre", "post")
_NS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*\.[a-z0-9][a-z0-9_.-]*$", re.IGNORECASE)


@dataclass(frozen=True)
class StepView:
    """The committed step a post-phase task observes (never a live agent handle)."""

    step_seq: int
    step_index: Optional[int]
    usage_status: str
    status: str = "ok"
    api_calls: int = 1
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    duration_ms: int = 0
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class TaskContext:
    """Frozen, deliberately narrow payload handed to a scheduled handler.

    No agent handle, no message list: turn/step tasks are observers. That invariant is what
    keeps prompt caching intact.
    """

    session_id: str
    profile_name: str
    agent_name: str
    phase: str
    now: float
    turn_id: Optional[str] = None
    turn_seq: Optional[int] = None
    step_seq: Optional[int] = None
    step_index: Optional[int] = None
    result_status: Optional[str] = None
    step: Optional[StepView] = None


Handler = Callable[[TaskContext], Any | Awaitable[Any]]


@dataclass(frozen=True)
class TaskSpec:
    plugin_id: str
    name: str
    period: str
    every: int
    phase: str
    handler: Handler
    timeout_ms: Optional[int] = None
    once: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.plugin_id, self.name)

    @property
    def handler_name(self) -> str:
        return getattr(self.handler, "__qualname__", repr(self.handler))


def validate_spec(spec: TaskSpec, *, max_tasks_per_plugin: int = 32, existing: int = 0) -> None:
    """Raise ``ValueError`` for a spec the engine must not register."""
    if spec.period not in ("turn", "step"):
        raise ValueError(f"unsupported period {spec.period!r} (turn|step)")
    if spec.phase not in PHASES:
        raise ValueError(f"unsupported phase {spec.phase!r} (pre|post)")
    if not isinstance(spec.every, int) or isinstance(spec.every, bool) or spec.every < 1:
        raise ValueError("every must be an integer >= 1")
    if not spec.name or not spec.name.startswith(f"{spec.plugin_id}."):
        raise ValueError("task name must be prefixed with its plugin id")
    if not _NS_PATTERN.match(spec.name):
        raise ValueError("task name must be namespaced (plugin.task)")
    if not callable(spec.handler):
        raise ValueError("handler must be callable")
    if spec.timeout_ms is not None and not (1 <= int(spec.timeout_ms) <= 60_000):
        raise ValueError("timeout_ms must be within 1..60000")
    if existing >= max_tasks_per_plugin:
        raise ValueError(f"plugin {spec.plugin_id!r} already registered {existing} tasks "
                         f"(max {max_tasks_per_plugin})")
