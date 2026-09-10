"""Typed structures for DAG orchestration (stdlib only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, MutableMapping, Sequence


class TaskStatus(str, Enum):
    """Lifecycle state for a graph node."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class GraphTaskSpec:
    """DAG node definition supplied by callers."""

    task_id: str
    goal: str
    depends_on: tuple[str, ...] = ()
    context: str | None = None
    toolsets: Sequence[str] | None = None
    profile_env: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GraphTaskRun:
    """Mutable runtime record while executing a node."""

    spec: GraphTaskSpec
    status: TaskStatus = TaskStatus.PENDING
    summary: str | None = None
    error: str | None = None
    extra: MutableMapping[str, Any] = field(default_factory=dict)
