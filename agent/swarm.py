"""Multi-agent swarm coordination for Hermes Agent.

Enables multiple agent instances to collaborate on complex tasks by:
1. **Task decomposition** — breaking large tasks into subtasks
2. **Parallel execution** — running subtasks concurrently
3. **Result aggregation** — combining results with conflict resolution
4. **Dynamic reassignment** — reassigning failed subtasks to other agents

Architecture
------------
```
                    ┌─────────────┐
                    │  Swarm API  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
        │ Agent A   │ │Agent B│ │ Agent C   │
        │ (search)  │ │(write)│ │ (review)  │
        └───────────┘ └───────┘ └───────────┘
```

Config
------
```yaml
swarm:
  enabled: true
  max_agents: 5
  task_timeout_seconds: 600
  retry_failed: true
  retry_limit: 2
```
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a swarm task."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Priority level for swarm tasks."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class SwarmTask:
    """A single task in a swarm operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = ""
    result: Any = None
    error: str = ""
    retries: int = 0
    max_retries: int = 2
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class SwarmOperation:
    """A multi-agent swarm operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    tasks: list[SwarmTask] = field(default_factory=list)
    max_agents: int = 5
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Return completion percentage (0-100)."""
        if not self.tasks:
            return 0.0
        done = sum(1 for t in self.tasks if t.is_done)
        return (done / len(self.tasks)) * 100

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)


class SwarmCoordinator:
    """Coordinates multi-agent swarm operations."""

    def __init__(
        self,
        max_agents: int = 5,
        task_timeout: float = 600.0,
        retry_failed: bool = True,
        retry_limit: int = 2,
    ):
        self.max_agents = max_agents
        self.task_timeout = task_timeout
        self.retry_failed = retry_failed
        self.retry_limit = retry_limit
        self._operations: dict[str, SwarmOperation] = {}
        self._handlers: dict[str, Callable] = {}

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a task type.

        Parameters
        ----------
        task_type:
            Task type identifier (e.g. "search", "write", "review").
        handler:
            Async callable that executes the task.
        """
        self._handlers[task_type] = handler
        logger.info("Swarm registered handler: %s", task_type)

    def create_operation(
        self,
        name: str,
        tasks: list[dict[str, Any]],
        description: str = "",
    ) -> SwarmOperation:
        """Create a new swarm operation.

        Parameters
        ----------
        name:
            Operation name.
        tasks:
            List of task dicts with keys: name, description, type, priority.
        description:
            Operation description.

        Returns
        -------
        SwarmOperation
        """
        op = SwarmOperation(
            name=name,
            description=description,
            max_agents=self.max_agents,
        )

        for task_def in tasks:
            task = SwarmTask(
                name=task_def.get("name", ""),
                description=task_def.get("description", ""),
                priority=TaskPriority[task_def.get("priority", "NORMAL").upper()],
                max_retries=self.retry_limit,
                metadata={"type": task_def.get("type", "general")},
            )
            op.tasks.append(task)

        self._operations[op.id] = op
        logger.info("Swarm created operation '%s' with %d tasks", name, len(tasks))
        return op

    def get_operation(self, op_id: str) -> Optional[SwarmOperation]:
        """Get an operation by ID."""
        return self._operations.get(op_id)

    def list_operations(self) -> list[SwarmOperation]:
        """List all operations."""
        return list(self._operations.values())

    def get_status_report(self, op_id: str) -> dict[str, Any]:
        """Get a detailed status report for an operation."""
        op = self._operations.get(op_id)
        if not op:
            return {"error": f"Operation {op_id} not found"}

        task_reports = []
        for task in op.tasks:
            task_reports.append({
                "id": task.id,
                "name": task.name,
                "status": task.status.name,
                "agent": task.assigned_agent,
                "duration": task.duration,
                "retries": task.retries,
                "error": task.error,
            })

        return {
            "operation": op.name,
            "id": op.id,
            "status": op.status.name,
            "progress": op.progress,
            "completed": op.completed_count,
            "failed": op.failed_count,
            "total": len(op.tasks),
            "tasks": task_reports,
        }


# Global swarm coordinator
_swarm: Optional[SwarmCoordinator] = None


def get_swarm_coordinator() -> SwarmCoordinator:
    """Get or create the global swarm coordinator."""
    global _swarm
    if _swarm is None:
        _swarm = SwarmCoordinator()
    return _swarm


def reset_swarm() -> None:
    """Reset the global swarm coordinator (for testing)."""
    global _swarm
    _swarm = None
