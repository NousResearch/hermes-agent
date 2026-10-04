"""
Load-Aware Task Router — Intelligent Routing Based on Agent Capacity.

Provides load-based routing that routes tasks to the least-loaded
subagent or queues them when capacity is reached.

Integrates with SubagentCoordinator for live load tracking and
TaskQueue for overflow handling.

GL Design Principles:
  - Module Independence: No coupling to AIAgent internals
  - Backward Compatible: Falls back to simple routing when no load data
  - Generative Loop: Updates load state after each task completion
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.subagent_coordinator import SubagentCoordinator

logger = logging.getLogger(__name__)


# ─── Load Metrics ─────────────────────────────────────────────────────────────


class LoadStrategy:
    """Load balancing strategy constants."""
    LEAST_LOADED = "least_loaded"  # Route to agent with fewest active tasks
    ROUND_ROBIN = "round_robin"     # Rotate among available agents
    PRIORITY_QUEUE = "priority_queue"  # Use TaskQueue for overflow


# ─── Agent Load Tracker ───────────────────────────────────────────────────────


@dataclass
class AgentLoad:
    """
    Load metrics for a single agent/subagent.

    Attributes:
        agent_id:     Unique identifier (usually thread ID or session ID)
        active_tasks: Number of currently running tasks
        completed:    Total tasks completed by this agent
        failed:       Total tasks that failed
        avg_duration: Rolling average task duration in seconds
        load_score:   Computed load score (lower = less loaded)
                      Formula: active_tasks * 10 + (failed / max(completed, 1)) * 5
        last_updated: Timestamp of last update
    """
    agent_id: Any
    active_tasks: int = 0
    completed: int = 0
    failed: int = 0
    avg_duration: float = 0.0
    total_duration: float = 0.0
    load_score: float = 0.0
    last_updated: float = field(default_factory=time.monotonic)

    def update_completion(self, duration: float, success: bool) -> None:
        """Update load after a task completes."""
        self.active_tasks = max(0, self.active_tasks - 1)
        self.completed += 1 if success else 0
        self.failed += 0 if success else 1
        self.total_duration += duration
        if self.completed > 0:
            self.avg_duration = self.total_duration / self.completed
        self._recompute_score()
        self.last_updated = time.monotonic()

    def update_start(self) -> None:
        """Update load when a task starts."""
        self.active_tasks += 1
        self._recompute_score()
        self.last_updated = time.monotonic()

    def _recompute_score(self) -> None:
        """Recompute the load score. Lower = less loaded."""
        failure_ratio = self.failed / max(self.completed, 1)
        self.load_score = self.active_tasks * 10.0 + failure_ratio * 5.0

    @property
    def is_available(self) -> bool:
        """True if the agent can accept more tasks."""
        return True  # Subclass can override with capacity limits


# ─── LoadAwareRouter ──────────────────────────────────────────────────────────


class LoadAwareRouter:
    """
    Routes tasks to agents based on live load metrics.

    Maintains a registry of agent loads and routes new tasks to the
    least-loaded available agent. When all agents are at capacity,
    tasks are queued in a TaskQueue for deferred routing.

    Thread-safe: all operations use the instance lock.

    Integration:
        - Initialized in bootstrap with SubagentCoordinator reference
        - Updated by SubagentCoordinator on task start/complete/error/timeout
        - Consulted by delegate_task before spawning a new subagent
    """

    def __init__(
        self,
        coordinator: Optional["SubagentCoordinator"] = None,
        max_concurrent: int = 3,
        strategy: str = LoadStrategy.LEAST_LOADED,
        queue_enabled: bool = True,
    ):
        """
        Args:
            coordinator:     SubagentCoordinator to sync load state with
            max_concurrent:  Maximum concurrent tasks before queueing (default: 3)
            strategy:        Load balancing strategy
            queue_enabled:    Whether to use TaskQueue for overflow (default: True)
        """
        self._coordinator = coordinator
        self._max_concurrent = max_concurrent
        self._strategy = strategy
        self._queue_enabled = queue_enabled

        # Agent load registry: agent_id → AgentLoad
        self._loads: Dict[Any, AgentLoad] = {}
        self._lock = threading.RLock()

        # Round-robin state
        self._rr_index: int = 0
        self._rr_lock = threading.Lock()

        # Lazy import to avoid cycle at module load
        self._task_queue = None

        # Feature flag (can be disabled for backward compatibility)
        self._enabled = True

        logger.debug(
            "LoadAwareRouter initialized: max_concurrent=%d, strategy=%s, queue=%s",
            max_concurrent, strategy, queue_enabled,
        )

    # ── Configuration ────────────────────────────────────────────────────────

    @property
    def max_concurrent(self) -> int:
        """Maximum concurrent tasks before queueing."""
        return self._max_concurrent

    @max_concurrent.setter
    def max_concurrent(self, value: int) -> None:
        self._max_concurrent = max(1, value)

    @property
    def strategy(self) -> str:
        return self._strategy

    @strategy.setter
    def strategy(self, value: str) -> None:
        if value in (LoadStrategy.LEAST_LOADED, LoadStrategy.ROUND_ROBIN, LoadStrategy.PRIORITY_QUEUE):
            self._strategy = value

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        self._enabled = value

    # ── Agent Registration ───────────────────────────────────────────────────

    def register_agent(self, agent_id: Any) -> AgentLoad:
        """
        Register an agent for load tracking.

        Returns the AgentLoad record for that agent.
        """
        with self._lock:
            if agent_id not in self._loads:
                self._loads[agent_id] = AgentLoad(agent_id=agent_id)
                logger.debug("LoadAwareRouter registered agent %s", agent_id)
            return self._loads[agent_id]

    def unregister_agent(self, agent_id: Any) -> None:
        """Remove an agent from load tracking."""
        with self._lock:
            if agent_id in self._loads:
                del self._loads[agent_id]
                logger.debug("LoadAwareRouter unregistered agent %s", agent_id)

    def get_load(self, agent_id: Any) -> Optional[AgentLoad]:
        """Return the load record for an agent, or None if not registered."""
        with self._lock:
            return self._loads.get(agent_id)

    # ── Load Updates ─────────────────────────────────────────────────────────

    def on_task_start(self, agent_id: Any) -> None:
        """Called when a task starts on an agent."""
        load = self.register_agent(agent_id)
        load.update_start()
        logger.debug(
            "LoadAwareRouter: task started on %s (active=%d, score=%.1f)",
            agent_id, load.active_tasks, load.load_score,
        )

    def on_task_complete(
        self,
        agent_id: Any,
        duration: float,
        success: bool = True,
    ) -> None:
        """Called when a task completes on an agent."""
        with self._lock:
            load = self._loads.get(agent_id)
            if load is None:
                logger.debug("LoadAwareRouter: unknown agent %s for completion", agent_id)
                return
            load.update_completion(duration, success)
            logger.debug(
                "LoadAwareRouter: task completed on %s (active=%d, completed=%d, score=%.1f)",
                agent_id, load.active_tasks, load.completed, load.load_score,
            )

    def sync_with_coordinator(self) -> None:
        """
        Sync load state from SubagentCoordinator records.

        Useful on initialization or after coordinator state changes.
        """
        if self._coordinator is None:
            return
        with self._lock:
            records = self._coordinator.get_records()
            for record in records:
                agent_id = id(record.agent)
                load = self.register_agent(agent_id)
                # Sync active count
                if record.is_alive and load.active_tasks == 0:
                    load.active_tasks = 1
                    load._recompute_score()

    # ── Routing Decision ─────────────────────────────────────────────────────

    def should_spawn(self) -> bool:
        """
        Determine whether to spawn a new subagent or queue the task.

        Returns True if a new agent can be spawned (under max_concurrent),
        False if tasks should be queued.
        """
        if not self._enabled:
            return True

        with self._lock:
            total_active = sum(l.active_tasks for l in self._loads.values())
            return total_active < self._max_concurrent

    def select_agent(self) -> Optional[Any]:
        """
        Select the best agent based on the configured strategy.

        Returns agent_id of the selected agent, or None if no agents registered.
        """
        if not self._enabled:
            return None

        with self._lock:
            if not self._loads:
                return None

            if self._strategy == LoadStrategy.LEAST_LOADED:
                return self._select_least_loaded()
            elif self._strategy == LoadStrategy.ROUND_ROBIN:
                return self._select_round_robin()
            else:
                return self._select_least_loaded()

    def _select_least_loaded(self) -> Optional[Any]:
        """Select the agent with the lowest load score."""
        available = [
            (load.agent_id, load.load_score, load.active_tasks)
            for load in self._loads.values()
            if load.is_available
        ]
        if not available:
            return None
        # Sort by score ASC, then active_tasks ASC (tiebreak)
        available.sort(key=lambda x: (x[1], x[2]))
        return available[0][0]

    def _select_round_robin(self) -> Optional[Any]:
        """Select agents in rotation, skipping agents at capacity."""
        if not self._loads:
            return None
        available = [aid for aid, load in self._loads.items() if load.is_available]
        if not available:
            return None
        # Advance to next available index
        with self._rr_lock:
            start = self._rr_index
            for _ in range(len(available)):
                self._rr_index = (self._rr_index + 1) % len(available)
                candidate = available[self._rr_index]
                if self._loads[candidate].active_tasks < self._max_concurrent:
                    return candidate
            # All at capacity — return the first anyway
            return available[self._rr_index % len(available)]

    # ── TaskQueue Integration ────────────────────────────────────────────────

    def _get_task_queue(self):
        """Lazily create and return the TaskQueue instance."""
        if self._task_queue is None:
            from agent.routing.task_queue import TaskQueue
            self._task_queue = TaskQueue(max_size=32)
        return self._task_queue

    @property
    def task_queue(self):
        """Access the TaskQueue (for monitoring / integration)."""
        return self._get_task_queue()

    def enqueue_task(self, goal: str, context: str = "", toolsets: Optional[List[str]] = None,
                     priority: int = 2, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a task to the overflow queue.

        Returns True if queued, False if queue is full and task was dropped.
        """
        if not self._queue_enabled:
            return False
        from agent.routing.task_queue import QueuedTask
        task = QueuedTask(
            goal=goal, context=context, toolsets=toolsets,
            priority=priority, metadata=metadata or {},
        )
        return self._get_task_queue().enqueue(task)

    def dequeue_task(self) -> Optional[Any]:
        """Dequeue the highest-priority overflow task."""
        if not self._queue_enabled:
            return None
        return self._get_task_queue().dequeue()

    def queue_depth(self) -> int:
        """Number of tasks in the overflow queue."""
        if not self._queue_enabled or self._task_queue is None:
            return 0
        return self._task_queue.depth

    # ── System Metrics ────────────────────────────────────────────────────────

    def get_total_active(self) -> int:
        """Total number of active tasks across all agents."""
        with self._lock:
            return sum(l.active_tasks for l in self._loads.values())

    def get_total_completed(self) -> int:
        """Total number of completed tasks."""
        with self._lock:
            return sum(l.completed for l in self._loads.values())

    def get_load_snapshot(self) -> List[Dict[str, Any]]:
        """Return a snapshot of all agent loads for monitoring."""
        with self._lock:
            return [
                {
                    "agent_id": str(load.agent_id),
                    "active_tasks": load.active_tasks,
                    "completed": load.completed,
                    "failed": load.failed,
                    "avg_duration": round(load.avg_duration, 2),
                    "load_score": round(load.load_score, 2),
                    "last_updated": round(load.last_updated, 2),
                }
                for load in self._loads.values()
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate routing statistics."""
        with self._lock:
            total_active = sum(l.active_tasks for l in self._loads.values())
            total_completed = sum(l.completed for l in self._loads.values())
            total_failed = sum(l.failed for l in self._loads.values())
            avg_score = (
                sum(l.load_score for l in self._loads.values()) / len(self._loads)
                if self._loads else 0.0
            )
            return {
                "enabled": self._enabled,
                "strategy": self._strategy,
                "max_concurrent": self._max_concurrent,
                "registered_agents": len(self._loads),
                "total_active_tasks": total_active,
                "total_completed": total_completed,
                "total_failed": total_failed,
                "avg_load_score": round(avg_score, 2),
                "queue_depth": self.queue_depth(),
                "queue_enabled": self._queue_enabled,
            }
