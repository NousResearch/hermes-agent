"""
TaskQueue — Priority Queue for Intelligent Task Scheduling.

Provides a thread-safe priority queue for queuing delegated tasks,
with priority levels (critical/high/normal/low) and FIFO ordering
within the same priority level.

Supports:
  - Priority-based ordering (lower priority number = higher urgency)
  - Max queue size with graceful overflow handling
  - Task cancellation by goal prefix
  - Queue depth monitoring for load metrics
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Task priority levels. Lower number = higher urgency."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class QueuedTask:
    """
    A task entry in the TaskQueue.

    Attributes:
        goal:        The task goal string
        context:     Optional context string
        toolsets:    Toolset restrictions for this task
        priority:    TaskPriority enum value (default: NORMAL)
        enqueued_at: Timestamp when task was enqueued
        metadata:    Additional task metadata (e.g. source agent, session)
    """
    goal: str
    context: str = ""
    toolsets: Optional[List[str]] = None
    priority: int = TaskPriority.NORMAL
    enqueued_at: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_index: int = 0  # Index in original tasks list

    @property
    def priority_label(self) -> str:
        return TaskPriority(self.priority).name


class TaskQueue:
    """
    Thread-safe priority queue for delegated tasks.

    Tasks are ordered by (priority, enqueued_at) — highest priority
    (lowest number) first, then FIFO within the same priority.

    Backward compatible: can be used as a simple FIFO list via enqueue/dequeue.
    """

    DEFAULT_MAX_SIZE = 32

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        """
        Args:
            max_size: Maximum number of queued tasks. When exceeded,
                      lower-priority tasks are dropped first.
        """
        self._queue: List[QueuedTask] = []
        self._lock = threading.RLock()
        self._max_size = max_size

    # ── Core Queue Operations ────────────────────────────────────────────────

    def enqueue(self, task: QueuedTask) -> bool:
        """
        Add a task to the queue, maintaining priority order.

        If the queue is full, drops the lowest-priority existing task
        before adding (only if the new task has higher priority).
        Returns True if the task was added, False if dropped.
        """
        with self._lock:
            # Evict lowest-priority tasks to make room
            while len(self._queue) >= self._max_size:
                evicted = self._evict_lowest_priority()
                if evicted is None:
                    # Queue empty after eviction — can't add
                    logger.debug(
                        "TaskQueue: failed to evict for incoming task (priority=%s): %r",
                        task.priority_label, task.goal[:60],
                    )
                    return False

            # Insert in sorted position: (priority, enqueued_at)
            self._queue.append(task)
            self._queue.sort(key=lambda t: (t.priority, t.enqueued_at))
            logger.debug(
                "TaskQueue enqueued (priority=%s, depth=%d): %r",
                task.priority_label, len(self._queue), task.goal[:60],
            )
            return True

    def dequeue(self) -> Optional[QueuedTask]:
        """Remove and return the highest-priority task, or None if empty."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def peek(self) -> Optional[QueuedTask]:
        """Return the highest-priority task without removing it."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def _evict_lowest_priority(self) -> Optional[QueuedTask]:
        """Remove the lowest-priority (last) task from the queue."""
        if not self._queue:
            return None
        # Last item has lowest priority (sorted by priority ASC, enqueued_at ASC)
        return self._queue.pop()

    # ── Bulk Operations ───────────────────────────────────────────────────────

    def enqueue_many(self, tasks: List[QueuedTask]) -> int:
        """
        Enqueue multiple tasks in priority order.

        Returns the number of tasks successfully enqueued.
        """
        count = 0
        for task in tasks:
            if self.enqueue(task):
                count += 1
        return count

    def drain(self) -> List[QueuedTask]:
        """Remove and return all tasks in priority order."""
        with self._lock:
            tasks = list(self._queue)
            self._queue.clear()
            return tasks

    # ── Queue Inspection ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def depth(self) -> int:
        """Number of tasks currently queued."""
        with self._lock:
            return len(self._queue)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._queue) >= self._max_size

    def get_snapshot(self) -> List[Dict[str, Any]]:
        """Return a snapshot of all queued tasks for monitoring."""
        with self._lock:
            return [
                {
                    "goal": t.goal[:80],
                    "priority": t.priority_label,
                    "enqueued_at": round(t.enqueued_at, 2),
                    "task_index": t.task_index,
                }
                for t in self._queue
            ]

    def get_by_priority(self, priority: int) -> List[QueuedTask]:
        """Return all tasks at a given priority level (in FIFO order)."""
        with self._lock:
            return [t for t in self._queue if t.priority == priority]

    # ── Task Management ───────────────────────────────────────────────────────

    def cancel_by_goal_prefix(self, prefix: str) -> int:
        """
        Remove all tasks whose goal starts with the given prefix.

        Returns the number of tasks cancelled.
        """
        with self._lock:
            before = len(self._queue)
            self._queue = [t for t in self._queue if not t.goal.startswith(prefix)]
            removed = before - len(self._queue)
            if removed:
                logger.debug("TaskQueue cancelled %d tasks with prefix %r", removed, prefix)
            return removed

    def clear(self) -> int:
        """Remove all tasks. Returns the number of tasks removed."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count
