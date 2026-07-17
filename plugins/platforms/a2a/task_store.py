"""Bounded A2A task persistence with monotonic terminal states."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import Task, TaskState

_TERMINAL_STATES = frozenset({
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
})


def _clone(task: Task) -> Task:
    return task.model_copy(deep=True)


class BoundedTaskStore(TaskStore):
    """Keep task state bounded and prevent terminal-state resurrection.

    The SDK's task managers mutate retrieved task objects before saving them.
    Returning deep copies prevents concurrent consumers (notably cancel and the
    original non-blocking send consumer) from racing through shared references.
    Once a terminal state is persisted it is authoritative and cannot be
    overwritten by an older queued ``working`` event.
    """

    def __init__(self, *, max_tasks: int, max_history_messages: int):
        if max_tasks <= 0 or max_history_messages <= 0:
            raise ValueError("A2A task-store bounds must be positive")
        self._max_tasks = max_tasks
        self._max_history_messages = max_history_messages
        self._tasks: OrderedDict[str, Task] = OrderedDict()
        self._lock = asyncio.Lock()

    async def save(self, task: Task, context: ServerCallContext | None = None) -> None:
        del context
        incoming = _clone(task)
        if incoming.history and len(incoming.history) > self._max_history_messages:
            incoming.history = incoming.history[-self._max_history_messages :]

        async with self._lock:
            current = self._tasks.get(incoming.id)
            if (
                current is not None
                and current.status.state in _TERMINAL_STATES
                and incoming.status.state not in _TERMINAL_STATES
            ):
                return
            if (
                current is not None
                and current.status.state in _TERMINAL_STATES
                and incoming.status.state in _TERMINAL_STATES
            ):
                return
            self._tasks[incoming.id] = incoming
            self._tasks.move_to_end(incoming.id)
            while len(self._tasks) > self._max_tasks:
                victim = next(
                    (
                        task_id
                        for task_id, retained in self._tasks.items()
                        if retained.status.state in _TERMINAL_STATES
                    ),
                    None,
                )
                if victim is None:
                    # Never make a running task disappear from tasks/get or
                    # tasks/cancel. Admission bounds active tasks; repair the
                    # temporary overshoot as soon as one becomes terminal.
                    break
                del self._tasks[victim]

    async def get(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> Task | None:
        del context
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            self._tasks.move_to_end(task_id)
            return _clone(task)

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        del context
        async with self._lock:
            self._tasks.pop(task_id, None)

    async def size(self) -> int:
        async with self._lock:
            return len(self._tasks)
