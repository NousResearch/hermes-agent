"""Bounded task persistence and terminal-state monotonicity."""

from __future__ import annotations

import asyncio

from a2a.types import TaskState, TaskStatus
from a2a.utils import new_task

from plugins.platforms.a2a.task_store import BoundedTaskStore


def _task(fakes, task_id: str, state: TaskState):
    task = new_task(fakes.make_user_message("work", context_id="ctx-store"))
    task.id = task_id
    task.status = TaskStatus(state=state)
    return task


def test_terminal_state_cannot_be_resurrected_by_stale_working_event(fakes):
    async def exercise():
        store = BoundedTaskStore(max_tasks=10, max_history_messages=10)
        await store.save(_task(fakes, "task-1", TaskState.working))
        await store.save(_task(fakes, "task-1", TaskState.canceled))
        await store.save(_task(fakes, "task-1", TaskState.working))

        persisted = await store.get("task-1")
        assert persisted is not None
        assert persisted.status.state == TaskState.canceled

    asyncio.run(exercise())


def test_task_count_and_status_history_are_bounded(fakes):
    async def exercise():
        store = BoundedTaskStore(max_tasks=2, max_history_messages=2)
        first = _task(fakes, "task-1", TaskState.working)
        first.history = [
            fakes.make_user_message(str(index), context_id="ctx-store")
            for index in range(4)
        ]
        await store.save(first)
        await store.save(_task(fakes, "task-2", TaskState.completed))
        await store.save(_task(fakes, "task-3", TaskState.working))

        assert await store.size() == 2
        assert await store.get("task-1") is not None
        assert await store.get("task-2") is None

        retained = _task(fakes, "task-1", TaskState.working)
        retained.history = [
            fakes.make_user_message(str(index), context_id="ctx-store")
            for index in range(4)
        ]
        await store.save(retained)
        persisted = await store.get("task-1")
        assert persisted is not None
        assert [message.parts[0].root.text for message in persisted.history] == [
            "2",
            "3",
        ]

    asyncio.run(exercise())


def test_all_active_tasks_temporarily_overshoot_then_repair(fakes):
    async def exercise():
        store = BoundedTaskStore(max_tasks=2, max_history_messages=2)
        for task_id in ("task-1", "task-2", "task-3"):
            await store.save(_task(fakes, task_id, TaskState.working))

        assert await store.size() == 3
        assert await store.get("task-1") is not None

        await store.save(_task(fakes, "task-1", TaskState.completed))
        assert await store.size() == 2
        assert await store.get("task-1") is None
        assert await store.get("task-2") is not None
        assert await store.get("task-3") is not None

    asyncio.run(exercise())
