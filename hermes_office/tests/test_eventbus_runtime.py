"""Tests for the asyncio EventBus and the simulated runtime."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from hermes_office.eventbus import EventBus
from hermes_office.models import Activity, ActivityEvent, Department, Employee, Task
from hermes_office.runtime.simulated import SimulatedRuntime


def _evt(text="hello", emp="emp_alpha000", dept="dept_alpha000"):
    return ActivityEvent(
        ts=datetime.now(tz=timezone.utc),
        employee_id=emp,
        department_id=dept,
        kind="assistant",
        text=text,
    )


@pytest.mark.asyncio
async def test_subscribe_publish_unsubscribe():
    bus = EventBus()
    q = await bus.subscribe()
    assert bus.subscriber_count == 1
    await bus.publish(_evt("first"))
    received = await asyncio.wait_for(q.get(), timeout=1.0)
    assert received.text == "first"
    await bus.unsubscribe(q)
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_publish_redacts_text():
    bus = EventBus()
    q = await bus.subscribe()
    await bus.publish(_evt("api_key=AbCdEfGhIj1234567890"))
    received = await asyncio.wait_for(q.get(), timeout=1.0)
    assert "AbCdEfGhIj1234567890" not in received.text
    assert "REDACTED" in received.text


@pytest.mark.asyncio
async def test_publish_drops_oldest_on_overflow():
    bus = EventBus(queue_size=4)
    q = await bus.subscribe()
    # Publish 6 events into a 4-slot queue.
    for i in range(6):
        await bus.publish(_evt(f"m{i}"))
    received = []
    while not q.empty():
        received.append((await q.get()).text)
    # Should keep the most recent ones.
    assert len(received) == 4
    assert received[-1] == "m5"


@pytest.mark.asyncio
async def test_simulated_runtime_emits_expected_kinds():
    rt = SimulatedRuntime(time_scale=0.0)        # don't sleep
    dept = Department(name="d", color="#000000")
    emp = Employee(department_id=dept.id, name="A", model="m", runtime="simulated")
    task = Task(text="hello world", department_id=dept.id, employee_id=emp.id)
    events: list[ActivityEvent] = []

    async def on_event(e: ActivityEvent) -> None:
        events.append(e)

    result = await rt.run_task(emp, task, on_event)
    assert result.status == "done"
    kinds = [e.kind for e in events]
    assert kinds[0] == "state_change"
    assert kinds[-1] == "state_change"
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "assistant" in kinds


@pytest.mark.asyncio
async def test_simulated_runtime_is_seeded():
    """Same task.id yields the same sequence of tools — required for deterministic tests."""
    rt = SimulatedRuntime(time_scale=0.0)
    dept = Department(name="d", color="#000000")
    emp = Employee(department_id=dept.id, name="A", model="m", runtime="simulated")
    task1 = Task(id="task_fixed00001", text="x", department_id=dept.id, employee_id=emp.id)
    task2 = Task(id="task_fixed00001", text="x", department_id=dept.id, employee_id=emp.id)

    async def grab(task):
        out = []
        async def on(e):
            out.append((e.kind, e.text))
        await rt.run_task(emp, task, on)
        return out

    a = await grab(task1)
    b = await grab(task2)
    assert a == b
