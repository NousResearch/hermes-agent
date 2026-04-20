"""Synthetic deterministic runtime — no real LLM calls.

Given a ``(employee, task)`` it emits a plausible sequence of
:class:`ActivityEvent` items. The seed is derived from ``task.id`` so the
behavior is reproducible for tests.

Total wall-time follows the formula in design.md §4.5::

    seconds = 5 + len(task.text) // 30

…clamped to [3, 30]. Tests can pass ``time_scale=0.0`` to skip sleeps.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from datetime import datetime, timezone

from ..models import Activity, ActivityEvent, Employee, Task
from . import EventCallback, Runtime, TaskResult


def _seed_from(task_id: str) -> int:
    h = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# Vocabulary for the synthetic chatter. Plain English so kids can read it.
_TOOL_VERBS = ("web_search", "read_file", "todo", "memory", "image_generate", "execute_code")
_THOUGHTS = (
    "Let me check what's already known.",
    "Breaking this into steps.",
    "Hmm, looking deeper.",
    "Drafting a quick answer.",
    "Refining the wording.",
    "Cross-checking the result.",
)
_FINISHES = (
    "Done — sending the result.",
    "Wrapped up. Anything else?",
    "Finished — handing back to you.",
    "Task complete.",
)


class SimulatedRuntime(Runtime):
    name = "simulated"

    def __init__(self, *, time_scale: float = 1.0) -> None:
        self._time_scale = max(0.0, float(time_scale))

    async def run_task(
        self,
        employee: Employee,
        task: Task,
        on_event: EventCallback,
    ) -> TaskResult:
        rng = random.Random(_seed_from(task.id))
        total_seconds = max(3, min(30, 5 + len(task.text) // 30))
        # Pick 3-5 micro steps.
        step_count = rng.randint(3, 5)

        async def emit(kind: str, text: str, **meta) -> None:
            evt = ActivityEvent(
                ts=_now(),
                employee_id=employee.id,
                department_id=employee.department_id,
                kind=kind,                  # type: ignore[arg-type]
                text=text,
                meta=meta,
            )
            await on_event(evt)

        await emit("state_change", f"{employee.name} starts working", to=Activity.WORKING)

        for i in range(step_count):
            await asyncio.sleep(self._time_scale * total_seconds / (step_count + 1) / 2)
            tool = rng.choice(_TOOL_VERBS)
            await emit("tool_call", f"calls {tool}(...)", tool=tool)
            await asyncio.sleep(self._time_scale * total_seconds / (step_count + 1) / 2)
            await emit("tool_result", f"{tool} returned ok", tool=tool, ok=True)
            if i % 2 == 0:
                await emit("assistant", rng.choice(_THOUGHTS))

        await emit("assistant", rng.choice(_FINISHES))
        await emit("state_change", f"{employee.name} returns to rest", to=Activity.RESTING)

        # Synthetic token accounting (so the UI's $/h badge has a number).
        tokens_in = max(50, len(task.text) * 3)
        tokens_out = max(50, total_seconds * 30)

        return TaskResult(
            status="done",
            summary=f"(simulated) processed {len(task.text)} chars in ~{total_seconds}s",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
