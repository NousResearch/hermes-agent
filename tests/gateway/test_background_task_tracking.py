"""Background tasks scheduled by the gateway must be strongly referenced.

asyncio only keeps a *weak* reference to a running task, so the result of a
bare ``asyncio.create_task(...)`` can be garbage-collected mid-flight. The
gateway's long-lived watchers (session expiry, kanban notifier/dispatcher,
platform reconnect, handoff, recovered process watchers) are scheduled via
``GatewayRunner._spawn_background_task``, which registers them in
``self._background_tasks`` so they stay alive and are cancelled on shutdown.
"""

import asyncio

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_spawn_background_task_is_tracked_then_auto_discarded():
    runner = GatewayRunner(GatewayConfig())

    started = asyncio.Event()
    release = asyncio.Event()

    async def _work():
        started.set()
        await release.wait()

    task = runner._spawn_background_task(_work())
    await started.wait()

    # While running, a strong reference is held so asyncio cannot collect it.
    assert task in runner._background_tasks

    # When it finishes, the done-callback removes it (no unbounded growth).
    release.set()
    await task
    await asyncio.sleep(0)  # allow done-callbacks to run
    assert task not in runner._background_tasks


@pytest.mark.asyncio
async def test_spawn_background_task_is_cancellable_on_shutdown():
    runner = GatewayRunner(GatewayConfig())

    async def _forever():
        while True:
            await asyncio.sleep(3600)

    task = runner._spawn_background_task(_forever())
    await asyncio.sleep(0)
    assert task in runner._background_tasks

    # Mirror the shutdown teardown that cancels every registered task.
    for tracked in list(runner._background_tasks):
        tracked.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
