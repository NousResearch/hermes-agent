"""Regression: the SIGINT/SIGTERM shutdown task must stay strongly referenced.

The gateway's signal handler cannot await, so it schedules ``stop()`` as a
task.  For a long time it did that with a bare ``asyncio.create_task(...)`` and
threw the handle away.  The event loop keeps only a weak reference to a task,
so a still-pending task can be garbage-collected mid-flight — which is exactly
why the *other* signal path (``request_restart``, SIGUSR1) holds
``self._restart_task`` and says so in a comment.

If the shutdown task is collected before ``stop()`` reaches
``self._stop_task = asyncio.create_task(_stop_impl())``, no teardown coroutine
is ever created: the gateway simply keeps running until the service manager's
stop timeout escalates to SIGKILL, so in-flight turns are never drained and
sessions are never finalized.

``GatewayRunner._schedule_shutdown_task`` owns that behaviour and is what the
signal handler calls.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner


def _make_runner_with_blocking_stop():
    """A bare runner whose ``stop()`` parks until the returned event is set."""
    runner, adapter = make_restart_runner()
    release = asyncio.Event()
    calls: list[int] = []

    async def _blocking_stop() -> None:
        calls.append(1)
        await release.wait()

    runner.stop = _blocking_stop
    runner._schedule_shutdown_task = GatewayRunner._schedule_shutdown_task.__get__(
        runner, GatewayRunner
    )
    return runner, adapter, release, calls


async def _drain(*tasks: asyncio.Task) -> None:
    for task in tasks:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_gateway_runner_declares_a_shutdown_task_slot():
    """The anchor needs a class-level default so bare runners inherit ``None``.

    Shutdown-path tests build runners via ``object.__new__`` (``stop()`` even
    carries a getattr-guard for them), so the attribute has to exist on the
    class the way ``_stop_task`` and ``_restart_task`` do.
    """
    assert GatewayRunner._stop_task is None
    assert GatewayRunner._restart_task is None
    assert GatewayRunner._shutdown_task is None


@pytest.mark.asyncio
async def test_scheduling_shutdown_anchors_the_task_on_the_runner():
    """The scheduled task is reachable from the runner, not just from the loop.

    The event loop holds only a weak reference to a task, so without this the
    shutdown can be collected while still pending.
    """
    runner, _adapter, release, calls = _make_runner_with_blocking_stop()

    task = runner._schedule_shutdown_task()
    await asyncio.sleep(0)

    assert runner._shutdown_task is task
    assert task.done() is False
    assert calls == [1]

    release.set()
    await task


@pytest.mark.asyncio
async def test_a_repeat_signal_does_not_replace_the_in_flight_shutdown_task():
    """A second signal must reuse the task that owns the live teardown.

    The handler re-enters in ordinary operation: a second Ctrl+C, a SIGINT
    followed by the service manager's SIGTERM, or ``_run_planned_stop_watcher``
    driving the same callable from its polling thread.  Re-pointing the anchor
    would drop the only strong reference to the running shutdown.
    """
    runner, _adapter, release, calls = _make_runner_with_blocking_stop()

    first = runner._schedule_shutdown_task()
    await asyncio.sleep(0)
    second = runner._schedule_shutdown_task()
    await asyncio.sleep(0)

    assert second is first
    assert runner._shutdown_task is first
    assert calls == [1], "the repeat signal started a second stop()"

    release.set()
    await first


@pytest.mark.asyncio
async def test_a_later_signal_schedules_again_once_the_previous_one_finished():
    """The guard must not wedge the gateway if an earlier shutdown completed."""
    runner, _adapter, release, calls = _make_runner_with_blocking_stop()

    first = runner._schedule_shutdown_task()
    await asyncio.sleep(0)
    release.set()
    await first

    second = runner._schedule_shutdown_task()
    await asyncio.sleep(0)

    assert second is not first
    assert runner._shutdown_task is second
    assert calls == [1, 1]

    await second


@pytest.mark.asyncio
async def test_stop_does_not_cancel_the_anchored_shutdown_task():
    """``_stop_impl``'s cancel sweep must skip ``_shutdown_task``.

    The anchored task is parked in ``await self._stop_task`` while the sweep
    runs, so cancelling it would push ``CancelledError`` into the very
    ``_stop_impl`` doing the cancelling.  This is why the shutdown task cannot
    simply be parked in ``_background_tasks`` — the same reason ``_stop_task``
    and ``_restart_task`` are already exempt.
    """
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    adapter.disconnect = AsyncMock()

    async def _park() -> None:
        await asyncio.Event().wait()

    shutdown_task = asyncio.create_task(_park())
    control_task = asyncio.create_task(_park())
    await asyncio.sleep(0)

    runner._shutdown_task = shutdown_task
    runner._background_tasks = {shutdown_task, control_task}

    try:
        with (
            patch("gateway.status.remove_pid_file"),
            patch("gateway.status.write_runtime_status"),
            patch("agent.auxiliary_client.shutdown_cached_clients"),
        ):
            await runner.stop()

        for _ in range(10):
            if control_task.done():
                break
            await asyncio.sleep(0)

        assert control_task.cancelled() is True, (
            "an ordinary background task should still be swept by _stop_impl"
        )
        assert shutdown_task.done() is False, (
            "_stop_impl cancelled the shutdown task, which is awaiting the very "
            "_stop_task that runs this sweep"
        )
    finally:
        await _drain(shutdown_task, control_task)
