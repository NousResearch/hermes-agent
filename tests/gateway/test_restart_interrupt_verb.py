"""The ``restart-interrupt`` control verb: an operator-consented restart skips the after-turn wait.

SIGUSR1 always waits up to ``agent.restart_after_turn_timeout`` for in-flight turns before ``stop()``.
Once ``hermes gateway restart`` has the operator's yes, the verb starts the restart without that wait
or ends one already running; ``stop()`` still owns the interrupt (and the resume handoff).
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run_restart_interrupt_verb import restart_interrupt_verb
from tests.gateway.restart_test_helpers import make_restart_runner


def _busy_runner():
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()
    runner._restart_after_turn_timeout = 300.0  # would hang the test if the wait were not skipped
    runner._running_agents["agent:main:telegram:dm:1"] = MagicMock()
    return runner


async def _ask(runner, pid):
    """Call the verb the way the control socket does: on an executor thread, off the loop."""
    return await asyncio.to_thread(restart_interrupt_verb(runner, asyncio.get_running_loop()), {"pid": pid})


@pytest.mark.asyncio
async def test_interrupt_starts_the_restart_without_the_after_turn_wait():
    runner = _busy_runner()

    answer = await _ask(runner, os.getpid())

    assert answer == {"interrupting": True, "already_restarting": False, "pid": os.getpid()}
    await asyncio.wait_for(runner._restart_task, timeout=2.0)
    runner.stop.assert_awaited_once_with(restart=True, detached_restart=False, service_restart=True)
    assert runner._running_agents, "stop() owns the interrupt; the verb does not kill the turn itself"


@pytest.mark.asyncio
async def test_interrupt_ends_an_after_turn_wait_already_running():
    runner = _busy_runner()
    waiting = asyncio.Event()
    runner._scale_to_zero_status = lambda *a, **k: waiting.set()  # published as the after-turn wait begins
    assert runner.request_restart(detached=False, via_service=True) is True  # a plain SIGUSR1 restart
    await asyncio.wait_for(waiting.wait(), timeout=2.0)
    runner.stop.assert_not_awaited()

    answer = await _ask(runner, os.getpid())

    assert answer["interrupting"] is True and answer["already_restarting"] is True
    await asyncio.wait_for(runner._restart_task, timeout=2.0)
    runner.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_consent_for_another_gateway_pid_is_declined():
    runner = _busy_runner()

    answer = await _ask(runner, os.getpid() + 1)

    assert answer == {"interrupting": False, "pid": os.getpid()}
    assert runner._draining is False and not getattr(runner, "_restart_interrupt_work", False)
