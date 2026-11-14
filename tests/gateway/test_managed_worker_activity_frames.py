"""A managed turn running a quiet tool sends no turn frame, yet its agent keeps recording activity
inside the worker. The restart wait reads a worker silent past ``agent.gateway_timeout`` as wedged
and stops waiting for it, so the worker forwards its agent's own activity clock to the owner: a
live quiet turn still holds the restart, a stalled one does not."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until

ROOT = Path(__file__).resolve().parents[2]
# A worker whose agent is inside one long, quiet tool: no turn frame, only its own activity clock,
# which keeps ticking (ACTIVE=1) or stopped when the tool began (ACTIVE=0).
CHILD = """
import os, sys, threading, time
from agent.managed_worker import WorkerChannel, forward_activity
from agent.session_activity import build_activity_snapshot
began = time.time()
class Agent:
    def get_activity_summary(self):
        return build_activity_snapshot(last_activity_at=time.time() if os.environ['ACTIVE'] == '1' else began,
                                       last_activity_description='sequential tool running')
channel = WorkerChannel(os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0))
forward_activity(Agent(), channel, threading.Event(), interval=0.2)
"""


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("active", [True, False])
def test_a_quiet_tool_holds_the_restart_only_while_its_agent_is_active(active):
    from gateway.session_managed_worker import ManagedWorker
    from gateway.session_runtime_workers import wedged_managed_turn_count
    process = subprocess.Popen([sys.executable, '-c', CHILD], cwd=ROOT, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               env={**os.environ, 'ACTIVE': '1' if active else '0'})
    worker = ManagedWorker(process)
    runner = SimpleNamespace(session_authority=SimpleNamespace(_managed_workers={'s1': worker}, sessions={}))

    async def scenario():
        turn = asyncio.create_task(worker.next_frame(None, ack=30))  # the owner's turn loop, parked
        try:
            # Several gateway timeouts with no turn frame at all.
            await async_wait_until(lambda: worker.reader is not None and worker.reader.is_alive(), timeout=10)
            started = asyncio.get_running_loop().time()
            await async_wait_until(lambda: asyncio.get_running_loop().time() - started >= 3, timeout=10)
            assert not turn.done(), 'the activity frame reached the turn loop'
            return wedged_managed_turn_count(runner, timeout=1.0)
        finally:
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)

    try:
        assert asyncio.run(scenario()) == (0 if active else 1)
    finally:
        worker.close()
