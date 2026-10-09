"""A Stop requested long after hello still receives its full acknowledgement window: only the
first actual Stop opens the budget, and repeated Stops share that one deadline (real child)."""
import asyncio
import subprocess
import sys
import time

import pytest

from hermes_state_runtime import RuntimeStoreError

# Harmless child: says hello, then on Stop either acknowledges (finished) or keeps chattering.
CHILD = """import json, sys, threading, time
chatty = sys.argv[1] == 'chatty'
def out(frame):
    sys.stdout.buffer.write(json.dumps(frame).encode() + b'\\n')
    sys.stdout.buffer.flush()
def chatter():
    while True:
        out({'type': 'delta', 'text': 'x'})
        time.sleep(.05)
out({'type': 'hello'})
for line in sys.stdin.buffer:
    if json.loads(line) == {'type': 'stop'}:
        if chatty:
            threading.Thread(target=chatter, daemon=True).start()
        else:
            out({'type': 'finished'})
"""
ACK = 0.6


def _run(mode, supervise):
    from gateway.session_managed_worker import HELLO_SECONDS, ManagedWorker
    child = subprocess.Popen([sys.executable, '-c', CHILD, mode], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    worker = ManagedWorker(child)

    async def main():
        async with asyncio.timeout(15):
            assert await worker.next_frame(HELLO_SECONDS, ack=0) == {'type': 'hello'}
            worker.writer.start()
            await asyncio.sleep(ACK + 0.4)  # the turn outlives the ack interval before any Stop
            return await supervise(worker)

    try:
        return asyncio.run(main())
    finally:
        worker.close()


@pytest.mark.platforms("linux")
def test_late_stop_gets_its_full_ack_window():
    async def supervise(worker):
        worker.control({'type': 'stop'})
        return await worker.next_frame(None, ack=ACK)

    assert _run('ack', supervise) == {'type': 'finished'}


@pytest.mark.platforms("linux")
def test_repeated_late_stop_shares_the_first_stops_deadline():
    async def supervise(worker):
        worker.control({'type': 'stop'})
        first = time.monotonic()
        frames = []
        with pytest.raises(RuntimeStoreError) as stopped:
            while True:
                frames.append(await worker.next_frame(None, ack=ACK))
                if len(frames) == 3:
                    worker.control({'type': 'stop'})  # must not renew the window
        return stopped.value.reason, time.monotonic() - first, frames

    reason, elapsed, frames = _run('chatty', supervise)
    assert reason == 'managed_worker_stopped'
    assert len(frames) >= 3, frames  # the late Stop did get a live window
    assert ACK - 0.1 < elapsed < ACK + 1.0, elapsed
