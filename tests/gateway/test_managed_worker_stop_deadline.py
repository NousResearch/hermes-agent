"""Stop bounds managed-worker supervision by ONE deadline from the first Stop: a child that keeps
emitting valid frames after Stop is escalated on the same budget as a silent one (real child)."""
import asyncio
import subprocess
import sys
import time

import pytest

from hermes_state_runtime import RuntimeStoreError

# Harmless child: on Stop it keeps streaming valid deltas and never sends its result.
CHATTY = """import json, sys, threading, time
def chatter():
    while True:
        sys.stdout.buffer.write(json.dumps({'type': 'delta', 'text': 'x'}).encode() + b'\\n')
        sys.stdout.buffer.flush()
        time.sleep(.05)
for line in sys.stdin.buffer:
    if json.loads(line) == {'type': 'stop'}:
        threading.Thread(target=chatter, daemon=True).start()
"""


@pytest.mark.platforms("linux")
def test_valid_output_after_stop_does_not_renew_the_ack_budget():
    from gateway.session_managed_worker import ManagedWorker
    child = subprocess.Popen([sys.executable, '-c', CHATTY], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    worker = ManagedWorker(child)
    worker.writer.start()
    ack = 1.5

    async def supervise():
        frames = []
        worker.control({'type': 'stop'})
        started = time.monotonic()
        async with asyncio.timeout(10):
            with pytest.raises(RuntimeStoreError) as stopped:
                while True:
                    frames.append(await worker.next_frame(None, ack=ack))
                    if len(frames) == 5:
                        worker.control({'type': 'stop'})  # a repeated Stop must not extend it either
        return stopped.value.reason, time.monotonic() - started, frames

    try:
        reason, elapsed, frames = asyncio.run(supervise())
    finally:
        worker.close()
    assert reason == 'managed_worker_stopped'
    assert len(frames) >= 5 and all(f == {'type': 'delta', 'text': 'x'} for f in frames)
    assert elapsed < ack + 1.0, elapsed
    assert child.poll() is not None
