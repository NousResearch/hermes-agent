"""Managed-worker pipe reads never occupy the loop's shared default executor: with as many
parked (silent) workers as the pool has threads, an unrelated ``asyncio.to_thread`` still runs,
and a cancelled read never leaves an orphaned reader that swallows the child's next frame."""
import asyncio
import concurrent.futures
import subprocess
import sys

import pytest

SILENT = 'import sys; sys.stdin.buffer.read()'


def _spawn(code):
    return subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


@pytest.mark.platforms("linux")
def test_parked_worker_reads_leave_the_default_executor_free():
    from gateway.session_managed_worker import ManagedWorker
    workers = [ManagedWorker(_spawn(SILENT)) for _ in range(4)]

    async def scenario():
        asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=4))
        parked = [asyncio.create_task(w.next_frame(None, ack=30)) for w in workers]
        try:
            await asyncio.sleep(.3)
            # Settlement, env preparation and control RPCs all run on this pool.
            return await asyncio.wait_for(asyncio.to_thread(lambda: 'answered'), 3)
        finally:
            for task in parked:
                task.cancel()
            for worker in workers:  # EOF releases any reader still blocked on the pipe
                worker.close()
            await asyncio.gather(*parked, return_exceptions=True)

    assert asyncio.run(scenario()) == 'answered'
    assert all(w.process.poll() is not None for w in workers)
    # close() never leaks a reader blocked on readline: EOF ends it and it closes the pipe.
    assert all(not w.reader.is_alive() and w.process.stdout.closed for w in workers)


@pytest.mark.platforms("linux")
def test_a_frame_arriving_after_a_cancelled_read_is_not_lost():
    from gateway.session_managed_worker import ManagedWorker
    worker = ManagedWorker(_spawn('import sys\nsys.stdin.buffer.readline()\n'
                                  'sys.stdout.buffer.write(b\'{"type":"delta","text":"kept"}\\n\'); sys.stdout.flush()\n'
                                  'sys.stdin.buffer.read()'))

    async def scenario():
        try:
            # A read cancelled while the child is silent (turn cancellation, Stop race).
            first = asyncio.create_task(worker.next_frame(None, ack=30))
            await asyncio.sleep(.3)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            worker.send({'type': 'go'})
            return await asyncio.wait_for(worker.next_frame(None, ack=30), 3)
        finally:
            worker.close()

    assert asyncio.run(scenario()) == {'type': 'delta', 'text': 'kept'}
