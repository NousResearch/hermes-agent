"""Worker teardown must cancel background tasks and join their executor threads."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest


@pytest.mark.parametrize("mode", ["persistent", "running-loop"])
def test_worker_exit_drains_background_task_and_default_executor(mode):
    from model_tools import _run_async

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    cleaned = threading.Event()
    threads = []

    def blocking_work():
        threads.append(threading.current_thread())
        started.set()
        try:
            assert release.wait(timeout=15)
        finally:
            finished.set()

    async def background():
        try:
            await asyncio.to_thread(blocking_work)
        finally:
            release.set()
            cleaned.set()

    async def operation():
        task = asyncio.create_task(background())
        await asyncio.sleep(0)
        assert started.wait(timeout=5)
        return asyncio.get_running_loop(), task

    async def running_caller():
        return _run_async(operation())

    loop = None
    try:
        if mode == "persistent":
            with ThreadPoolExecutor(max_workers=1) as pool:
                loop, task = pool.submit(_run_async, operation()).result(timeout=10)
        else:
            loop, task = asyncio.run(running_caller())

        assert (
            loop.is_closed(),
            task.done(),
            task.cancelled(),
            cleaned.is_set(),
            finished.is_set(),
            any(thread.is_alive() for thread in threads),
        ) == (True, True, True, True, True, False)
    finally:
        release.set()
        if loop is not None and not loop.is_closed():

            async def cleanup():
                pending = asyncio.all_tasks()
                pending.discard(asyncio.current_task())
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                await asyncio.get_running_loop().shutdown_default_executor()

            loop.run_until_complete(cleanup())
            loop.close()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
