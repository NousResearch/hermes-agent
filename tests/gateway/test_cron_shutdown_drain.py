"""Regression tests for #58818.

On restart the gateway must drain an in-flight cron delivery instead of
dropping it. A cron delivery is a coroutine scheduled onto the gateway event
loop (``safe_schedule_threadsafe``) while the ticker thread blocks on its
future. The shutdown wait therefore must NOT block the loop with a synchronous
``thread.join()`` — doing so deadlocks the delivery (the loop can never run it)
and the message is silently lost. ``_await_thread_exit`` waits cooperatively so
the pending delivery completes first.
"""
import asyncio
import threading

import pytest

import gateway.run as gateway_run


@pytest.mark.asyncio
async def test_await_thread_exit_lets_loop_scheduled_delivery_complete():
    # Reproduces the drop: the worker schedules a coroutine onto THIS loop and
    # blocks on its result, exactly like cron/_deliver_result. A blocking join
    # would deadlock it; the cooperative wait lets it finish.
    loop = asyncio.get_running_loop()
    delivered = threading.Event()
    worker_done = threading.Event()

    async def _delivery():
        await asyncio.sleep(0.05)
        delivered.set()
        return "ok"

    def _cron_worker():
        fut = asyncio.run_coroutine_threadsafe(_delivery(), loop)
        fut.result(timeout=10)
        worker_done.set()

    thread = threading.Thread(target=_cron_worker, daemon=True)
    thread.start()

    exited = await gateway_run._await_thread_exit(thread, timeout=10)

    assert exited is True
    assert delivered.is_set(), "in-flight delivery coroutine never ran (loop was blocked)"
    assert worker_done.is_set()


@pytest.mark.asyncio
async def test_await_thread_exit_returns_false_on_timeout():
    keep_alive = threading.Event()

    def _spin():
        keep_alive.wait(5)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        exited = await gateway_run._await_thread_exit(thread, timeout=0.2, poll=0.02)
        assert exited is False
        assert thread.is_alive()
    finally:
        keep_alive.set()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_await_thread_exit_handles_none_and_dead_thread():
    assert await gateway_run._await_thread_exit(None, timeout=1) is True

    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()
    thread.join(timeout=2)
    assert await gateway_run._await_thread_exit(thread, timeout=1) is True
