"""Regression test for Sentry NICHE-BOTS-M (CancelledError).

Root cause: BasePlatformAdapter.cancel_background_tasks() bounds its inner
asyncio.gather() with asyncio.wait_for(timeout=5.0). gateway/run.py's
_bounded_adapter_teardown() wraps that whole coroutine in an OUTER guard
(_await_adapter_cleanup_with_timeout) using the *same* 5.0s default budget.
When a background task takes ~5s+ to unwind after cancel(), the two
independent 5.0s timers race: the outer guard can cancel the coroutine while
it's suspended inside the inner asyncio.wait_for/gather, orphaning the inner
_GatheringFuture. Nobody ever calls .result()/.exception() on that future,
so asyncio's default exception handler logs "_GatheringFuture exception was
never retrieved" as an error-level CancelledError -- exactly the Sentry
NICHE-BOTS-M breadcrumb.

Fix (gateway/platforms/base.py + gateway/run.py):
  1. cancel_background_tasks() now attaches consume_detached_task_result as
     a done-callback directly on the inner gather future, so its
     exception/cancellation is always retrieved regardless of which layer
     cancels first.
  2. gateway/run.py's outer guard now sizes its cancel_background_tasks()
     timeout strictly ABOVE the adapter's own round timeout
     (CANCEL_BACKGROUND_TASKS_ROUND_TIMEOUT_SECS + buffer) instead of an
     identical constant, removing the race in the common single-round case.

This test exercises the REAL BasePlatformAdapter.cancel_background_tasks()
and gateway/run.py teardown helpers (no re-implementation), with a
background task that hangs past the adapter's round timeout, and asserts
no exception is ever reported as "never retrieved" to the event loop's
exception handler.
"""
import asyncio
import gc

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, *a, **kw):
        pass

    async def get_chat_info(self, *a):
        return {}


def _make_adapter(round_timeout: float = 0.2) -> _StubAdapter:
    config = PlatformConfig(enabled=True, token="test")
    adapter = _StubAdapter(config=config, platform=Platform.TELEGRAM)
    # Shrink the round timeout so the test runs fast while still exercising
    # the exact same code path/timeout race as production (default 5.0s).
    adapter.CANCEL_BACKGROUND_TASKS_ROUND_TIMEOUT_SECS = round_timeout
    return adapter


async def _slow_unwinding_task(hang_seconds: float):
    """Mirrors a wedged ClickClack WS send / typing-cleanup at SIGTERM:
    absorbs cancellation slowly instead of exiting immediately."""
    try:
        await asyncio.sleep(hang_seconds)
    except asyncio.CancelledError:
        await asyncio.sleep(hang_seconds)
        raise


@pytest.mark.asyncio
async def test_cancel_background_tasks_never_orphans_gather_future():
    """Even when the OUTER caller cancels us mid-drain, the inner
    _GatheringFuture's exception must always be retrieved -- no
    "exception was never retrieved" report to the loop's exception handler.
    """
    loop = asyncio.get_event_loop()
    captured = []
    loop.set_exception_handler(lambda _loop, context: captured.append(context))

    round_timeout = 0.2
    adapter = _make_adapter(round_timeout=round_timeout)
    bg_task = asyncio.ensure_future(_slow_unwinding_task(round_timeout * 3))
    adapter._background_tasks.add(bg_task)

    # Simulate gateway/run.py's outer guard cancelling cancel_background_tasks()
    # itself partway through -- at (or even before) the inner round timeout,
    # to reproduce the worst-case race directly.
    outer = asyncio.ensure_future(adapter.cancel_background_tasks())
    await asyncio.sleep(round_timeout * 0.5)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer

    # Let the loop process any pending callbacks / GC any orphaned futures --
    # this is when asyncio would normally fire "exception was never retrieved".
    for _ in range(5):
        await asyncio.sleep(0.05)
    gc.collect()
    await asyncio.sleep(0.05)

    never_retrieved = [
        c for c in captured
        if "never retrieved" in str(c.get("message", "")).lower()
    ]
    assert not never_retrieved, f"orphaned exception(s) detected: {never_retrieved}"

    # Cleanup: let the stray background task finish so pytest-asyncio
    # doesn't warn about a still-pending task at teardown.
    bg_task.cancel()
    try:
        await bg_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_gateway_outer_timeout_exceeds_adapter_round_timeout():
    """gateway/run.py must size its cancel_background_tasks() guard strictly
    above the adapter's own round timeout so they stop racing on the same
    constant (part 2 of the NICHE-BOTS-M fix)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = _make_adapter(round_timeout=5.0)

    base_timeout = 5.0
    outer_timeout = runner._adapter_cancel_background_tasks_timeout_secs(adapter, base_timeout)
    assert outer_timeout > adapter.CANCEL_BACKGROUND_TASKS_ROUND_TIMEOUT_SECS, (
        "outer guard timeout must exceed the adapter's inner round timeout, "
        "otherwise the two independent timers race on the same deadline"
    )
