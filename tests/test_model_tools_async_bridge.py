"""Regression tests for the _run_async() event-loop lifecycle.

These tests verify the fix for GitHub issue #2104:
  "Event loop is closed" after vision_analyze used as first call in session.

Root cause: asyncio.run() creates and *closes* a fresh event loop on every
call.  Cached httpx/AsyncOpenAI clients that were bound to the now-dead loop
would crash with RuntimeError("Event loop is closed") when garbage-collected.

The fix replaces asyncio.run() with a persistent event loop in _run_async().
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_current_loop():
    """Return the running event loop from inside a coroutine."""
    return asyncio.get_event_loop()


async def _create_and_return_transport():
    """Simulate an async client creating a transport on the current loop.

    Returns a simple asyncio.Future bound to the running loop so we can
    later check whether the loop is still alive.
    """
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    fut.set_result("ok")
    return loop, fut


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunAsyncLoopLifecycle:
    def test_cached_transport_survives_between_calls(self):
        """A transport/future created in call 1 must be valid in call 2."""
        from model_tools import _run_async

        loop, fut = _run_async(_create_and_return_transport())

        assert not loop.is_closed()
        assert fut.result() == "ok"

        loop2 = _run_async(_get_current_loop())
        assert loop2 is loop, "Loop changed between calls"
        assert not loop.is_closed(), "Loop closed before second call"


def test_concurrent_workers_reuse_distinct_loops():
    from concurrent.futures import ThreadPoolExecutor
    from model_tools import _run_async

    main = _run_async(_get_current_loop())
    barrier = threading.Barrier(3, timeout=10)

    def worker():
        loop, future = _run_async(_create_and_return_transport())
        barrier.wait()
        assert _run_async(_get_current_loop()) is loop
        assert not loop.is_closed() and future.result() == "ok"
        return loop, threading.get_ident()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker) for _ in range(3)]
        results = [future.result(timeout=15) for future in futures]
    assert len({thread for _, thread in results}) == 3
    assert len({main, *(loop for loop, _ in results)}) == 4


class TestRunAsyncWithRunningLoop:
    """When a loop is already running, _run_async falls back to a thread."""

    @pytest.mark.asyncio
    async def test_run_async_from_async_context(self):
        """_run_async should still work when called from inside an
        already-running event loop (gateway / Atropos path)."""
        from model_tools import _run_async

        async def _simple():
            return 42

        assert _run_async(_simple()) == 42
        loop = _run_async(_get_current_loop())
        assert loop is not asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_timeout_uses_nonblocking_executor_shutdown(self, monkeypatch):
        """A timeout in the running-loop branch must not block the caller.

        If shutdown ever waits for a stuck worker, a tool coroutine that
        ignores (or can't observe) cancellation would hang the whole agent.
        Guard: the caller must raise TimeoutError and pool.shutdown must be
        called with wait=False. The worker's own event loop handles cleanup
        (cancellation is scheduled via call_soon_threadsafe before the
        caller returns).
        """
        import concurrent.futures
        from model_tools import _run_async

        events = {
            "result_timeout": None,
            "shutdown_calls": [],
            "submitted_fn": None,
        }

        class TimeoutFuture:
            def result(self, timeout=None):
                events["result_timeout"] = timeout
                raise concurrent.futures.TimeoutError()

            def cancel(self):
                return True

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.shutdown(wait=True)
                return False

            def submit(self, fn, *args, **kwargs):
                # Record which function got submitted -- should be the
                # in-function worker wrapper, not bare asyncio.run, so we
                # know _run_async is using a loop it owns and can cancel.
                events["submitted_fn"] = getattr(fn, "__name__", repr(fn))
                return TimeoutFuture()

            def shutdown(self, wait=True, cancel_futures=False):
                events["shutdown_calls"].append((wait, cancel_futures))

        async def _never_finishes():
            await asyncio.sleep(999)

        monkeypatch.setattr(
            concurrent.futures,
            "ThreadPoolExecutor",
            FakeExecutor,
        )

        with pytest.raises(concurrent.futures.TimeoutError):
            _run_async(_never_finishes())

        assert events["result_timeout"] == 300
        # The worker wrapper creates its own event loop so _run_async can
        # cancel the task on timeout — this must NOT be bare asyncio.run.
        assert events["submitted_fn"] != "run", (
            "_run_async submitted asyncio.run directly — it must submit a "
            "worker wrapper that owns the event loop so timeouts can cancel "
            "the task"
        )
        # Critical: shutdown must NOT wait. If wait=True, a stuck coroutine
        # would freeze the caller (converts a thread leak into a hang).
        assert events["shutdown_calls"], "shutdown was never called"
        for wait, _cancel in events["shutdown_calls"]:
            assert wait is False, (
                f"shutdown called with wait={wait} — a stuck tool coroutine "
                f"would hang the caller indefinitely"
            )

    @pytest.mark.asyncio
    async def test_timeout_cancels_coroutine_in_worker_loop(self, monkeypatch):
        """On timeout, the worker's event loop must receive a cancel request
        so the coroutine stops and the thread exits — not leaked.

        Before the fix, future.cancel() on a running ThreadPoolExecutor
        future is a no-op, so the worker thread kept running the coroutine
        to completion (leaking one thread per tool-timeout).
        """
        from model_tools import _run_async

        # Shrink the 300s internal timeout by patching future.result.
        # We do this surgically: let everything else run for real so the
        # worker loop actually exists and can observe cancellation.
        import concurrent.futures as _cf

        real_pool_cls = _cf.ThreadPoolExecutor

        class FastTimeoutPool(real_pool_cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)

        # Patch future.result to time out after 1s instead of 300s.
        real_result = _cf.Future.result

        def fast_result(self, timeout=None):
            return real_result(self, timeout=1.0 if timeout == 300 else timeout)

        monkeypatch.setattr(_cf.Future, "result", fast_result)

        cancel_observed = threading.Event()

        async def _slow_cancellable():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_observed.set()
                raise

        import time as _time
        t0 = _time.time()
        with pytest.raises(_cf.TimeoutError):
            _run_async(_slow_cancellable())
        elapsed = _time.time() - t0

        # Caller must return fast (no hang waiting for the coro).
        assert elapsed < 3.0, (
            f"_run_async blocked caller for {elapsed:.1f}s — should return "
            f"on timeout regardless of whether the coroutine has finished"
        )

        # Worker thread must cancel the task (not leak).
        deadline = _time.time() + 5
        while not cancel_observed.is_set() and _time.time() < deadline:
            _time.sleep(0.05)
        assert cancel_observed.is_set(), (
            "Coroutine never received CancelledError — worker thread leaked "
            "(ThreadPoolExecutor.cancel() is a no-op on a running future; "
            "_run_async must cancel the task inside its worker loop)"
        )


# ---------------------------------------------------------------------------
# Integration: full vision_analyze dispatch chain
# ---------------------------------------------------------------------------

def _mock_vision_response():
    """Build a fake LLM response matching async_call_llm's return shape."""
    message = SimpleNamespace(content="A cat sitting on a chair.")
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/vision", usage=None)


class TestVisionDispatchLoopSafety:
    def test_consecutive_image_dispatches_keep_the_loop_alive(self):
        import base64
        import io
        from PIL import Image
        from model_tools import _get_tool_loop
        from tools.registry import registry

        image = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image, format="PNG")
        args = {"image_url": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode(),
                "question": "Describe"}
        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock,
                   return_value=_mock_vision_response()):
            first = json.loads(registry.dispatch("vision_analyze", args))
            loop = _get_tool_loop()
            second = json.loads(registry.dispatch("vision_analyze", args))
        assert first.get("success") is True, first
        assert second.get("success") is True, second
        assert "cat" in first["analysis"].lower()
        assert _get_tool_loop() is loop and not loop.is_closed()
