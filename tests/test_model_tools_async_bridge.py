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
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Verify _run_async() keeps the event loop alive after returning."""

    def test_loop_not_closed_after_run_async(self):
        """The loop used by _run_async must still be open after the call."""
        from model_tools import _run_async

        loop = _run_async(_get_current_loop())

        assert not loop.is_closed(), (
            "_run_async() closed the event loop — cached async clients will "
            "crash with 'Event loop is closed' on GC (issue #2104)"
        )

    def test_same_loop_reused_across_calls(self):
        """Consecutive _run_async calls should reuse the same loop."""
        from model_tools import _run_async

        loop1 = _run_async(_get_current_loop())
        loop2 = _run_async(_get_current_loop())

        assert loop1 is loop2, (
            "_run_async() created a new loop on the second call — cached "
            "async clients from the first call would be orphaned"
        )

    def test_cached_transport_survives_between_calls(self):
        """A transport/future created in call 1 must be valid in call 2."""
        from model_tools import _run_async

        loop, fut = _run_async(_create_and_return_transport())

        assert not loop.is_closed()
        assert fut.result() == "ok"

        loop2 = _run_async(_get_current_loop())
        assert loop2 is loop, "Loop changed between calls"
        assert not loop.is_closed(), "Loop closed before second call"


class TestRunAsyncWorkerThread:
    """Verify worker threads get persistent per-thread loops (delegate_task fix)."""

    def test_worker_thread_loop_not_closed(self):
        """A worker thread's loop must stay open after _run_async returns,
        so cached httpx/AsyncOpenAI clients don't crash on GC."""
        from concurrent.futures import ThreadPoolExecutor
        from model_tools import _run_async

        def _run_on_worker():
            loop = _run_async(_get_current_loop())
            still_open = not loop.is_closed()
            return loop, still_open

        with ThreadPoolExecutor(max_workers=1) as pool:
            loop, still_open = pool.submit(_run_on_worker).result()

        assert still_open, (
            "Worker thread's event loop was closed after _run_async — "
            "cached async clients will crash with 'Event loop is closed'"
        )

    def test_worker_thread_reuses_loop_across_calls(self):
        """Multiple _run_async calls on the same worker thread should
        reuse the same persistent loop (not create-and-destroy each time)."""
        from concurrent.futures import ThreadPoolExecutor
        from model_tools import _run_async

        def _run_twice_on_worker():
            loop1 = _run_async(_get_current_loop())
            loop2 = _run_async(_get_current_loop())
            return loop1, loop2

        with ThreadPoolExecutor(max_workers=1) as pool:
            loop1, loop2 = pool.submit(_run_twice_on_worker).result()

        assert loop1 is loop2, (
            "Worker thread created different loops for consecutive calls — "
            "cached clients from the first call would be orphaned"
        )
        assert not loop1.is_closed()

    def test_parallel_workers_get_separate_loops(self):
        """Different worker threads must get their own loops to avoid
        contention (the original reason for the worker-thread branch)."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from model_tools import _run_async

        barrier = threading.Barrier(3, timeout=5)

        def _get_loop_id():
            # Use a barrier to force all 3 threads to be alive simultaneously,
            # ensuring the ThreadPoolExecutor actually uses 3 distinct threads.
            loop = _run_async(_get_current_loop())
            barrier.wait()
            return id(loop), not loop.is_closed(), threading.current_thread().ident

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_get_loop_id) for _ in range(3)]
            results = [f.result() for f in as_completed(futures)]

        loop_ids = {r[0] for r in results}
        thread_ids = {r[2] for r in results}
        all_open = all(r[1] for r in results)

        assert all_open, "At least one worker thread's loop was closed"
        # The barrier guarantees 3 distinct threads were used
        assert len(thread_ids) == 3, f"Expected 3 threads, got {len(thread_ids)}"
        # Each thread should have its own loop
        assert len(loop_ids) == 3, (
            f"Expected 3 distinct loops for 3 parallel workers, "
            f"got {len(loop_ids)} — workers may be contending on a shared loop"
        )

    def test_worker_loop_separate_from_main_loop(self):
        """Worker thread loops must be different from the main thread's
        persistent loop to avoid cross-thread contention."""
        from concurrent.futures import ThreadPoolExecutor
        from model_tools import _run_async, _get_tool_loop

        main_loop = _get_tool_loop()

        def _get_worker_loop_id():
            loop = _run_async(_get_current_loop())
            return id(loop)

        with ThreadPoolExecutor(max_workers=1) as pool:
            worker_loop_id = pool.submit(_get_worker_loop_id).result()

        assert worker_loop_id != id(main_loop), (
            "Worker thread used the main thread's loop — this would cause "
            "cross-thread contention on the event loop"
        )


class TestRunAsyncWithRunningLoop:
    """When a loop is already running, _run_async falls back to a thread."""

    @pytest.mark.asyncio
    async def test_run_async_from_async_context(self):
        """_run_async should still work when called from inside an
        already-running event loop (gateway / Atropos path)."""
        from model_tools import _run_async

        async def _simple():
            return 42

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_async, _simple()
        )
        assert result == 42

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
    """Simulate the full registry.dispatch('vision_analyze') chain and
    verify the event loop stays alive afterwards — the exact scenario
    from issue #2104."""

    def test_vision_dispatch_keeps_loop_alive(self, tmp_path):
        """After dispatching vision_analyze via the registry, the event
        loop must remain open so cached async clients don't crash on GC."""
        from model_tools import _run_async, _get_tool_loop
        from tools.registry import registry

        fake_response = _mock_vision_response()

        with (
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=fake_response,
            ),
            patch(
                "tools.vision_tools._download_image",
                new_callable=AsyncMock,
                side_effect=lambda url, dest, **kw: _write_fake_image(dest),
            ),
            patch(
                "tools.vision_tools._validate_image_url",
                return_value=True,
            ),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/jpeg;base64,abc",
            ),
        ):
            result_json = registry.dispatch(
                "vision_analyze",
                {"image_url": "https://example.com/cat.png", "question": "What is this?"},
            )

        result = json.loads(result_json)
        assert result.get("success") is True, f"dispatch failed: {result}"
        assert "cat" in result.get("analysis", "").lower()

        loop = _get_tool_loop()
        assert not loop.is_closed(), (
            "Event loop closed after vision_analyze dispatch — cached async "
            "clients will crash with 'Event loop is closed' (issue #2104)"
        )

    def test_two_consecutive_vision_dispatches(self, tmp_path):
        """Two back-to-back vision_analyze dispatches must both succeed
        and share the same loop (simulates 'first call fails, second
        works' from the issue report)."""
        from model_tools import _get_tool_loop
        from tools.registry import registry

        fake_response = _mock_vision_response()

        with (
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=fake_response,
            ),
            patch(
                "tools.vision_tools._download_image",
                new_callable=AsyncMock,
                side_effect=lambda url, dest, **kw: _write_fake_image(dest),
            ),
            patch(
                "tools.vision_tools._validate_image_url",
                return_value=True,
            ),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/jpeg;base64,abc",
            ),
        ):
            args = {"image_url": "https://example.com/cat.png", "question": "Describe"}

            r1 = json.loads(registry.dispatch("vision_analyze", args))
            loop_after_first = _get_tool_loop()

            r2 = json.loads(registry.dispatch("vision_analyze", args))
            loop_after_second = _get_tool_loop()

        assert r1.get("success") is True
        assert r2.get("success") is True
        assert loop_after_first is loop_after_second, "Loop changed between dispatches"
        assert not loop_after_second.is_closed()


def _write_fake_image(dest):
    """Write minimal bytes so vision_analyze_tool thinks download succeeded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"\xff\xd8\xff" + b"\x00" * 16)
    return dest
