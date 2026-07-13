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
    """When a loop is already running, _run_async uses the bridge loop."""

    @pytest.mark.asyncio
    async def test_run_async_from_async_context(self):
        """_run_async should still work when called from inside an
        already-running event loop (gateway / Atropos path)."""
        from model_tools import _run_async

        async def _simple():
            return 42

        result = _run_async(_simple())
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout_cancels_bridge_future(self, monkeypatch):
        """A timeout in the running-loop branch should cancel the bridge task."""
        import concurrent.futures
        import model_tools
        from model_tools import _run_async

        events = {
            "cancelled": False,
            "result_timeout": None,
        }

        class TimeoutFuture:
            def result(self, timeout=None):
                events["result_timeout"] = timeout
                raise concurrent.futures.TimeoutError()

            def cancel(self):
                events["cancelled"] = True
                return True

        async def _never_finishes():
            await asyncio.sleep(999)

        class DummyLoop:
            def is_closed(self):
                return False

            def call_soon_threadsafe(self, callback, *args):
                callback(*args)

        dummy_loop = DummyLoop()

        def fake_run_coroutine_threadsafe(coro, loop):
            assert loop is dummy_loop
            coro.close()
            return TimeoutFuture()

        monkeypatch.setattr(model_tools, "_get_async_bridge_loop", lambda: dummy_loop)
        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

        with pytest.raises(concurrent.futures.TimeoutError):
            _run_async(_never_finishes())

        assert events["result_timeout"] == 300
        assert events["cancelled"] is True

    @pytest.mark.asyncio
    async def test_timeout_cancels_coroutine_in_bridge_loop(self, monkeypatch):
        """On timeout, the bridge loop task must receive cancellation."""
        from model_tools import _run_async, shutdown_async_bridge_loop

        import concurrent.futures as _cf
        import time as _time

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

        t0 = _time.time()
        try:
            with pytest.raises(_cf.TimeoutError):
                _run_async(_slow_cancellable())
            elapsed = _time.time() - t0

            assert elapsed < 3.0, (
                f"_run_async blocked caller for {elapsed:.1f}s — should return "
                f"on timeout regardless of whether the coroutine has finished"
            )

            deadline = _time.time() + 5
            while not cancel_observed.is_set() and _time.time() < deadline:
                _time.sleep(0.05)
            assert cancel_observed.is_set(), (
                "Coroutine never received CancelledError — the bridge task "
                "must be cancelled when future.result(timeout=...) expires"
            )
        finally:
            shutdown_async_bridge_loop()

    @pytest.mark.asyncio
    async def test_timeout_retires_stuck_bridge_loop_for_next_call(self, monkeypatch):
        """After a stuck bridge call times out, later calls should use a fresh loop."""
        import model_tools

        from model_tools import _run_async, shutdown_async_bridge_loop

        blocker = threading.Event()
        started = threading.Event()
        stuck_loop = None
        stuck_thread = None

        async def _blocks_bridge_loop():
            nonlocal stuck_loop, stuck_thread
            stuck_loop = asyncio.get_running_loop()
            stuck_thread = threading.current_thread()
            started.set()
            blocker.wait()
            return "unblocked"

        async def _simple():
            return "fresh-ok"

        monkeypatch.setattr(model_tools, "_ASYNC_BRIDGE_TIMEOUT", 0.05)

        try:
            with pytest.raises(TimeoutError):
                _run_async(_blocks_bridge_loop())

            assert started.is_set()

            assert _run_async(_simple()) == "fresh-ok"
            assert model_tools._async_bridge_loop is not stuck_loop
        finally:
            blocker.set()
            shutdown_async_bridge_loop()
            if (
                stuck_loop is not None
                and stuck_loop is not model_tools._async_bridge_loop
                and not stuck_loop.is_closed()
            ):
                try:
                    stuck_loop.call_soon_threadsafe(stuck_loop.stop)
                except RuntimeError:
                    pass
            if stuck_thread is not None:
                stuck_thread.join(timeout=1)
            if (
                stuck_thread is not None
                and not stuck_thread.is_alive()
                and stuck_loop is not None
                and not stuck_loop.is_closed()
            ):
                stuck_loop.close()

    @pytest.mark.asyncio
    async def test_retired_bridge_loop_closes_after_thread_exits(self, monkeypatch):
        """A retired unhealthy bridge loop should close when its thread exits."""
        import time as _time
        import model_tools

        from model_tools import _run_async, shutdown_async_bridge_loop

        blocker = threading.Event()
        started = threading.Event()
        stuck_loop = None
        stuck_thread = None

        async def _blocks_bridge_loop():
            nonlocal stuck_loop, stuck_thread
            stuck_loop = asyncio.get_running_loop()
            stuck_thread = threading.current_thread()
            started.set()
            blocker.wait()

        monkeypatch.setattr(model_tools, "_ASYNC_BRIDGE_TIMEOUT", 0.05)
        monkeypatch.setattr(model_tools, "_ASYNC_BRIDGE_HEALTH_PROBE_TIMEOUT", 0.01)

        try:
            with pytest.raises(TimeoutError):
                _run_async(_blocks_bridge_loop())

            assert started.is_set()
            assert stuck_loop is not None
            assert stuck_thread is not None and stuck_thread.is_alive()
            assert model_tools._async_bridge_loop is None

            blocker.set()
            deadline = _time.monotonic() + 2
            while stuck_thread.is_alive() and _time.monotonic() < deadline:
                stuck_thread.join(timeout=0.05)

            assert not stuck_thread.is_alive()
            assert stuck_loop.is_closed(), (
                "Retired bridge loop thread exited but left its event loop open"
            )
        finally:
            blocker.set()
            shutdown_async_bridge_loop()
            if (
                stuck_thread is not None
                and stuck_thread.is_alive()
                and stuck_loop is not None
                and not stuck_loop.is_closed()
            ):
                try:
                    stuck_loop.call_soon_threadsafe(stuck_loop.stop)
                except RuntimeError:
                    pass
                stuck_thread.join(timeout=1)
            if (
                stuck_thread is not None
                and not stuck_thread.is_alive()
                and stuck_loop is not None
                and not stuck_loop.is_closed()
            ):
                stuck_loop.close()

    def test_timeout_does_not_stop_healthy_bridge_submission(self, monkeypatch):
        """A timed-out call must not stop another in-flight bridge submission."""
        import concurrent.futures as _cf
        import model_tools

        from model_tools import _run_async, shutdown_async_bridge_loop

        real_result = _cf.Future.result
        thread_timeouts = threading.local()

        def per_thread_result(self, timeout=None):
            return real_result(
                self,
                timeout=getattr(thread_timeouts, "timeout", timeout),
            )

        monkeypatch.setattr(_cf.Future, "result", per_thread_result)

        slow_started = threading.Event()
        healthy_started = threading.Event()
        slow_done = threading.Event()
        healthy_done = threading.Event()
        results = {}
        bridge_loop = None
        bridge_thread = None

        async def _slow():
            slow_started.set()
            await asyncio.sleep(60)

        async def _healthy():
            healthy_started.set()
            await asyncio.sleep(0.2)
            return "healthy-ok"

        def _run_in_async_context(name, coro_factory, timeout):
            thread_timeouts.timeout = timeout

            async def _call_run_async():
                return _run_async(coro_factory())

            try:
                results[name] = asyncio.run(_call_run_async())
            except BaseException as exc:
                results[name] = exc
            finally:
                if name == "slow":
                    slow_done.set()
                else:
                    healthy_done.set()

        slow_thread = threading.Thread(
            target=_run_in_async_context,
            args=("slow", _slow, 0.05),
            daemon=True,
            name="test-slow-bridge-caller",
        )
        healthy_thread = threading.Thread(
            target=_run_in_async_context,
            args=("healthy", _healthy, 0.4),
            daemon=True,
            name="test-healthy-bridge-caller",
        )

        try:
            slow_thread.start()
            assert slow_started.wait(timeout=1)

            healthy_thread.start()
            assert healthy_started.wait(timeout=1)
            bridge_loop = model_tools._async_bridge_loop
            bridge_thread = model_tools._async_bridge_thread

            assert slow_done.wait(timeout=1)
            assert isinstance(results["slow"], _cf.TimeoutError)

            assert healthy_done.wait(timeout=1)
            assert results["healthy"] == "healthy-ok"
        finally:
            slow_thread.join(timeout=1)
            healthy_thread.join(timeout=1)
            shutdown_async_bridge_loop(timeout=1)
            if (
                bridge_loop is not None
                and bridge_loop is not model_tools._async_bridge_loop
                and not bridge_loop.is_closed()
            ):
                try:
                    bridge_loop.call_soon_threadsafe(bridge_loop.stop)
                except RuntimeError:
                    pass
            if bridge_thread is not None:
                bridge_thread.join(timeout=1)
            if (
                bridge_thread is not None
                and not bridge_thread.is_alive()
                and bridge_loop is not None
                and not bridge_loop.is_closed()
            ):
                bridge_loop.close()

    @pytest.mark.asyncio
    async def test_async_context_reuses_persistent_bridge_loop(self):
        """Direct calls from an already-running loop should reuse one bridge loop."""
        from model_tools import _run_async

        loop1 = _run_async(_get_current_loop())
        loop2 = _run_async(_get_current_loop())

        assert loop1 is loop2, (
            "_run_async() created a new bridge loop for the second async-context "
            "call — cached async clients will accumulate across gateway turns"
        )
        assert not loop1.is_closed(), (
            "The async-context bridge loop was closed after returning — cached "
            "async clients become orphaned and leak descriptors in gateway mode"
        )

    def test_concurrent_submissions_isolate_profile_and_callbacks(self, tmp_path):
        """Shared bridge tasks must retain each caller's profile and callbacks."""
        from concurrent.futures import ThreadPoolExecutor

        import model_tools
        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.terminal_tool import (
            _get_approval_callback,
            _get_sudo_password_callback,
            set_approval_callback,
            set_sudo_password_callback,
        )

        profiles = {
            "a": tmp_path / "profile-a",
            "b": tmp_path / "profile-b",
        }
        callbacks = {
            "a": (object(), object()),
            "b": (object(), object()),
        }
        caller_barrier = threading.Barrier(2)
        bridge_barrier = asyncio.Barrier(2)

        async def _observe_context():
            def _snapshot():
                return (
                    str(get_hermes_home()),
                    _get_approval_callback(),
                    _get_sudo_password_callback(),
                    asyncio.get_running_loop(),
                    threading.get_ident(),
                )

            before = _snapshot()
            await asyncio.wait_for(bridge_barrier.wait(), timeout=2)
            await asyncio.sleep(0)
            return before, _snapshot()

        async def _driver():
            return model_tools._run_async(_observe_context())

        def _submit(key):
            approval_cb, sudo_cb = callbacks[key]
            token = set_hermes_home_override(profiles[key])
            set_approval_callback(approval_cb)
            set_sudo_password_callback(sudo_cb)
            try:
                caller_barrier.wait(timeout=2)
                observed = asyncio.run(_driver())
                caller_callbacks = (
                    _get_approval_callback(),
                    _get_sudo_password_callback(),
                )
                return observed, caller_callbacks
            finally:
                set_approval_callback(None)
                set_sudo_password_callback(None)
                reset_hermes_home_override(token)

        model_tools.shutdown_async_bridge_loop()
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                submitted = {key: pool.submit(_submit, key) for key in profiles}
                results = {key: future.result(timeout=5) for key, future in submitted.items()}

            bridge_loops = set()
            bridge_threads = set()
            for key, (observed, caller_callbacks) in results.items():
                approval_cb, sudo_cb = callbacks[key]
                expected = (str(profiles[key]), approval_cb, sudo_cb)
                before, after = observed
                assert before[:3] == expected
                assert after[:3] == expected
                assert caller_callbacks == (approval_cb, sudo_cb)
                bridge_loops.add(id(before[3]))
                bridge_threads.add(before[4])
                assert before[3] is after[3]
                assert before[4] == after[4]

            assert len(bridge_loops) == 1
            assert len(bridge_threads) == 1

            previous_approval = _get_approval_callback()
            previous_sudo = _get_sudo_password_callback()
            set_approval_callback(None)
            set_sudo_password_callback(None)

            async def _read_unbound_callbacks():
                return _get_approval_callback(), _get_sudo_password_callback()

            async def _unbound_driver():
                return model_tools._run_async(_read_unbound_callbacks())

            try:
                assert asyncio.run(_unbound_driver()) == (None, None)
            finally:
                set_approval_callback(previous_approval)
                set_sudo_password_callback(previous_sudo)
        finally:
            model_tools.shutdown_async_bridge_loop()

    def test_nested_bridge_fallback_preserves_profile_and_callbacks(self, tmp_path):
        """The one-off nested fallback must inherit the bridge task's context."""
        import model_tools
        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.terminal_tool import (
            _get_approval_callback,
            _get_sudo_password_callback,
            set_approval_callback,
            set_sudo_password_callback,
        )

        profile = tmp_path / "nested-profile"
        approval_cb = object()
        sudo_cb = object()

        async def _inner():
            return (
                str(get_hermes_home()),
                _get_approval_callback(),
                _get_sudo_password_callback(),
            )

        async def _outer():
            return model_tools._run_async(_inner())

        async def _driver():
            return model_tools._run_async(_outer())

        previous_approval = _get_approval_callback()
        previous_sudo = _get_sudo_password_callback()
        token = set_hermes_home_override(profile)
        set_approval_callback(approval_cb)
        set_sudo_password_callback(sudo_cb)
        try:
            assert asyncio.run(_driver()) == (str(profile), approval_cb, sudo_cb)
        finally:
            set_approval_callback(previous_approval)
            set_sudo_password_callback(previous_sudo)
            reset_hermes_home_override(token)
            model_tools.shutdown_async_bridge_loop()

    def test_nested_call_from_bridge_loop_uses_worker_loop(self, monkeypatch):
        """Calling _run_async from the bridge loop itself must not deadlock."""
        import concurrent.futures as _cf
        import model_tools

        from model_tools import _run_async, shutdown_async_bridge_loop

        real_result = _cf.Future.result

        def fast_bridge_thread_result(self, timeout=None):
            if (
                threading.current_thread().name == "model-tools-async-bridge"
                and timeout == 300
            ):
                return real_result(self, timeout=0.05)
            return real_result(self, timeout=timeout)

        async def _inner():
            return "nested-ok"

        async def _outer():
            return _run_async(_inner())

        monkeypatch.setattr(_cf.Future, "result", fast_bridge_thread_result)

        try:
            bridge_loop = model_tools._get_async_bridge_loop()
            future = asyncio.run_coroutine_threadsafe(_outer(), bridge_loop)

            assert future.result(timeout=2) == "nested-ok"
        finally:
            shutdown_async_bridge_loop()

    def test_nested_bridge_loop_timeout_cancels_oneoff_task(self, monkeypatch):
        """Timed-out nested bridge-loop calls must cancel the one-off task."""
        import model_tools

        from model_tools import _run_async, shutdown_async_bridge_loop

        cancel_observed = threading.Event()
        release_uncancelled = threading.Event()
        oneoff_finished = threading.Event()

        async def _inner_waits_for_cancellation():
            try:
                await asyncio.to_thread(release_uncancelled.wait)
            except asyncio.CancelledError:
                cancel_observed.set()
                raise
            finally:
                oneoff_finished.set()

        async def _outer_runs_on_bridge_loop():
            with pytest.raises(TimeoutError):
                _run_async(_inner_waits_for_cancellation())
            return cancel_observed.wait(timeout=1)

        monkeypatch.setattr(model_tools, "_ASYNC_BRIDGE_TIMEOUT", 0.05)

        try:
            bridge_loop = model_tools._get_async_bridge_loop()
            future = asyncio.run_coroutine_threadsafe(
                _outer_runs_on_bridge_loop(),
                bridge_loop,
            )

            assert future.result(timeout=2), (
                "Nested one-off coroutine never received CancelledError after "
                "the bridge-loop fallback timed out"
            )
        finally:
            release_uncancelled.set()
            oneoff_finished.wait(timeout=1)
            shutdown_async_bridge_loop()

    def test_bridge_loop_startup_failure_closes_and_resets(self, monkeypatch):
        """A failed bridge-thread startup should not publish a dead loop."""
        import model_tools

        class NeverReadyEvent:
            def set(self):
                pass

            def wait(self, timeout=None):
                return False

        class NeverStartedThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def is_alive(self):
                return False

        created_loops = []
        original_new_event_loop = asyncio.new_event_loop

        def fake_new_event_loop():
            loop = original_new_event_loop()
            created_loops.append(loop)
            return loop

        model_tools.shutdown_async_bridge_loop()
        monkeypatch.setattr(model_tools, "_async_bridge_loop", None)
        monkeypatch.setattr(model_tools, "_async_bridge_thread", None)
        monkeypatch.setattr(threading, "Event", NeverReadyEvent)
        monkeypatch.setattr(threading, "Thread", NeverStartedThread)
        monkeypatch.setattr(asyncio, "new_event_loop", fake_new_event_loop)

        with pytest.raises(RuntimeError, match="async bridge loop"):
            model_tools._get_async_bridge_loop()

        assert model_tools._async_bridge_loop is None
        assert model_tools._async_bridge_thread is None
        assert created_loops and created_loops[0].is_closed()

    def test_shutdown_async_bridge_loop_stops_thread_and_closes_loop(self):
        """The process cleanup path should stop and close the bridge loop."""
        import model_tools

        loop = model_tools._get_async_bridge_loop()
        thread = model_tools._async_bridge_thread
        assert thread is not None and thread.is_alive()

        model_tools.shutdown_async_bridge_loop()

        assert model_tools._async_bridge_loop is None
        assert model_tools._async_bridge_thread is None
        assert loop.is_closed()
        assert not thread.is_alive()


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
                "tools.vision_tools._validate_image_url_async",
                new_callable=AsyncMock,
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
                "tools.vision_tools._validate_image_url_async",
                new_callable=AsyncMock,
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
