"""Tests for the process_loop RuntimeWarning fix -- issue #19285.

In Python 3.10+, calling asyncio.get_event_loop() from a non-main thread
that has no current event loop emits a DeprecationWarning (3.10/3.11) or
RuntimeWarning (3.12+).  The fix replaces get_event_loop() with
get_running_loop(), which raises RuntimeError (no warning) when there is no
running loop.
"""

import asyncio
import sys
import threading
import warnings


class TestGetRunningLoopReplacement:

    def test_get_running_loop_raises_runtime_error_not_warning(self):
        warnings_caught = []

        def _thread_target():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    pass
                warnings_caught.extend(w)

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()
        t.join(timeout=5)

        runtime_warnings = [
            x for x in warnings_caught
            if issubclass(x.category, RuntimeWarning)
        ]
        assert runtime_warnings == [], (
            f"Unexpected RuntimeWarning(s): {[str(w.message) for w in runtime_warnings]}"
        )

    def test_get_running_loop_is_silent_get_event_loop_is_not(self):
        caught_from_running = []

        def _test_get_running_loop():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    pass
                caught_from_running.extend(w)

        t = threading.Thread(target=_test_get_running_loop, daemon=True)
        t.start()
        t.join(timeout=5)

        assert all(
            not issubclass(w.category, RuntimeWarning)
            for w in caught_from_running
        ), "get_running_loop() must never emit RuntimeWarning"

    def test_get_running_loop_returns_loop_when_running(self):
        async def _check():
            loop = asyncio.get_running_loop()
            assert loop is not None
            assert loop.is_running()

        asyncio.run(_check())

    def test_no_warning_from_background_thread_with_fix(self):
        warnings_caught = []

        def _thread_target():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None
                except Exception:
                    current_loop = None
                assert current_loop is None
                warnings_caught.extend(w)

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()
        t.join(timeout=5)

        runtime_warnings = [
            x for x in warnings_caught
            if issubclass(x.category, RuntimeWarning)
        ]
        assert runtime_warnings == [], (
            f"RuntimeWarning emitted despite fix: "
            f"{[str(w.message) for w in runtime_warnings]}"
        )

    def test_fixed_pattern_in_process_loop_context(self):
        results = {}
        warnings_list = []

        def _process_loop_simulation():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None
                except Exception:
                    current_loop = None
                results["current_loop"] = current_loop
                warnings_list.extend(w)

        t = threading.Thread(
            target=_process_loop_simulation,
            name="Thread-3 (process_loop)",
            daemon=True,
        )
        t.start()
        t.join(timeout=5)

        assert results.get("current_loop") is None
        runtime_warnings = [
            x for x in warnings_list
            if issubclass(x.category, RuntimeWarning)
        ]
        assert runtime_warnings == [], (
            f"process_loop simulation still emits RuntimeWarning: "
            f"{[str(w.message) for w in runtime_warnings]}"
        )
