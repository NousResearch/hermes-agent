"""Tests for agent.async_utils.safe_schedule_threadsafe."""

from __future__ import annotations

import asyncio
import gc
import warnings
from concurrent.futures import Future
from unittest.mock import patch

import pytest

from agent.async_utils import safe_schedule_threadsafe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_unawaited_warnings(caught, *, coro_name: str = "") -> bool:
    """Return True if no "X was never awaited" warning slipped through.

    When *coro_name* is provided, only warnings naming that coroutine are
    counted — xdist workers may emit unrelated unawaited-coroutine warnings
    (e.g. ``AsyncMockMixin._execute_mock_call``) from concurrent tests.
    """
    bad = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "was never awaited" in str(w.message)
        and (not coro_name or coro_name in str(w.message))
    ]
    return not bad


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSafeScheduleThreadsafe:
    def test_returns_future_on_success(self):
        loop = asyncio.new_event_loop()
        try:
            import threading
            ready = threading.Event()
            stop = threading.Event()

            def _runner():
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_until_complete(_wait_for_stop(stop))

            async def _wait_for_stop(ev):
                while not ev.is_set():
                    await asyncio.sleep(0.005)

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            ready.wait(timeout=2)

            async def _sample():
                return 42

            fut = safe_schedule_threadsafe(_sample(), loop)
            assert isinstance(fut, Future)
            assert fut.result(timeout=2) == 42

            stop.set()
            t.join(timeout=2)
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            loop.close()

    def test_closed_loop_returns_none_and_closes_coroutine(self):
        loop = asyncio.new_event_loop()
        loop.close()

        async def _sample():
            return "ok"

        coro = _sample()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = safe_schedule_threadsafe(coro, loop)
            del coro
            gc.collect()

        assert result is None
        assert _no_unawaited_warnings(caught, coro_name='_sample')

    def test_none_loop_returns_none_and_closes_coroutine(self):
        async def _sample():
            return "ok"

        coro = _sample()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = safe_schedule_threadsafe(coro, None)
            del coro
            gc.collect()

        assert result is None
        assert _no_unawaited_warnings(caught, coro_name='_sample')

    def test_scheduling_exception_closes_coroutine(self):
        """If run_coroutine_threadsafe raises, close the coroutine and return None."""
        # A loop that *looks* open but raises on submission
        loop = asyncio.new_event_loop()
        try:
            async def _sample():
                return "ok"

            coro = _sample()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch(
                    "agent.async_utils.asyncio.run_coroutine_threadsafe",
                    side_effect=RuntimeError("scheduler down"),
                ):
                    result = safe_schedule_threadsafe(coro, loop)
                del coro
                gc.collect()

            assert result is None
            assert _no_unawaited_warnings(caught, coro_name='_sample')
        finally:
            loop.close()

    def test_logs_at_specified_level(self, caplog):
        import logging
        loop = asyncio.new_event_loop()
        loop.close()

        async def _sample():
            return None

        custom = logging.getLogger("test_async_utils")
        with caplog.at_level(logging.WARNING, logger="test_async_utils"):
            result = safe_schedule_threadsafe(
                _sample(), loop,
                logger=custom,
                log_message="custom-msg",
                log_level=logging.WARNING,
            )

        assert result is None
        assert any("custom-msg" in rec.message for rec in caplog.records)

    def test_non_coroutine_arg_does_not_crash(self):
        """Defensive: even if the caller hands us something weird, don't blow up."""
        loop = asyncio.new_event_loop()
        loop.close()

        # Pass a non-coroutine sentinel
        result = safe_schedule_threadsafe("not-a-coroutine", loop)  # type: ignore[arg-type]
        assert result is None
