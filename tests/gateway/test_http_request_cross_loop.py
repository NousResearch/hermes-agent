"""Regression test for the cross-loop TimerContext bug in http_request().

When an aiohttp.ClientSession is created on one event loop but used from
a different loop, aiohttp's ``TimerContext.__enter__`` raises::

    RuntimeError: Timeout context manager should be used inside a task

because it calls ``asyncio.current_task(loop=session_loop)`` which returns
``None`` when the current task is on a different loop.

http_request() detects this and routes the full request through the
session's own loop via :func:`asyncio.run_coroutine_threadsafe`.
"""

import asyncio
import threading

import pytest


class TestHttpRequestCrossLoop:
    """http_request() must not raise TimerContext when the session is on
    a different event loop than the caller."""

    def test_real_cross_loop_session_does_not_raise_timer_context(self):
        """Create a real aiohttp.ClientSession on a foreign thread's
        event loop, then call http_request from the main thread and
        verify the TimerContext error does NOT occur.
        """
        import aiohttp
        from gateway.platforms.base import http_request

        session_holder = {}
        loop_holder = {}

        def _session_thread(ready):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _setup():
                session = aiohttp.ClientSession(timeout=None)
                session_holder["session"] = session
                loop_holder["loop"] = loop
                ready.set()

            loop.run_until_complete(_setup())
            loop.run_forever()

        ready = threading.Event()
        t = threading.Thread(target=_session_thread, args=(ready,), daemon=True)
        t.start()
        ready.wait()

        session = session_holder["session"]
        session_loop = loop_holder["loop"]

        try:
            from model_tools import _run_async

            async def _call():
                # Connection-refused is fine — the point is no TimerContext.
                try:
                    await http_request(
                        session, "get",
                        "http://127.0.0.1:1/__probe",
                        timeout=2,
                    )
                except asyncio.TimeoutError:
                    return {"timeout": True}
                except Exception as e:
                    return {"error": str(e)}
                return {"ok": True}

            result = _run_async(_call())
            err = result.get("error", "")
            assert "Timeout context manager" not in err, (
                f"Cross-loop TimerContext error triggered: {err}"
            )
            assert "TimerContext" not in err, (
                f"TimerContext error triggered: {err}"
            )
        finally:
            # Stop the foreign loop and close the session from the main
            # thread's loop (the session's loop is being torn down).
            session_loop.call_soon_threadsafe(session_loop.stop)
            t.join(timeout=5)
