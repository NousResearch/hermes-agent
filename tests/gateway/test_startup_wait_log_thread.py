"""Gate-release timeout warning must not run inline on the event loop (#115516).

``GatewayStartupMixin._wait_bounded_or_release`` runs right after an
``asyncio.wait`` timeout — a contended handler lock parks the loop. The
warning must be offloaded with ``asyncio.to_thread`` so logging can't
block the loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import gateway.run_startup as sut
from gateway.run_startup import GatewayStartupMixin


class _Runner(GatewayStartupMixin):
    def _late_failure_callback(self, message: str, *, level: int = logging.WARNING):
        return lambda _task: None

    def _retain_background_task(self, task: "asyncio.Task") -> "asyncio.Task":
        return task


def test_wait_timeout_warning_logged_off_loop(monkeypatch):
    """The timeout warning goes through ``asyncio.to_thread``, not inline."""
    to_thread_calls: list = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)
    warn_calls: list = []
    loop_ident: dict = {}

    def fake_warning(*a, **k):
        warn_calls.append((a, k, threading.get_ident()))

    monkeypatch.setattr(sut.logger, "warning", fake_warning)

    async def drive():
        loop_ident["ident"] = threading.get_ident()
        runner = _Runner()
        slow = asyncio.ensure_future(asyncio.sleep(10))
        try:
            done = await runner._wait_bounded_or_release(
                {slow}, 0.01, "still running after %.0fs (%d pending)", "late failure",
            )
            assert done == set()
        finally:
            slow.cancel()
            try:
                await slow
            except asyncio.CancelledError:
                pass

    asyncio.run(drive())

    assert len(to_thread_calls) == 1
    func, args, _kwargs = to_thread_calls[0]
    assert func is sut.logger.warning
    assert args[0] == "still running after %.0fs (%d pending)"
    assert args[1] == 0.01
    assert args[2] == 1
    assert len(warn_calls) == 1
    assert warn_calls[0][2] != loop_ident["ident"], "warning must not run on the event loop thread"


def test_wait_no_timeout_no_warning_thread(monkeypatch):
    """Fast tasks: no warning at all, on any thread."""
    to_thread_calls: list = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)
    warned: list = []
    monkeypatch.setattr(sut.logger, "warning", lambda *a, **k: warned.append((a, k)))

    async def drive():
        runner = _Runner()
        fast = asyncio.ensure_future(asyncio.sleep(0))
        done = await runner._wait_bounded_or_release(
            {fast}, 5, "still running after %.0fs (%d pending)", "late failure",
        )
        assert done == {fast}

    asyncio.run(drive())

    assert warned == []
    assert to_thread_calls == []
