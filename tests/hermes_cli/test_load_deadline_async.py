"""Async non-blocking plugin loading.

Regression for #130797 (multiplex gateway startup blocks the event loop in
run_with_load_deadline's threading.join, starving heartbeats).
"""
from __future__ import annotations

import asyncio
import time

import pytest

from hermes_cli.plugins_loader import (
    PluginLoadTimeout,
    arun_with_load_deadline,
)


class _Ctx:
    def __init__(self):
        self.abandoned = False

    def _abandon_load(self):
        self.abandoned = True


def test_async_loader_returns_result():
    async def _main():
        ctx = _Ctx()
        return await arun_with_load_deadline("plug", ctx, lambda: 42)

    assert asyncio.run(_main()) == 42


def test_async_loader_propagates_exception():
    async def _main():
        ctx = _Ctx()
        with pytest.raises(ValueError, match="boom"):
            await arun_with_load_deadline("plug", ctx, _raise)

    def _raise():
        raise ValueError("boom")

    asyncio.run(_main())


def test_async_loader_keeps_event_loop_responsive(monkeypatch):
    """A slow plugin load must not stall heartbeats."""
    from hermes_cli import plugins_loader

    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 5.0)

    async def _main():
        ctx = _Ctx()
        ticks: list[float] = []
        stop = asyncio.Event()

        async def _heartbeat():
            while not stop.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        def _slow():
            time.sleep(0.4)
            return "done"

        hb = asyncio.create_task(_heartbeat())
        try:
            result = await asyncio.wait_for(
                arun_with_load_deadline("slow-plug", ctx, _slow), timeout=5.0
            )
        finally:
            stop.set()
            await hb
        return result, ticks

    result, ticks = asyncio.run(_main())
    assert result == "done"
    # Heartbeat kept firing roughly every 20ms during the 400ms load.
    assert len(ticks) >= 5


def test_async_loader_timeout_abandons_ctx(monkeypatch):
    from hermes_cli import plugins_loader

    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 0.05)

    async def _main():
        ctx = _Ctx()

        def _hang():
            time.sleep(5.0)

        with pytest.raises(PluginLoadTimeout):
            await arun_with_load_deadline("hang-plug", ctx, _hang)
        return ctx

    ctx = asyncio.run(_main())
    assert ctx.abandoned is True


def test_async_timeout_prunes_dead_and_counts_one_slot(monkeypatch):
    """The timeout path prunes dead ledger entries and counts exactly one slot.

    Pre-fill the ledger with three dead loaders and one live one: after a
    timeout the dead entries must be gone, the live one kept, and the ledger
    must hold live + 1 counted slot — the cap still bounds accumulation even
    though the placeholder reads dead immediately.
    """
    import threading

    from hermes_cli import plugins_loader as pl

    monkeypatch.setattr(pl, "_resolve_plugin_load_timeout", lambda: 0.05)
    saved = list(pl._ABANDONED_LOADERS)
    stop = threading.Event()
    live = threading.Thread(target=stop.wait, daemon=True)
    live.start()
    dead = [threading.Thread(daemon=True, target=lambda: None) for _ in range(3)]
    pl._ABANDONED_LOADERS[:] = [*dead, live]
    try:

        async def _main():
            ctx = _Ctx()

            def _hang():
                time.sleep(5.0)

            with pytest.raises(PluginLoadTimeout):
                await arun_with_load_deadline("hang-plug", ctx, _hang)
            return ctx

        ctx = asyncio.run(_main())
        assert ctx.abandoned is True
        with pl._ABANDONED_LOADERS_LOCK:
            remaining = list(pl._ABANDONED_LOADERS)
        assert live in remaining
        assert len(remaining) == 2, f"live loader + exactly one counted slot, got {len(remaining)}"
        assert all(t is live or not t.is_alive() for t in remaining)
    finally:
        stop.set()
        live.join(timeout=5)
        with pl._ABANDONED_LOADERS_LOCK:
            pl._ABANDONED_LOADERS[:] = saved
