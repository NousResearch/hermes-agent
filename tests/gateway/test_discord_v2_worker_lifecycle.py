from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


class _FakeStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeWorker:
    def __init__(self) -> None:
        self.stop_called = False
        self.cancel_observed = False
        self.store = _FakeStore()

    async def run_forever(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_observed = True
            raise

    def stop(self) -> None:
        self.stop_called = True


def _runner(enabled: bool, mode: str) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        discord_native_multibot=SimpleNamespace(enabled=enabled, mode=mode),
    )
    runner._discord_v2_worker = None
    runner._discord_v2_worker_task = None
    runner._discord_v2_worker_owns_store = False
    runner._discord_v2_outbox_task = None
    runner._discord_v2_outbox_stop_event = None
    runner._background_tasks = set()
    runner.adapters = {}
    return runner


@pytest.mark.parametrize(
    ("enabled", "mode"),
    [
        (False, "off"),
        (False, "listen_only"),
        (True, "off"),
        (True, "listen_only"),
        (True, "shadow"),
    ],
)
def test_discord_v2_worker_not_started_when_disabled_or_off(monkeypatch, enabled, mode):
    runner = _runner(enabled, mode)
    built = []

    def fake_builder(self):
        if not self._discord_v2_worker_required():
            return None
        built.append(True)
        return _FakeWorker()

    monkeypatch.setattr(GatewayRunner, "_build_discord_v2_worker", fake_builder)

    runner._start_discord_v2_worker_if_needed()

    assert built == []
    assert runner._discord_v2_worker is None
    assert runner._discord_v2_worker_task is None


@pytest.mark.asyncio
async def test_discord_v2_worker_starts_and_stops_only_for_active_mode(monkeypatch):
    runner = _runner(True, "active")
    worker = _FakeWorker()

    def fake_builder(self):
        assert self._discord_v2_worker_required() is True
        self._discord_v2_worker_owns_store = True
        return worker

    monkeypatch.setattr(GatewayRunner, "_build_discord_v2_worker", fake_builder)

    runner._start_discord_v2_worker_if_needed()
    task = runner._discord_v2_worker_task
    await asyncio.sleep(0)

    assert runner._discord_v2_worker is worker
    assert task is not None
    assert not task.done()
    assert task in runner._background_tasks

    await runner._stop_discord_v2_worker()

    assert worker.stop_called is True
    assert worker.cancel_observed is True
    assert task.done()
    assert runner._discord_v2_worker is None
    assert runner._discord_v2_worker_task is None
    assert runner._discord_v2_worker_owns_store is False
    assert worker.store.closed is True


@pytest.mark.asyncio
async def test_discord_v2_active_runtime_starts_worker_and_outbox_loop(monkeypatch):
    from gateway.config import Platform

    runner = _runner(True, "active")
    worker = _FakeWorker()
    outbox_calls = 0

    class Adapter:
        async def run_outbox_once(self):
            nonlocal outbox_calls
            outbox_calls += 1
            await asyncio.sleep(0)
            return None

    runner.adapters = {Platform.DISCORD: Adapter()}

    def fake_builder(self):
        assert self._discord_v2_worker_required() is True
        return worker

    monkeypatch.setattr(GatewayRunner, "_build_discord_v2_worker", fake_builder)

    runner._start_discord_v2_worker_if_needed()
    runner._start_discord_v2_outbox_loop_if_needed()
    worker_task = runner._discord_v2_worker_task
    outbox_task = runner._discord_v2_outbox_task
    await asyncio.sleep(0.02)

    assert worker_task is not None
    assert outbox_task is not None
    assert not worker_task.done()
    assert outbox_calls >= 1
    assert worker_task in runner._background_tasks
    assert outbox_task in runner._background_tasks

    await runner._stop_discord_v2_worker()
    await runner._stop_discord_v2_outbox_loop()

    assert worker.stop_called is True
    assert worker.cancel_observed is True
    assert runner._discord_v2_worker_task is None
    assert runner._discord_v2_outbox_task is None


@pytest.mark.asyncio
async def test_discord_v2_outbox_loop_runs_only_in_active_mode():
    from gateway.config import Platform

    runner = _runner(True, "active")
    calls = 0

    class Adapter:
        async def run_outbox_once(self):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return None

    runner.adapters = {Platform.DISCORD: Adapter()}

    runner._start_discord_v2_outbox_loop_if_needed()
    task = runner._discord_v2_outbox_task
    await asyncio.sleep(0.02)
    await runner._stop_discord_v2_outbox_loop()

    assert task is not None
    assert task.done()
    assert calls >= 1
    assert runner._discord_v2_outbox_task is None


def test_discord_v2_outbox_loop_noops_outside_active_mode():
    for mode in ("off", "shadow", "listen_only"):
        runner = _runner(True, mode)
        runner._start_discord_v2_outbox_loop_if_needed()
        assert runner._discord_v2_outbox_task is None
