"""In-place Telegram polling recovery replays finals refused with ``send_path_degraded``."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import telegram_send_path_recovery as recovery
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Runner:
    """The runner seam the adapter calls: the runtime-ledger sweep for one bot identity."""

    def __init__(self, gate=None):
        self.sweeps = []
        self.gate = gate

    async def _redeliver_failed_obligations_for_platform(self, platform, *, profile=None):
        self.sweeps.append((platform.value, profile))
        if self.gate is not None:
            await self.gate.wait()
        return 0


def _degraded_adapter(runner) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter.gateway_runner = runner
    adapter._running = True
    adapter._owner_profile = "work"
    generation, _progress = adapter._begin_polling_generation()
    assert adapter.send_path_degraded
    return adapter, generation


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    monkeypatch.setattr(recovery, "_SWEEP_SETTLE_SECONDS", 0)


async def _drain(adapter):
    task = getattr(adapter, "_send_path_recovery_task", None)
    if task is not None:
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_in_place_recovery_sweeps_the_adapters_own_identity_once():
    runner = _Runner()
    adapter, generation = _degraded_adapter(runner)

    assert adapter._record_polling_progress(generation)
    await _drain(adapter)
    assert not adapter.send_path_degraded
    assert runner.sweeps == [("telegram", "work")]

    # Progress on an already-healthy send path is not a recovery edge: no further sweeps.
    assert adapter._record_polling_progress(generation)
    await asyncio.sleep(0)
    await _drain(adapter)
    assert runner.sweeps == [("telegram", "work")]


@pytest.mark.asyncio
async def test_an_edge_during_a_running_sweep_coalesces_into_one_rerun():
    gate = asyncio.Event()
    runner = _Runner(gate=gate)
    adapter, generation = _degraded_adapter(runner)

    assert adapter._record_polling_progress(generation)
    for _ in range(20):
        if runner.sweeps:
            break
        await asyncio.sleep(0)
    assert runner.sweeps == [("telegram", "work")]
    # Two more degraded -> healthy flaps while the first sweep is still sending.
    for _ in range(2):
        generation, _progress = adapter._begin_polling_generation()
        assert adapter._record_polling_progress(generation)
    gate.set()
    await _drain(adapter)
    assert runner.sweeps == [("telegram", "work"), ("telegram", "work")]
