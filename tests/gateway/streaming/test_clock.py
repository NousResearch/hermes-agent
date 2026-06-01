import pytest
from gateway.calls.native.streaming.clock import VirtualClock

pytestmark = pytest.mark.asyncio


async def test_virtualclock_starts_at_zero_and_advances():
    clock = VirtualClock()
    assert clock.now_ms() == 0
    await clock.advance(500)
    assert clock.now_ms() == 500


async def test_virtualclock_sleep_resolves_only_after_advance():
    clock = VirtualClock()
    woke = []
    async def sleeper():
        await clock.sleep(300)
        woke.append(clock.now_ms())
    import asyncio
    task = asyncio.create_task(sleeper())
    await asyncio.sleep(0)     # let the sleeper register its waiter first
    await clock.advance(299)
    assert woke == []          # not yet
    await clock.advance(1)
    await asyncio.sleep(0)     # let the sleeper resume
    assert woke == [300]
