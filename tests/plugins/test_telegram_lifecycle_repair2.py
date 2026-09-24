"""N3: stable queued drains, repeated stop and loop-close scheduling races."""
import asyncio
import importlib.util
from pathlib import Path

import pytest

from gateway import live_todo

spec = importlib.util.spec_from_file_location('repair2_lifecycle_harness', Path(__file__).with_name('test_telegram_live_todo_repair.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
manager = h.manager


@pytest.mark.asyncio
async def test_queued_drains_survive_actual_done_compaction_and_repeated_close(manager, monkeypatch):
    lanes = [h.r.new_lane(h.h.StubTelegramBot(), topic=str(i)) for i in (951, 952)]
    sources = [lane[-1] for lane in lanes]
    loop = asyncio.get_running_loop()
    callbacks, drains = [], []

    class QueuedLoop:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback):
            callbacks.append(callback)

        def create_task(self, coro):
            task = loop.create_task(coro)
            drains.append(task)
            return task

    tasks = [asyncio.create_task(source.run()) for source in sources]
    await asyncio.sleep(0)
    for source in sources:
        source.loop = QueuedLoop()
        source.unknown = True  # settled-unknown tombstone, no actual remote request
        consumer = source.consumer
        original = consumer.close
        def close(original=original):
            assert all(not s.active for s in sources), 'fence all sources before the first extension close'
            original()
        monkeypatch.setattr(consumer, 'close', close)
    registration = manager._live_todo_registration
    registration.close()
    registration.close()
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    assert callbacks
    assert all(source.loop is None and source.consumer is None for source in sources)
    assert not registration.sources
    assert all(live_todo._surfaces[source.binding.surface] is source for source in sources)
    for callback in callbacks:
        callback()  # source.loop is already None: captured scheduling reference must work
    await asyncio.gather(*drains)
    registration.close()
    assert all(not source.admitted() for source in sources)


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['already-closed', 'during-schedule'])
async def test_loop_shutdown_does_not_abort_remaining_sources(manager, phase):
    lanes = [h.r.new_lane(h.h.StubTelegramBot(), topic=str(i)) for i in (961, 962)]
    sources = tuple(manager._live_todo_registration.sources)
    tasks = [asyncio.create_task(source.run()) for source in sources]
    await asyncio.sleep(0)

    class ClosingLoop:
        closed = phase == 'already-closed'
        def is_closed(self):
            return self.closed
        def call_soon_threadsafe(self, callback):
            self.closed = True
            raise RuntimeError('Event loop is closed')

    sources[0].loop = ClosingLoop()
    manager._live_todo_registration.close()
    await asyncio.wait_for(asyncio.gather(*tasks), 5)
    assert all(not source.active and not source.admitted() for source in sources)
    assert not manager._live_todo_registration.sources
    assert all(not lane[0]._live_todo_sources for lane in lanes)
    manager._live_todo_registration.close()
