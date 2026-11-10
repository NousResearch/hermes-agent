"""Mode changes must not abandon accepted native parallel work."""
import asyncio

import pytest

from tests.gateway.test_parallel_native_ingress import admissions, event, setup_owner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_old_work_reply_keeps_its_route_after_switch_to_queue(monkeypatch):
    gate = asyncio.Event()
    entered = []

    async def produce(authority, ref, row):
        entered.append((row['payload']['text'], ref.session_id))
        if row['payload']['text'] == 'first':
            await gate.wait()
        return row['payload']['text'].upper()

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        await adapter.handle_message(event(adapter, 'old', 'first'))
        await async_wait_until(lambda: len(entered) == 1, what='old work running')
        monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: 'queue')
        await adapter.handle_message(event(adapter, 'continuation', 'follow', reply='old'))
        await async_wait_until(lambda: len(admissions(authority)) == 2, what='reply admitted')
        assert len(entered) == 1
        assert len({row['target_session_id'] for row in admissions(authority)}) == 1
        gate.set()
        await async_wait_until(lambda: 'FOLLOW' in adapter.sent, what='same work continued')
        assert entered[0][1] == entered[1][1]
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='cleanup')
        authority.db.close()
