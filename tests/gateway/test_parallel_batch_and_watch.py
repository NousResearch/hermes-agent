"""Private executed adversarial probes for the frozen implementation."""
import asyncio

import pytest

from gateway.session_parallel import conversation_scope
from hermes_state_parallel import by_anchor
from tests.gateway.test_parallel_native_ingress import admissions, event, setup_owner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_discord_text_batch_does_not_merge_replies_to_distinct_work(monkeypatch):
    gate = asyncio.Event()
    entered = []

    async def produce(authority, ref, row):
        entered.append((row['payload']['text'], ref.session_id))
        if row['payload']['text'] in ('work one', 'work two'):
            await gate.wait()
        return row['payload']['text'].upper()

    _runner, authority, adapter = await setup_owner(monkeypatch, produce)
    # Discord's real adapter calls this inherited batching API before handle_message
    # for ordinary text; its constructor's default quiet period is 0.6 seconds.
    adapter._text_batch_delay_seconds = 0.6
    try:
        await adapter.handle_message(event(adapter, 'm1', 'work one'))
        await adapter.handle_message(event(adapter, 'm2', 'work two'))
        await async_wait_until(lambda: len(entered) == 2, what='two works running')
        adapter._enqueue_text_event(event(adapter, 'f1', 'follow one', reply='m1'))
        adapter._enqueue_text_event(event(adapter, 'f2', 'follow two', reply='m2'))
        await async_wait_until(lambda: not adapter._pending_text_batch_tasks, what='both batch timers dispatched')
        await async_wait_until(lambda: len(admissions(authority)) == 4, what='each reply admitted separately')
        rows = admissions(authority)

        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks and not adapter._pending_text_batch_tasks, what='batch cleanup')

        assert len(rows) == 4, 'Two replies to different work collapsed into one admission on the first work'
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks and not adapter._pending_text_batch_tasks, what='cleanup')
        authority.db.close()


@pytest.mark.asyncio
async def test_background_watcher_reply_remains_correlated(monkeypatch):
    gate = asyncio.Event()

    async def produce(*args):
        await gate.wait()
        return 'DONE'

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        original = event(adapter, 'request', 'long-running work')
        await adapter.handle_message(original)
        await async_wait_until(lambda: len(admissions(authority)) == 1, what='work admitted')
        route = runner._session_key_for_source(original.source)
        await runner._send_watcher_message('discord', 'room', None, 'Background command completed', {'session_key': route})
        receipt = adapter.deliveries[-1][3]
        assert adapter.deliveries[-1][1] == 'Background command completed'
        anchor = by_anchor(authority.db, conversation_scope(original.source), receipt)
        assert anchor is not None, 'Delivered background reply has no work-affinity receipt'
        await adapter.handle_message(event(adapter, 'followup', 'inspect the command output', reply=receipt))
        await async_wait_until(lambda: len(admissions(authority)) == 2, what='watcher reply admitted')
        assert len({row['target_session_id'] for row in admissions(authority)}) == 1
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='cleanup')
        authority.db.close()
