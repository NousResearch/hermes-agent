"""Confirmed public receipts, not text guesses, select parallel continuations."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session_parallel import conversation_scope, record_output, route_input
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_state_parallel import by_anchor
from tests.gateway.test_parallel_native_ingress import admissions, event, setup_owner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_stream_prompt_and_multipart_receipts_keep_work_affinity(monkeypatch):
    gate = asyncio.Event()

    async def produce(*args):
        await gate.wait()
        return 'FINAL'

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        original = event(adapter, 'm1', 'work')
        await adapter.handle_message(original)
        await async_wait_until(lambda: len(admissions(authority)) == 1, what='admitted')
        scope = conversation_scope(original.source)
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id='room', config=StreamConsumerConfig(),
            initial_reply_to_id='m1',
            on_receipt=lambda result: record_output(runner, original.source, result))
        await consumer._first_send('A streamed reply sufficiently long to deliver.', finalize=True)
        turn = object.__new__(TurnRunner)
        turn._runner = runner
        turn._ctx = SimpleNamespace(_status_adapter=adapter, source=original.source)
        turn._approval_owner = None
        prompt = await turn._send_shared_clarify(
            None, chat_id='room', question='Which?', choices=None,
            clarify_id='probe', session_key=runner._session_key_for_source(original.source))
        multi = SendResult(success=True, message_id='head', continuation_message_ids=('tail',),
                           raw_response={'message_ids': ['head', 'tail', 'overflow']})
        record_output(runner, original.source, multi)
        for mid in (consumer.message_id, prompt.message_id, 'head', 'tail', 'overflow'):
            assert by_anchor(authority.db, scope, mid)['work_id'] == original.source.parallel_work.work_id
            reply = event(adapter, 'reply-' + mid, 'continue', reply=mid)
            assert not adapter._drop_unresolved(reply)
            assert await route_input(runner, reply) is None
            assert reply.source.parallel_work == original.source.parallel_work
        record_output(runner, original.source, SendResult(success=False, message_id='not-delivered'))
        assert by_anchor(authority.db, scope, 'not-delivered') is None
        unrelated = event(adapter, 'unknown', 'yes', reply='missing-receipt')
        assert not adapter._drop_unresolved(unrelated)
        assert await route_input(runner, unrelated) is not None
        assert unrelated.source.parallel_work is None
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='cleanup')
        authority.db.close()


@pytest.mark.asyncio
async def test_background_work_without_authoritative_origin_never_falls_into_room(monkeypatch):
    gate = asyncio.Event()

    async def produce(*args):
        await gate.wait()
        return 'FINAL'

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        original = event(adapter, 'm1', 'work')
        await adapter.handle_message(original)
        await async_wait_until(lambda: len(admissions(authority)) == 1, what='admitted')
        key = runner._session_key_for_source(original.source)
        assert runner._session_key_for_source(runner._build_process_event_source({'session_key': key})) == key
        entry = runner.session_store._entries.pop(key)
        monkeypatch.setattr(runner, '_get_cached_session_source', lambda key: None)
        assert runner._build_process_event_source({'session_key': key}) is None
        runner.session_store._entries[key] = entry
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='cleanup')
        authority.db.close()
