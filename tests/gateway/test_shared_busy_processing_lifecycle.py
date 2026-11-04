"""Busy waiters complete their own original event only after a settled delivery."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session_ingress import admit_message, dispatch_shared_busy
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_shared_busy_receive_loop import Adapter
from tests.waiting import async_wait_until


class LifecycleAdapter(Adapter):
    async def on_processing_start(self, event):
        pass


class NativeLifecycleAdapter(LifecycleAdapter):
    @asynccontextmanager
    async def native_execution_scope(self, event):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize('status,result,response,transport,expected', [
    ('terminal', 'completed', 'answer', 'ok', ProcessingOutcome.SUCCESS),
    ('terminal', 'completed', None, 'ok', ProcessingOutcome.SUCCESS),
    ('terminal', 'completed', 'answer', 'failed', ProcessingOutcome.FAILURE),
    ('terminal', 'completed', 'answer', 'mixed', ProcessingOutcome.SUCCESS),
    ('terminal', 'completed', 'answer', 'raises', ProcessingOutcome.FAILURE),
    ('terminal', 'completed', 'answer', 'cancel', ProcessingOutcome.CANCELLED),
    ('terminal', 'completed', 'answer', 'unexpected_cancel', ProcessingOutcome.FAILURE),
    ('terminal', 'failed', 'failure notice', 'ok', ProcessingOutcome.FAILURE),
    ('terminal', 'rejected', 'refusal notice', 'ok', ProcessingOutcome.FAILURE),
    ('terminal', 'interrupted', 'interruption notice', 'ok', ProcessingOutcome.CANCELLED),
    ('terminal', 'cancelled', None, 'ok', ProcessingOutcome.CANCELLED),
    ('queued', None, 'pause', 'ok', None),
    ('queued', None, None, 'ok', None),
    ('unknown', None, 'pause', 'ok', None),
])
async def test_shared_busy_completion_uses_exact_receipt_and_delivery(
        monkeypatch, status, result, response, transport, expected):
    adapter = NativeLifecycleAdapter()
    original = MessageEvent(text='follow-up', message_id='next', source=adapter.build_source(
        chat_id='chat', user_id='sender', chat_type='dm'))
    ref = SimpleNamespace(session_id='owned-session')
    row = {'status': 'queued', 'outcome': None}
    authority = SimpleNamespace(
        admit_native=AsyncMock(return_value=SimpleNamespace(status='queued', admission_id='exact', ref=ref)),
        native_waiters=set(), waiters={}, db=None,
        runner=SimpleNamespace(_adapter_for_source=lambda source: adapter),
        sessions={ref.session_id: SimpleNamespace(pause_notified=response is None, source=original.source)})
    adapter._message_handler = lambda event: admit_message(authority, event)
    adapter._notify_turn_error = AsyncMock()
    completions, deliveries, callbacks = [], [], []

    async def complete(event, outcome):
        completions.append((event, outcome))

    async def send_final(event, _key, text, _metadata, _ephemeral, _ttl, record_delivery):
        deliveries.append((event, text))
        if transport == 'raises':
            raise ValueError('transport failed')
        if transport in ('cancel', 'unexpected_cancel'):
            if transport == 'cancel':
                adapter._expected_cancelled_tasks.add(asyncio.current_task())
            raise asyncio.CancelledError()
        record_delivery(SendResult(success=transport == 'ok'))

    async def attachments(_event, _extracted, _metadata, *, record_delivery, **_kwargs):
        if transport == 'mixed':
            record_delivery(SendResult(success=True))

    async def callback():
        callbacks.append('unowned')

    guard = asyncio.Event()
    guard._hermes_run_generation = 99
    adapter._active_sessions['route'] = guard
    adapter.register_post_delivery_callback('route', callback, generation=99)
    monkeypatch.setattr(adapter, 'on_processing_complete', complete)
    monkeypatch.setattr(adapter, '_send_final_text', send_final)
    monkeypatch.setattr(adapter, '_deliver_attachments', attachments)
    monkeypatch.setattr('hermes_state_runtime.get_session_admission', lambda _db, admission_id: row)
    await dispatch_shared_busy(adapter, original, 'route')
    tasks = list(adapter._background_tasks)
    assert len(tasks) == 1
    row.update(status=status, outcome=result)
    if status in ('queued', 'unknown'):
        authority.waiters['exact'].set_exception(RuntimeStoreError('session_busy'))
    else:
        authority.waiters['exact'].set_result(response)
    await asyncio.gather(*tasks, return_exceptions=True)
    assert completions == ([] if expected is None else [(original, expected)])
    assert adapter._active_sessions['route'] is guard
    assert callbacks == []
    assert adapter.pop_post_delivery_callback('route', generation=99) is callback
    if deliveries:
        assert all(event is not original and event.message_id == original.message_id for event, _ in deliveries)


@pytest.mark.asyncio
async def test_shared_busy_refusal_never_completes_as_success(monkeypatch):
    adapter = NativeLifecycleAdapter()
    original = MessageEvent(text='refused', message_id='refused', source=adapter.build_source(
        chat_id='chat', user_id='sender', chat_type='dm'))
    authority = SimpleNamespace(
        admit_native=AsyncMock(side_effect=RuntimeStoreError('permission_denied')),
        runner=SimpleNamespace(_adapter_for_source=lambda source: adapter))
    adapter._message_handler = lambda event: admit_message(authority, event)
    complete = AsyncMock()
    monkeypatch.setattr(adapter, 'on_processing_complete', complete)
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await dispatch_shared_busy(adapter, original, 'route')
    complete.assert_awaited_once_with(original, ProcessingOutcome.FAILURE)


@pytest.mark.asyncio
async def test_normal_waiter_keeps_reply_without_new_postsettlement_storage_dependency(monkeypatch):
    adapter = Adapter()
    event = MessageEvent(text='normal', source=adapter.build_source(chat_id='chat', user_id='sender'))
    authority = SimpleNamespace(
        admit_native=AsyncMock(return_value=SimpleNamespace(status='queued', admission_id='normal')),
        native_waiters=set(), waiters={}, db=None)
    read = 0

    def read_row(_db, admission_id):
        nonlocal read
        read += 1
        if read > 1:
            raise RuntimeError('storage unavailable after the settled reply was handed over')
        return {'status': 'queued'}

    monkeypatch.setattr('hermes_state_runtime.get_session_admission', read_row)
    task = asyncio.create_task(admit_message(authority, event))
    await async_wait_until(lambda: 'normal' in authority.waiters, timeout=5, what='normal waiter')
    authority.waiters['normal'].set_result('settled answer')
    assert await task == 'settled answer'
