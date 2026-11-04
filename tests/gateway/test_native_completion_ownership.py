"""Only the exact native delivery owner closes an opted-in adapter lifecycle."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session_ingress import admit_message, deliver_settled
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_shared_busy_processing_lifecycle import LifecycleAdapter


class NativeAdapter(LifecycleAdapter):
    @asynccontextmanager
    async def native_execution_scope(self, event):
        yield


def fixture():
    adapter = NativeAdapter()
    event = MessageEvent(text='question', message_id='inbound', source=adapter.build_source(
        chat_id='chat', user_id='sender', chat_type='dm'))
    authority = SimpleNamespace(
        runner=SimpleNamespace(_adapter_for_source=lambda source: adapter),
        profile_id='symbolic-owner', db=None, native_waiters=set(), waiters={},
        pending_deliveries={}, marker_adopters={})
    adapter.on_processing_complete = AsyncMock()
    return adapter, event, authority


@pytest.mark.asyncio
@pytest.mark.parametrize('response,transport,state,expected', [
    ('answer', True, ('terminal', 'completed'), ProcessingOutcome.SUCCESS),
    (None, True, ('terminal', 'completed'), ProcessingOutcome.SUCCESS),
    ('answer', False, ('terminal', 'completed'), ProcessingOutcome.FAILURE),
    ('answer', 'cancel', ('terminal', 'completed'), ProcessingOutcome.FAILURE),
    ('notice', True, ('terminal', 'failed'), ProcessingOutcome.FAILURE),
    ('notice', True, ('terminal', 'interrupted'), ProcessingOutcome.CANCELLED),
])
async def test_drain_completes_actual_event_once_after_delivery(monkeypatch, response, transport, state, expected):
    adapter, actual, authority = fixture()
    actual._native_execution_lifecycle = True
    authority.pending_deliveries['admission'] = (adapter, actual, 'route', response, None)
    monkeypatch.setattr('hermes_state_runtime.get_session_admission',
                        lambda *args, **kwargs: dict(zip(('status', 'outcome'), state)))

    async def send(*args):
        adapter.on_processing_complete.assert_not_awaited()
        if transport == 'cancel':
            raise asyncio.CancelledError()
        return transport

    monkeypatch.setattr('gateway.session_ingress.deliver_response', send)
    if transport == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            await deliver_settled(authority, 'admission')
    else:
        await deliver_settled(authority, 'admission')
    await deliver_settled(authority, 'admission')
    adapter.on_processing_complete.assert_awaited_once_with(actual, expected)


@pytest.mark.asyncio
async def test_late_waiter_handoff_defers_completion_until_original_delivers(monkeypatch):
    adapter, actual, authority = fixture()
    original = MessageEvent(text=actual.text, message_id=actual.message_id, source=actual.source)
    actual._native_execution_lifecycle = True
    original._native_completion_owner = False
    authority.pending_deliveries['admission'] = (adapter, actual, 'route', 'answer', None)
    authority.native_waiters.add('admission')
    authority.marker_adopters['admission'] = original
    send = AsyncMock()
    monkeypatch.setattr('gateway.session_ingress.deliver_response', send)
    await deliver_settled(authority, 'admission')
    send.assert_not_awaited()
    adapter.on_processing_complete.assert_not_awaited()
    assert original._native_execution_event is actual
    assert actual._native_delivery_owner is original
    await adapter._run_processing_hook('on_processing_complete', original, ProcessingOutcome.SUCCESS)
    adapter.on_processing_complete.assert_not_awaited()
    original._native_completion_owner = True  # Exact waiter handed back its settled reply.
    await adapter._run_processing_hook('on_processing_complete', original, ProcessingOutcome.SUCCESS)
    await adapter._run_processing_hook('on_processing_complete', actual, ProcessingOutcome.SUCCESS)
    adapter.on_processing_complete.assert_awaited_once_with(original, ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['invalid_params', 'queue_full', 'stale_generation', 'cancel'])
async def test_precommit_refusal_keeps_original_cleanup_owner(failure):
    adapter, event, authority = fixture()

    async def refuse(original):
        assert original._native_completion_owner is False  # Before any admission suspension.
        await asyncio.sleep(0)
        if failure == 'cancel':
            raise asyncio.CancelledError()
        raise RuntimeStoreError(failure, field='route' if failure == 'stale_generation' else 'metadata')

    authority.admit_native = refuse
    if failure == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            await admit_message(authority, event)
    else:
        await admit_message(authority, event)
    await adapter._run_processing_hook('on_processing_complete', event, ProcessingOutcome.FAILURE)
    adapter.on_processing_complete.assert_awaited_once_with(event, ProcessingOutcome.FAILURE)


@pytest.mark.asyncio
async def test_terminal_receipt_cannot_complete_an_execution_it_did_not_deliver():
    adapter, event, authority = fixture()
    authority.admit_native = AsyncMock(return_value=SimpleNamespace(status='terminal', admission_id='old'))
    assert await admit_message(authority, event) is None
    await adapter._run_processing_hook('on_processing_complete', event, ProcessingOutcome.SUCCESS)
    adapter.on_processing_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_after_shielded_commit_begins_does_not_complete_busy_original():
    from gateway.session_authority import native_write_started
    from gateway.session_ingress import _complete_shared_busy
    adapter, event, authority = fixture()
    event._shared_busy_admission = None

    async def committing(original):
        assert original._native_completion_owner is False
        native_write_started.set(True)
        raise asyncio.CancelledError()

    authority.admit_native = committing
    with pytest.raises(asyncio.CancelledError):
        await admit_message(authority, event)
    await _complete_shared_busy(adapter, event, ProcessingOutcome.CANCELLED)
    adapter.on_processing_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_producer_owned_completion_is_not_duplicated_by_native_drain(monkeypatch):
    adapter, event, authority = fixture()
    event._native_execution_lifecycle = True
    event._native_producer_lifecycle = True
    authority.pending_deliveries['admission'] = (adapter, event, 'route', None, None)
    producer = AsyncMock()
    monkeypatch.setattr('gateway.session_ingress.close_producer_turn', producer)
    await deliver_settled(authority, 'admission')
    producer.assert_awaited_once_with(authority, 'admission', ProcessingOutcome.SUCCESS)
    adapter.on_processing_complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['storage', 'profile'])
async def test_completion_infrastructure_error_is_isolated(monkeypatch, caplog, boundary):
    from gateway.session_adapter_execution import complete_native_execution
    adapter, event, authority = fixture()
    event._native_execution_lifecycle = True

    def unavailable(*args, **kwargs):
        raise RuntimeError('completion infrastructure unavailable')

    if boundary == 'storage':
        monkeypatch.setattr('hermes_state_runtime.get_session_admission', unavailable)
    else:
        monkeypatch.setattr('gateway.session_ingress._execution_scope', AsyncMock(side_effect=unavailable))
    await complete_native_execution(authority, adapter, event, 'admission', True)
    adapter.on_processing_complete.assert_not_awaited()
    assert 'Native completion failed for admission admission' in caplog.text


@pytest.mark.asyncio
async def test_real_transaction_queue_cap_refusal_retains_original_cleanup(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority, native_write_started
    from gateway.session_ingress_context import native_callback
    from hermes_state_runtime import list_session_admissions
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'sender')
    monkeypatch.setattr('hermes_state_runtime.ADMISSION_QUEUE_MAX', 1)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter, event, _ = fixture()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.gateway_runner = runner
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='queue-cap')
    runner._wire_adapter_handlers(adapter)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with native_callback(runner, event, runner._native_transport_homes[None]):
        receipt = await authority.admit_native(event)
    from dataclasses import replace
    event = replace(event, message_id='refused-new-input')
    writes, write = [], authority._admit_native_write

    def attempted(**kwargs):
        writes.append(native_write_started.get())
        return write(**kwargs)

    monkeypatch.setattr(authority, '_admit_native_write', attempted)
    with native_callback(runner, event, runner._native_transport_homes[None]):
        reply = await admit_message(authority, event)
    assert 'Too many messages' in reply
    assert writes == [True]  # The real shielded writer attempted this transaction.
    assert len(list_session_admissions(authority.db, session_id=receipt.ref.session_id)) == 1
    await adapter._run_processing_hook('on_processing_complete', event, ProcessingOutcome.FAILURE)
    adapter.on_processing_complete.assert_awaited_once_with(event, ProcessingOutcome.FAILURE)
    authority.db.close()


@pytest.mark.asyncio
async def test_busy_settled_answer_survives_observational_select_failure(monkeypatch):
    import sqlite3
    from gateway.session_ingress import dispatch_shared_busy
    adapter, event, authority = fixture()
    authority.admit_native = AsyncMock(return_value=SimpleNamespace(status='queued', admission_id='reply'))
    adapter._message_handler = lambda original: admit_message(authority, original)
    adapter._notify_turn_error = AsyncMock()
    reads = 0

    def read_row(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads > 1:
            raise sqlite3.OperationalError('synthetic SELECT failure after settlement')
        return {'status': 'queued'}

    monkeypatch.setattr('hermes_state_runtime.get_session_admission', read_row)
    await dispatch_shared_busy(adapter, event, 'route')
    tasks = list(adapter._background_tasks)
    authority.waiters['reply'].set_result('settled answer')
    await asyncio.gather(*tasks)
    assert adapter.sent == ['settled answer']
    adapter._notify_turn_error.assert_not_awaited()
    adapter.on_processing_complete.assert_not_awaited()  # No trustworthy refreshed verdict.


@pytest.mark.asyncio
@pytest.mark.parametrize('adapter_type', [NativeAdapter, LifecycleAdapter])
async def test_handled_busy_control_without_admission_does_not_get_new_failure_hook(adapter_type):
    from gateway.session_ingress import dispatch_shared_busy
    adapter = adapter_type()
    event = MessageEvent(text='yes', message_id='approval', source=adapter.build_source(
        chat_id='chat', user_id='sender', chat_type='dm'))
    adapter._message_handler = AsyncMock(return_value=None)  # Approval/plugin consumer already handled it.
    adapter.on_processing_complete = AsyncMock()
    await dispatch_shared_busy(adapter, event, 'route')
    await asyncio.gather(*list(adapter._background_tasks))
    adapter._message_handler.assert_awaited_once_with(event)
    adapter.on_processing_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_native_route_fence_completes_as_cancelled():
    from gateway.session_ingress import dispatch_shared_busy
    adapter, event, authority = fixture()
    authority.admit_native = AsyncMock(side_effect=RuntimeStoreError('stale_generation', field='route'))
    adapter._message_handler = lambda original: admit_message(authority, original)
    await dispatch_shared_busy(adapter, event, 'route')
    await asyncio.gather(*list(adapter._background_tasks))
    adapter.on_processing_complete.assert_awaited_once_with(event, ProcessingOutcome.CANCELLED)


@pytest.mark.asyncio
@pytest.mark.parametrize('shutdown', [True, False])
async def test_busy_prewrite_cancellation_preserves_expected_shutdown_outcome(shutdown):
    from gateway.session_ingress import dispatch_shared_busy
    adapter, event, authority = fixture()
    entered = asyncio.Event()

    async def before_write(original):
        entered.set()
        await asyncio.Event().wait()

    authority.admit_native = before_write
    adapter._message_handler = lambda original: admit_message(authority, original)
    receiving = asyncio.create_task(dispatch_shared_busy(adapter, event, 'route'))
    await asyncio.wait_for(entered.wait(), 5)
    tasks = list(adapter._background_tasks)
    assert len(tasks) == 1
    if shutdown:
        await adapter.cancel_background_tasks()
    else:
        tasks[0].cancel()
    await asyncio.gather(receiving, *tasks, return_exceptions=True)
    expected = ProcessingOutcome.CANCELLED if shutdown else ProcessingOutcome.FAILURE
    adapter.on_processing_complete.assert_awaited_once_with(event, expected)


@pytest.mark.asyncio
async def test_expected_cancel_does_not_rewrite_failed_admission_outcome():
    from gateway.session_ingress import _complete_shared_busy
    adapter, event, _ = fixture()
    event._native_completion_owner = True
    event._shared_busy_admission = ('terminal', 'failed')
    await _complete_shared_busy(adapter, event, ProcessingOutcome.CANCELLED)
    adapter.on_processing_complete.assert_awaited_once_with(event, ProcessingOutcome.FAILURE)


@pytest.mark.asyncio
async def test_expected_cancel_during_refusal_notice_does_not_erase_refusal():
    from gateway.session_ingress import _complete_shared_busy
    adapter, event, authority = fixture()
    event._shared_busy_admission = None
    authority.admit_native = AsyncMock(side_effect=RuntimeStoreError('queue_full'))
    assert 'Too many messages' in await admit_message(authority, event)
    await _complete_shared_busy(adapter, event, ProcessingOutcome.CANCELLED)
    adapter.on_processing_complete.assert_awaited_once_with(event, ProcessingOutcome.FAILURE)
