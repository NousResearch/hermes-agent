"""A native reply reaches the chat exactly once even when an already-running drain claims and
runs the admission before ``admit_message`` resumes to register its delivery waiter
(andrexibiza-6, andrexibiza-5480238075-1). Every native reply goes through one of two senders:
the drain's ``deliver_settled`` (no live waiter) or the waiting adapter lifecycle.
Concurrent exact retries and cancellation during handoff are regressions for #160304."""
import asyncio
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway import session_ingress
from gateway.config import GatewayConfig, Platform
from gateway.delivery_ledger import compute_obligation_id
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_contract import Submission
from hermes_cli.observability.shared_metrics_busy import stamp_busy_arrival
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.test_shared_busy_receive_loop import Adapter


@pytest_asyncio.fixture
async def native_delivery(tmp_path, monkeypatch):
    """Real native admission, drain and ledger; stub inference and the network transport."""
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    monkeypatch.setattr(adapter, 'gateway_runner', runner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    source = adapter.build_source(chat_id='native-chat', chat_type='dm', user_id='fixture-user')
    route = runner.session_store._generate_session_key(source)
    attachment = tmp_path / 'report.pdf'
    attachment.write_bytes(b'attachment fixture')
    reply = f'model reply\nMEDIA:{attachment.as_posix()}'
    send_document = AsyncMock(return_value=SendResult(success=True, message_id='attachment'))
    monkeypatch.setattr(adapter, 'send_document', send_document)
    started, finish = asyncio.Event(), asyncio.Event()

    async def answer(event):
        await runner._mark_durable_active_turn(event, route)
        started.set()
        await finish.wait()
        return reply

    inference = AsyncMock(side_effect=answer)
    monkeypatch.setattr(runner, '_handle_message', inference)

    def event():
        message = MessageEvent(text='hi', source=source, message_id='m1', platform_update_id=101)
        message._turn_marker_handoff = True
        return message

    async def receive(message):
        try:
            response = await session_ingress.admit_message(authority, message)
            await session_ingress.deliver_response(adapter, message, route, response)
            return response
        finally:
            await adapter._release_turn_marker(message)

    def check_delivery():
        inference.assert_awaited_once()
        assert adapter.sent == ['model reply']
        send_document.assert_awaited_once()
        assert send_document.await_args.kwargs['file_path'] == attachment.as_posix()
        obligation_id = compute_obligation_id(route, 'm1', 'model reply')
        with raw_holder(authority.persistence)._read_ctx() as conn:
            states = conn.execute('SELECT obligation_id, state FROM delivery_obligations').fetchall()
        assert [tuple(row) for row in states] == [(obligation_id, 'delivered')]
        assert not authority.waiters and not authority.native_waiters and not authority.marker_adopters
        assert not authority.pending_deliveries

    return SimpleNamespace(authority=authority, adapter=adapter, started=started, finish=finish,
                           event=event, receive=receive, reply=reply, check_delivery=check_delivery)


def _receive_with_busy_ack(delivery, event):
    loop = asyncio.get_running_loop()
    accepted, committed = loop.create_future(), loop.create_future()
    context = copy_context()
    context.run(session_ingress.busy_acceptance.set, accepted)
    context.run(session_ingress.busy_commit.set, committed)
    return asyncio.create_task(delivery.receive(event), context=context), accepted, committed


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['started', 'queued'])
async def test_concurrent_native_retries_only_deliver_from_the_first_waiter(native_delivery, monkeypatch, status):
    delivery = native_delivery
    authority = delivery.authority
    first_event = delivery.event()
    unblock = asyncio.Event()
    if status == 'queued':
        blocker_started = asyncio.Event()
        runner = authority.runner
        inference = runner._handle_message

        async def answer(event):
            if event.message_id == 'blocker':
                blocker_started.set()
                await unblock.wait()
                return None
            return await inference(event)

        monkeypatch.setattr(runner, '_handle_message', answer)
        blocker = MessageEvent(text='earlier turn', source=first_event.source, message_id='blocker')
        await authority.admit_native(blocker)
        await asyncio.wait_for(blocker_started.wait(), 10)
        stamp_busy_arrival(delivery.adapter, first_event)

    first, first_accepted, _ = _receive_with_busy_ack(delivery, first_event)
    try:
        if status == 'queued':
            await asyncio.wait_for(first_accepted, 10)
        else:
            await asyncio.wait_for(delivery.started.wait(), 10)
        session_id = next(iter(authority.sessions))
        rows = authority.persistence.admissions.list(session_id=session_id)
        states = {row.request_id: row.status for row in rows}
        assert states == ({'blocker': 'started', 'm1': 'queued'} if status == 'queued' else {'m1': 'started'})
        duplicate_event = delivery.event()
        if status == 'queued':
            stamp_busy_arrival(delivery.adapter, duplicate_event)
            duplicate_event._producer_delivery = duplicate_event._busy_observed = True

        # The retry releases the busy reader and returns before the original turn can settle.
        duplicate, accepted, committed = _receive_with_busy_ack(delivery, duplicate_event)
        assert await asyncio.wait_for(duplicate, 10) is None
        assert accepted.done() and committed.done()
        if status == 'queued':
            # A queued retry must not bind its arrival tag or observer to the existing admission.
            assert first_event._busy_arrival.admission_id is not None
            assert duplicate_event._busy_arrival.admission_id is None
            assert not hasattr(duplicate_event, '_observed_turn')
    finally:
        unblock.set()
        delivery.finish.set()
        reply = await asyncio.wait_for(first, 10)
        await asyncio.wait_for(next(iter(authority.sessions.values())).task, 10)

    assert reply == delivery.reply
    assert await asyncio.wait_for(delivery.receive(delivery.event()), 10) is None
    delivery.check_delivery()


@pytest.mark.asyncio
@pytest.mark.parametrize('after_settlement', [False, True], ids=['during-inference', 'before-taking-reply'])
async def test_cancelled_native_owner_leaves_reply_for_the_drain(native_delivery, monkeypatch, after_settlement):
    delivery = native_delivery
    authority, finish = delivery.authority, delivery.finish
    first = asyncio.create_task(delivery.receive(delivery.event()))
    await asyncio.wait_for(delivery.started.wait(), 10)
    (admission_id,) = authority.waiters
    drain = next(iter(authority.sessions.values())).task

    if after_settlement:
        # Enqueue cancellation just before the settled waiter wakes. No adapter has taken the
        # reply yet, even though the future has been resolved by the time cancellation runs.
        resolve = authority._resolve_settled_waiter

        async def resolve_then_cancel(admission_id, response):
            asyncio.get_running_loop().call_soon(first.cancel)
            await resolve(admission_id, response)

        monkeypatch.setattr(authority, '_resolve_settled_waiter', resolve_then_cancel)
        finish.set()
    else:
        first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, 10)
    assert admission_id not in authority.waiters
    assert admission_id not in authority.native_waiters
    assert admission_id not in authority.marker_adopters
    finish.set()
    await asyncio.wait_for(drain, 10)
    delivery.check_delivery()


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_owner', [False, True], ids=['native-owner', 'cancelled-owner'])
async def test_failed_native_reply_is_delivered_once(native_delivery, monkeypatch, cancel_owner):
    delivery = native_delivery
    authority = delivery.authority
    started, fail = asyncio.Event(), asyncio.Event()
    runner = authority.runner

    async def crash(event):
        route = runner.session_store._generate_session_key(event.source)
        assert await runner._mark_durable_active_turn(event, route)
        started.set()
        await fail.wait()
        raise RuntimeError('handler failed before capturing a reply')

    handler = AsyncMock(side_effect=crash)
    monkeypatch.setattr(runner, '_handle_message', handler)
    owner = asyncio.create_task(delivery.receive(delivery.event()))
    await asyncio.wait_for(started.wait(), 10)
    drain = next(iter(authority.sessions.values())).task
    if cancel_owner:
        resolve = authority._resolve_settled_waiter

        async def cancel_before_take(admission_id, response):
            asyncio.get_running_loop().call_soon(owner.cancel)
            await resolve(admission_id, response)

        monkeypatch.setattr(authority, '_resolve_settled_waiter', cancel_before_take)
    fail.set()
    if cancel_owner:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owner, 10)
    else:
        assert await asyncio.wait_for(owner, 10) == 'The admitted turn failed.'
    await asyncio.wait_for(drain, 10)
    assert await asyncio.wait_for(delivery.receive(delivery.event()), 10) is None
    handler.assert_awaited_once()
    assert delivery.adapter.sent == ['The admitted turn failed.']


@pytest.mark.asyncio
@pytest.mark.parametrize('already_settled', [False, True], ids=['claimed', 'settled'])
async def test_late_native_waiter_never_duplicates_delivery(tmp_path, monkeypatch, already_settled):
    from gateway import session_ingress
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionActor, SessionRef
    from tests.gateway.test_prompt_attachments import _authority as _store_authority

    turn_started, finish_turn = asyncio.Event(), asyncio.Event()

    async def answer(event):
        turn_started.set()
        await finish_turn.wait()
        return 'model reply'
    authority = await _store_authority(tmp_path, monkeypatch, answer)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c')
    authority.sessions['s'] = LiveSession(source, 's')
    authority.runner._adapter_for_source = lambda source: object()
    drain_sent = []

    async def deliver(adapter, event, session_key, response):
        drain_sent.append(response)
    monkeypatch.setattr(session_ingress, 'deliver_response', deliver)
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')

    async def admit_native(event):
        receipt = await authority.submit(actor, Submission('m1', SessionRef('p', 's'), {'text': 'hi'}, 'queue'))
        # The admitting caller resumes only after the drain already claimed and started the turn.
        await turn_started.wait()
        if already_settled:
            finish_turn.set()
            await authority.sessions['s'].task
        else:
            # admit_message installs the waiter without yielding after this receipt returns.
            asyncio.get_running_loop().call_soon(finish_turn.set)
        return receipt
    monkeypatch.setattr(authority, 'admit_native', admit_native)
    adapter_reply = await asyncio.wait_for(
        session_ingress.admit_message(authority, MessageEvent(text='hi', source=source)), 10)
    await asyncio.wait_for(authority.sessions['s'].task, 10)
    if already_settled:
        assert adapter_reply is None
        assert drain_sent == ['model reply']
    else:
        assert adapter_reply == 'model reply'
        assert drain_sent == []
    assert not authority.native_waiters and not authority.pending_deliveries
