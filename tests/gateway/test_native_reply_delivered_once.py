"""A native reply reaches the chat exactly once even when an already-running drain claims and
runs the admission before ``admit_message`` resumes to register its delivery waiter
(andrexibiza-6, andrexibiza-5480238075-1). Every native reply goes through one of two senders:
the drain's ``deliver_settled`` (no live waiter) or the waiting adapter lifecycle.
Concurrent exact retries and cancellation during handoff are regressions for #160304."""
import asyncio
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

    return authority, started, finish, event, receive, reply, check_delivery


@pytest.mark.asyncio
async def test_concurrent_native_retries_only_deliver_from_the_first_waiter(native_delivery, caplog, monkeypatch):
    authority, started, finish, event, receive, reply, check_delivery = native_delivery
    caplog.set_level('INFO', logger='gateway.session_ingress')
    first_event, duplicate_event = event(), event()
    first = asyncio.create_task(receive(first_event))
    await asyncio.wait_for(started.wait(), 10)
    (admission_id,) = authority.waiters
    assert authority.marker_adopters[admission_id] is first_event

    # The busy reader must be released even when its duplicate does not own a reply.
    accepted, committed = (asyncio.get_running_loop().create_future() for _ in range(2))
    noted = asyncio.Event()
    note_busy_admitted = session_ingress.note_busy_admitted

    def note(event, *args):
        note_busy_admitted(event, *args)
        if event is duplicate_event:
            noted.set()

    monkeypatch.setattr(session_ingress, 'note_busy_admitted', note)
    acceptance_token = session_ingress.busy_acceptance.set(accepted)
    commit_token = session_ingress.busy_commit.set(committed)
    try:
        duplicate = asyncio.create_task(receive(duplicate_event))
    finally:
        session_ingress.busy_acceptance.reset(acceptance_token)
        session_ingress.busy_commit.reset(commit_token)
    await asyncio.wait_for(noted.wait(), 10)
    adopter = authority.marker_adopters[admission_id]
    assert accepted.done() and committed.done()
    drain = next(iter(authority.sessions.values())).task
    finish.set()
    results = await asyncio.wait_for(asyncio.gather(first, duplicate), 10)
    await asyncio.wait_for(drain, 10)

    check_delivery()
    assert results == [reply, None]
    assert adopter is first_event
    assert f'admission_id={admission_id}' in caplog.text
    assert 'message_id=m1' in caplog.text and 'event_id=101' in caplog.text
    assert await asyncio.wait_for(receive(event()), 10) is None
    check_delivery()


@pytest.mark.asyncio
@pytest.mark.parametrize('after_settlement', [False, True], ids=['during-inference', 'before-taking-reply'])
async def test_cancelled_native_owner_leaves_reply_for_the_drain(native_delivery, monkeypatch, after_settlement):
    authority, started, finish, event, receive, _, check_delivery = native_delivery
    first = asyncio.create_task(receive(event()))
    await asyncio.wait_for(started.wait(), 10)
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
    check_delivery()


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

    turn_started = asyncio.Event()

    async def answer(event):
        turn_started.set()
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
            await authority.sessions['s'].task
        return receipt
    monkeypatch.setattr(authority, 'admit_native', admit_native)
    adapter_sent = await asyncio.wait_for(
        session_ingress.admit_message(authority, MessageEvent(text='hi', source=source)), 10)
    await asyncio.wait_for(authority.sessions['s'].task, 10)
    deliveries = drain_sent + ([adapter_sent] if adapter_sent else [])
    assert deliveries == ['model reply'], f'the reply was sent {len(deliveries)} times: {deliveries}'
    assert not authority.native_waiters and not authority.pending_deliveries
