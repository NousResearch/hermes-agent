"""A relay message sent while the chat's turn waits on a clarify question cancels the question.

Relay deliveries skip ``_handle_message`` and commit straight to the durable FIFO, so a user who
wrote something instead of picking a choice queued behind the turn waiting for them, which sat out
the whole ``clarify_timeout`` (an hour) before their message ran. Real ``GatewayRunner`` + session
authority + relay transport against a fixture connector; only the model turn is stubbed.
"""
import asyncio

import pytest

from gateway.session_results import execution_result
from tests.gateway.relay.test_relay_native_turn_lifecycle import _gateway, _stub_model
from tests.waiting import async_wait_until

pytest.importorskip('websockets')


@pytest.fixture(autouse=True)
def _fresh_consumed(monkeypatch):
    """The consumed-delivery record is process-wide: each test starts with an empty one."""
    from collections import OrderedDict

    from gateway import session_ingress_live_turn
    monkeypatch.setattr(session_ingress_live_turn, '_consumed', OrderedDict())

FIRST, ROOT = '1700000000.000500', 'D5'


class _Agent:
    """The running turn's agent: records what interrupted it."""

    def __init__(self):
        self.interrupts = []
        self.interrupted = asyncio.Event()
        self.loop = asyncio.get_running_loop()

    def interrupt(self, text=None):
        self.interrupts.append(text)
        self.loop.call_soon_threadsafe(self.interrupted.set)


def _asking_turn(runner, agent, answers, *, hold_until_interrupted):
    async def asks(event):
        key = runner._session_key_for_source(event.source)
        runner._session_state(key).turn.agent = agent
        from tools import clarify_gateway as clarify
        clarify.register('c-relay', key, 'Which?', ['a', 'b'])
        answers.append(await asyncio.to_thread(clarify.wait_for_response, 'c-relay', 60))
        if hold_until_interrupted:
            await asyncio.wait_for(agent.interrupted.wait(), 20)
        execution_result.get()['result'] = {'final_response': 'first done', 'messages': []}
        return 'first done'
    return asks


@pytest.mark.asyncio
async def test_a_message_instead_of_an_answer_cancels_the_question_and_takes_over_in_queue_mode(
        tmp_path, monkeypatch):
    from tools import clarify_gateway as clarify
    answers, agent = [], _Agent()
    async with _gateway(tmp_path, 'relay-clarify-prose') as (runner, _authority, send, _outgoing, _adapter):
        # queue mode would leave the message behind a turn that carries on without its answer
        monkeypatch.setattr(runner, '_busy_input_mode', 'queue', raising=False)
        executed = _stub_model(runner, monkeypatch, {'first': _asking_turn(
            runner, agent, answers, hold_until_interrupted=True)})
        await send(FIRST, ROOT, 'first')
        await async_wait_until(lambda: clarify.get_pending('c-relay') is not None, timeout=20,
                               what='the question is pending')
        await send('1700000000.000501', ROOT, 'can i set up multiple bots', thread_id=FIRST)
        await async_wait_until(lambda: executed == ['first', 'can i set up multiple bots'], timeout=20,
                               what='the message ran')
        assert answers == [clarify.CANCELLED]
        assert agent.interrupts == ['can i set up multiple bots']


@pytest.mark.asyncio
async def test_a_resent_answer_is_not_run_again_as_a_new_turn(tmp_path, monkeypatch):
    """The connector re-sends a frame whose ACK was lost: an answer consumed by the question, never
    committed, must not reach the FIFO the second time."""
    answers = []
    async with _gateway(tmp_path, 'relay-clarify-resend') as (runner, _authority, send, outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch, {'first': _asking_turn(
            runner, _Agent(), answers, hold_until_interrupted=False)})
        await send(FIRST, ROOT, 'first')
        from tools import clarify_gateway as clarify
        await async_wait_until(lambda: clarify.get_pending('c-relay') is not None, timeout=20,
                               what='the question is pending')
        await send('1700000000.000502', ROOT, '2', thread_id=FIRST)
        await async_wait_until(lambda: answers == ['b'], timeout=20, what='the answer resolved the question')
        acks = lambda: [f for f in outgoing if f.get('type') == 'inbound_ack']
        await async_wait_until(lambda: len(acks()) >= 2, timeout=20, what='both frames acknowledged')
        await send('1700000000.000502', ROOT, '2', thread_id=FIRST)
        await async_wait_until(lambda: len(acks()) >= 3, timeout=20, what='the re-send acknowledged')
        # The chat's FIFO runs a later message after anything the re-send could have queued.
        await send('1700000000.000503', ROOT, 'next', thread_id=FIRST)
        await async_wait_until(lambda: 'next' in executed, timeout=20, what='the next message ran')
        assert executed == ['first', 'next']


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True], ids=['overlapping', 'after-cancel'])
async def test_a_re_sent_yes_never_approves_the_next_command(tmp_path, monkeypatch, cancel):
    """An approval resolves, then awaits its confirmation send; the connector re-sends the unACKed
    "yes" while that send is in flight, or after a shutdown cancelled it. Either way the re-send must
    not approve the command the turn asks about next."""
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    async with _gateway(tmp_path, 'relay-approval-resend') as (runner, _authority, send, outgoing, adapter):
        pending, answers = {}, []
        sending = asyncio.Event()
        release = asyncio.Event()
        original_send = runner._send_busy_reply

        async def held_confirmation(*args, **kwargs):
            sending.set()
            await release.wait()
            return await original_send(*args, **kwargs)

        monkeypatch.setattr(runner, '_send_busy_reply', held_confirmation)

        async def ask_twice(event):
            key = runner._session_key_for_source(event.source)
            pending['key'] = key
            for number in (1, 2):
                entry = _ApprovalEntry({'command': f'rm -rf build{number}', 'description': 'recursive delete'})
                with approval._lock:
                    approval._gateway_queues.setdefault(key, []).append(entry)
                pending['number'], pending['entry'] = number, entry
                await asyncio.to_thread(entry.event.wait, 20)
                answers.append(entry.result)
            execution_result.get()['result'] = {'final_response': 'finished', 'messages': []}
            return 'finished'

        _stub_model(runner, monkeypatch, {'clean': ask_twice})
        thread = '1700000000.000001'
        message_id = '1700000000.000110'
        try:
            await send('1700000000.000100', 'D1', 'clean', thread_id=thread)
            await async_wait_until(lambda: pending.get('number') == 1, timeout=20, what='first approval')
            await send(message_id, 'D1', 'yes', thread_id=thread)
            await asyncio.wait_for(sending.wait(), 10)
            await async_wait_until(lambda: pending.get('number') == 2, timeout=10, what='second approval')
            task = adapter._admission_lanes[('slack', 'D1')]
            if cancel:
                await adapter.cancel_background_tasks()
                assert task.cancelled()
                assert adapter._admission_lanes == {}
            assert not any(f.get('type') == 'inbound_ack' and f.get('bufferId') == 'buf-' + message_id for f in outgoing)
            await send(message_id, 'D1', 'yes', thread_id=thread)
            if not cancel:
                await async_wait_until(lambda: adapter._admission_lanes.get(('slack', 'D1')) is not task,
                                       timeout=10, what='overlapping resend queued in lane')
            release.set()
            expected_acks = 1 if cancel else 2
            await async_wait_until(lambda: sum(f.get('type') == 'inbound_ack' and f.get('bufferId') == 'buf-' + message_id for f in outgoing) == expected_acks, timeout=10, what='retry ACK')
            assert not pending['entry'].event.is_set(), 'replayed first yes approved the second dangerous command'
        finally:
            release.set()
            if 'key' in pending:
                approval.resolve_gateway_approval(pending['key'], 'deny', resolve_all=True)


@pytest.mark.asyncio
async def test_the_replay_lookup_uses_the_request_id_admission_stores(tmp_path):
    """An edit is admitted under ``<message id>:edit:<revision>``; the producer's replay lookup and
    delivery lock must use that id, or a re-sent edit misses its row and answers a new prompt."""
    from dataclasses import replace

    from gateway.platforms import webhook_ingress
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    async with _gateway(tmp_path, 'relay-edit-replay') as (_runner, authority, _send, _outgoing, adapter):
        source = SessionSource(platform=adapter.platform, chat_type='dm', chat_id='D8', user_id='U-D8')
        original = MessageEvent(text='helo', source=source, message_id='1700000000.000800')
        edit = replace(original, text='hello', edit_revision='1800000010')
        for event in (original, edit):
            receipt = await authority.admit_native(event)
            payload, principal = await webhook_ingress._prepared(authority, event)
            await webhook_ingress._release_preparation(authority, payload)
            rows = webhook_ingress._committed(authority, principal, webhook_ingress._request_id(event))
            assert [row['admission_id'] for row in rows] == [receipt.admission_id]


def test_an_edit_that_answered_is_not_the_original_message():
    """An edit is its own input: consuming it never marks the original send (or another edit) as
    already consumed."""
    from dataclasses import replace

    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway import session_ingress_live_turn as live_turn

    source = SessionSource(platform=Platform.SLACK, chat_type='dm', chat_id='D9', user_id='U-D9')
    original = MessageEvent(text='helo', source=source, message_id='1700000000.000900')
    edit = replace(original, text='hello', edit_revision='1800000010')
    assert live_turn._replay_key(original) != live_turn._replay_key(edit)
