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
