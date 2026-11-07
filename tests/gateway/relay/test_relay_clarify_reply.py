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


@pytest.mark.asyncio
async def test_a_message_instead_of_an_answer_cancels_the_question_and_runs(tmp_path, monkeypatch):
    from tools import clarify_gateway as clarify
    answers = []

    async def asks(event):
        clarify.register('c-relay', runner._session_key_for_source(event.source), 'Which?', ['a', 'b'])
        answers.append(await asyncio.to_thread(clarify.wait_for_response, 'c-relay', 60))
        execution_result.get()['result'] = {'final_response': 'first done', 'messages': []}
        return 'first done'

    async with _gateway(tmp_path, 'relay-clarify-prose') as (runner, _authority, send, _outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch, {'first': asks})
        await send('1700000000.000500', 'D5', 'first')
        await async_wait_until(lambda: clarify.get_pending('c-relay') is not None, timeout=20,
                               what='the question is pending')
        await send('1700000000.000501', 'D5', 'can i set up multiple bots', thread_id='1700000000.000500')
        await async_wait_until(lambda: executed == ['first', 'can i set up multiple bots'], timeout=20,
                               what='the message ran')
        assert answers == [clarify.CANCELLED]
