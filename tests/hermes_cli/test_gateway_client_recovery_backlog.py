"""Recovery RPCs awaited from a client's only event consumer (#134889 CLI-35/CLI-37 review).

``GatewayChatView`` and ACP await ``session.resume`` / ``prompt.receipt`` from the task that drains
``GatewayClient.events``. The owner attaches the new subscription before it answers, so a burst
larger than the client queue can sit ahead of the answer: the reader must keep reading toward it,
and ACP's post-ack recovery must never be dropped when that queue is full. Real ``GatewayClient``
over a fake socket; probes from @dokterdok's review.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State


def event(kind, seq=1, sid='s', admission=None):
    return {'jsonrpc': '2.0', 'method': 'event', 'params': {
        'session_id': sid, 'type': kind, 'replay_epoch': 'e', 'seq': seq, 'admission_id': admission,
        'payload': {'text': 'x'}}}


class ResumeWire:
    state = State.OPEN  # GatewayClient.rpc reads the socket state before it sends

    def __init__(self, burst):
        self.burst = burst
        self.frames = asyncio.Queue()
        self.answer_read = asyncio.Event()

    async def send(self, raw):
        req = json.loads(raw)
        assert req['method'] == 'session.resume'
        # An attached owner can send a batch before its resume answer (owner fanout: 16,384
        # frames; the client queue: 4,096).
        for seq in range(2, self.burst + 2):
            self.frames.put_nowait(event('message.delta', seq))
        self.frames.put_nowait({'jsonrpc': '2.0', 'id': req['id'], 'result': {
            'stored_session_id': 's', 'replay_epoch': 'e', 'last_sequence': 1, 'pending': [], 'prompts': []}})

    async def __aiter__(self):
        while True:
            frame = await self.frames.get()
            if 'id' in frame:
                self.answer_read.set()
            yield json.dumps(frame)


@pytest.mark.asyncio
@pytest.mark.parametrize('consumer', ['chat', 'acp'])
@pytest.mark.parametrize('burst', [4096, 4097, 6000])
async def test_resume_answer_behind_a_burst_stays_reachable(monkeypatch, consumer, burst):
    from hermes_cli import gateway_client
    monkeypatch.setattr(gateway_client, 'EVENT_STALL_TIMEOUT', 30)
    wire = ResumeWire(burst)
    async with gateway_client.GatewayClient(wire) as client:
        if consumer == 'chat':
            from hermes_cli.gateway_chat_view import GatewayChatView
            view = GatewayChatView(client, {'stored_session_id': 's'}, quiet=True)
            pump = asyncio.create_task(view.render())
            failure = lambda: view.failure
        else:
            from acp_adapter.gateway_server import GatewayACPAgent
            view = GatewayACPAgent()
            view._gateway, view._conn = client, AsyncMock()
            view._snapshots['s'] = {'replay_epoch': 'e', 'last_sequence': 0}
            pump = view._event_task = asyncio.create_task(view._events())
            failure = lambda: view._failure
        answered = asyncio.create_task(wire.answer_read.wait())
        try:
            wire.frames.put_nowait(event('session.replay_gap'))
            # Well under EVENT_STALL_TIMEOUT: a parked reader would not reach the answer in time.
            await asyncio.wait({pump, answered}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
            assert wire.answer_read.is_set(), f'{consumer}: answer unread, queued={client.events.qsize()}'
            assert failure() is None
        finally:
            pump.cancel()
            answered.cancel()
            await asyncio.gather(pump, answered, return_exceptions=True)


class PostAckWire:
    state = State.OPEN  # GatewayClient.rpc reads the socket state before it sends

    def __init__(self):
        self.frames = asyncio.Queue()
        self.submit_id = None
        self.receipts = []

    async def send(self, raw):
        req = json.loads(raw)
        if req['method'] == 'prompt.submit':
            self.submit_id = req['id']
            self.frames.put_nowait(event('session.replay_gap', admission='other'))
        elif req['method'] == 'session.resume':
            # The turn completed in the gap before its ack reached ACP.
            self.frames.put_nowait({'id': req['id'], 'result': {
                'stored_session_id': 's', 'replay_epoch': 'e', 'last_sequence': 1, 'pending': [], 'prompts': []}})
            self.frames.put_nowait(event('message.delta', sid='t', admission='other'))
        elif req['method'] == 'prompt.receipt':
            self.receipts.append(req['params']['admission_id'])
            self.frames.put_nowait({'id': req['id'], 'result': {
                'admission_id': 'a', 'status': 'terminal', 'outcome': 'completed',
                'result': {'final_response': 'finished'}}})
        else:
            raise AssertionError(req)

    async def __aiter__(self):
        while True:
            yield json.dumps(await self.frames.get())


@pytest.mark.asyncio
async def test_a_full_queue_never_drops_a_post_ack_recovery():
    """Session t fills the queue while session s's ack lands after a gap re-attach covering it."""
    import acp
    from acp_adapter.gateway_server import GatewayACPAgent
    from hermes_cli.gateway_client import GatewayClient
    from tests.waiting import async_wait_until

    wire = PostAckWire()
    entered, release = asyncio.Event(), asyncio.Event()
    async with GatewayClient(wire) as client:
        agent = GatewayACPAgent()
        agent._gateway, agent._conn = client, AsyncMock()
        agent._snapshots.update({sid: {'replay_epoch': 'e', 'last_sequence': 0} for sid in ('s', 't')})

        async def deliver(**kwargs):
            if kwargs['session_id'] == 't' and not entered.is_set():
                entered.set()
                await release.wait()

        agent._conn.session_update.side_effect = deliver
        pump = agent._event_task = asyncio.create_task(agent._events())
        prompt = asyncio.create_task(agent.prompt([acp.text_block('q')], 's'))
        try:
            await async_wait_until(lambda: entered.is_set() or prompt.done() or agent._failure is not None, timeout=5)
            assert entered.is_set(), f'recovery failed before the post-ack race: {agent._failure}'
            for seq in range(2, client.events.maxsize + 2):
                wire.frames.put_nowait(event('message.delta', seq, sid='t', admission='other'))
            wire.frames.put_nowait({'id': wire.submit_id, 'result': {'admission_id': 'a', 'status': 'queued'}})
            await async_wait_until(lambda: agent._admissions.get('s') == 'a', timeout=5)
            assert client.events.full(), 'probe did not exercise the full-queue branch'
            await asyncio.sleep(0)  # let _after_ack hand its recovery over
            release.set()
            response = await asyncio.wait_for(prompt, 10)
            assert response.stop_reason == 'end_turn', (wire.receipts, agent._failure)
            assert wire.receipts == ['a']
        finally:
            release.set()
            prompt.cancel()
            pump.cancel()
            await asyncio.gather(prompt, pump, return_exceptions=True)
