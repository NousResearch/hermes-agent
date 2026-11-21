"""The settled card turn closes its own session (circuit-breaker, card t_6c7c4e0e).

Every cleanly completed kanban worker used to leave its local session row with ``ended_at``
NULL forever: the client exits without a viewer close, and only viewer closes stamp the
end. The fix rides the existing verb — after a terminal receipt the client detaches its own
subscription with ``end_reason='kanban_worker_end'``, which runs the same settle path main's
classic CLI used for ``cli_close``. These tests pin the client half; a lost detach may never
change the worker's exit, and an ``unknown`` receipt (the owner still holds the attempt) must
detach nothing.
"""
import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from websockets.protocol import State

from hermes_cli import kanban_worker_client
from hermes_cli.gateway_client import GatewayClient, GatewayClientError


class _Owner:
    """Answers ``kanban.run`` with a fresh subscription and one receipt verdict, and records
    every frame it receives so the test can read the wire."""

    state = State.OPEN

    def __init__(self, *, receipt_status='terminal', subscription_id='sub-1'):
        self.inbox, self.sent = asyncio.Queue(), []
        self.receipt_status = receipt_status
        self.subscription_id = subscription_id

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append(req)
        if req['method'] == 'kanban.run':
            result = {'session_id': 's1',
                      'receipt': {'status': self.receipt_status, 'admission_id': 'a1'}}
            if self.subscription_id is not None:
                result['subscription_id'] = self.subscription_id
            await self.inbox.put(json.dumps({'id': req['id'], 'result': result}))
        elif req['method'] == 'session.detach':
            result = {'session_id': req['params']['session_id'],
                      'subscription_id': req['params']['subscription_id'], 'detached': True}
            await self.inbox.put(json.dumps({'id': req['id'], 'result': result}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


def _connect(owner):
    @asynccontextmanager
    async def fake(endpoint=None, subject=None, *, max_size=8 * 1024 * 1024):
        async with GatewayClient(owner) as client:
            yield client
    return fake


_PARAMS = {'board': 'b', 'task_id': 't1', 'run_id': 1, 'claim_lock': 'lock'}


@pytest.mark.asyncio
async def test_a_terminal_card_turn_detaches_with_the_end_reason(monkeypatch, tmp_path):
    """The closing detach must carry the ids the owner can settle with, and run must still exit 0."""
    from gateway import session_kanban
    owner = _Owner()
    monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', _connect(owner))
    monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0, 'last_output': ''})
    code = await kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db'))
    assert code == 0
    detach = [r for r in owner.sent if r['method'] == 'session.detach']
    assert len(detach) == 1, 'exactly one closing detach for a terminal card turn'
    assert detach[0]['params'] == {
        'session_id': 's1', 'subscription_id': 'sub-1', 'end_reason': 'kanban_worker_end'}


@pytest.mark.asyncio
async def test_an_unknown_attempt_detaches_nothing(monkeypatch, tmp_path):
    """``unknown`` means the owner still holds the attempt: the client keeps failing honestly
    (the t_3ed790be refusal shape) and must not end a session that may still be resumed."""
    from gateway import session_kanban
    owner = _Owner(receipt_status='unknown')
    monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', _connect(owner))
    monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0})
    with pytest.raises(GatewayClientError, match='unknown'):
        await kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db'))
    assert not [r for r in owner.sent if r['method'] == 'session.detach']


@pytest.mark.asyncio
async def test_an_owner_without_a_subscription_answer_keeps_the_old_exit(monkeypatch, tmp_path):
    """A pre-fix owner answers ``kanban.run`` without a subscription_id: the worker exits exactly
    as before and sends no detach, so a mixed-version fleet never stalls on this verb."""
    from gateway import session_kanban
    owner = _Owner(subscription_id=None)
    monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', _connect(owner))
    monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0, 'last_output': ''})
    code = await kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db'))
    assert code == 0
    assert not [r for r in owner.sent if r['method'] == 'session.detach']
