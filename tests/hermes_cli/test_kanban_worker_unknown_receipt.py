"""A booked attempt result outranks an ``unknown`` receipt; a bare unknown still refuses.

``gateway.session_managed_worker`` commits the admission ``unknown`` whenever a managed worker
dies after its reservation (a credential wall during agent construction, a lost interpreter):
the owner cannot prove the turn did not run, so it must not guess. The attempt's own worker,
though, books its exact exit in the attempt's ``worker_result`` (``record_start_failure`` /
``_record_worker_result``). Reading that answer is the difference between the dispatcher booking
the attempt's real outcome — a quota wall is requeued and the card's failure budget is untouched
— and spending the card's retries, and eventually ``gave_up``, on a fault the work never had.
"""

import asyncio
import json

import pytest
from websockets.protocol import State

from hermes_cli import kanban_worker_client
from hermes_cli.gateway_client import GatewayClient, GatewayClientError


class _UnknownOwner:
    """Admits the claim, then answers a receipt the owner could not settle as terminal."""

    state = State.OPEN  # GatewayClient.rpc reads the socket state before it sends

    def __init__(self):
        self.inbox, self.sent = asyncio.Queue(), []

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append(req)
        if req['method'] == 'kanban.run':
            await self.inbox.put(json.dumps({'id': req['id'], 'result': {
                'session_id': 's1', 'receipt': {'status': 'unknown', 'admission_id': 'a1'}}}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


@pytest.fixture
def owner():
    return _UnknownOwner()


PARAMS = {'board': 'b', 'task_id': 't1', 'run_id': 7, 'claim_lock': 'lock'}


async def _settle(client, board_db):
    """The submitter's two steps for one connection: wait for the receipt, then book its exit."""
    claim = kanban_worker_client._Claim(answered=0.0)
    receipt = await kanban_worker_client._await_receipt(client, PARAMS, board_db, claim)
    return kanban_worker_client._exit_code(receipt, PARAMS, board_db)


@pytest.mark.asyncio
async def test_a_booked_result_answers_an_unknown_receipt(monkeypatch, tmp_path, owner, capsys):
    from gateway import session_kanban
    from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
    seen = []

    def worker_result(db, params):
        seen.append((db, dict(params)))
        return {'exit_code': KANBAN_RATE_LIMIT_EXIT_CODE,
                'last_output': 'Anthropic credentials are rate-limited for claude-opus-5-5'}

    monkeypatch.setattr(session_kanban, 'worker_result', worker_result)
    async with GatewayClient(owner) as client:
        code = await _settle(client, str(tmp_path / 'kanban.db'))
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE
    # Queried for THIS attempt only: the run and claim lease the dispatcher handed the worker.
    assert seen == [(str(tmp_path / 'kanban.db'), PARAMS)]
    # The worker's own output still ends the log, so crash diagnostics quote the real cause.
    assert 'rate-limited for claude-opus-5-5' in capsys.readouterr().out
    assert [r['method'] for r in owner.sent] == ['kanban.run']


@pytest.mark.asyncio
async def test_an_unbooked_attempt_still_refuses(monkeypatch, tmp_path, owner):
    """Nothing booked means nothing is known: the ambiguous refusal must survive."""
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {})
    async with GatewayClient(owner) as client:
        with pytest.raises(GatewayClientError, match='retained by owner'):
            await _settle(client, str(tmp_path / 'kanban.db'))
