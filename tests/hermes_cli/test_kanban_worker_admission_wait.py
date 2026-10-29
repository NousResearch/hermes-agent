"""KB-7: the Kanban submitter waits for ``kanban.run``'s admission answer past the default RPC budget.

Before it answers, the owner's ``kanban.run`` checks the claim and writes its ``owner_admitted``
marker in one ``BEGIN IMMEDIATE`` on the board, the cross-profile dispatch bus, whose busy
timeout is 120 s. Behind another board writer, the answer can come after the client's 30 s
default. The submitter then exited "Gateway connection failed" while the owner went on to admit
the attempt. The later-bound worker's submitter-liveness watcher ended it, and the admission was
left ``unknown``. Time is scaled 1000x: 30 s -> 30 ms.
"""
import asyncio
import json

import pytest

from hermes_cli import kanban_worker_client
from hermes_cli.gateway_client import GatewayClient


class _Owner:
    """Answers ``kanban.run`` after a 120 s (scaled) board wait, then a terminal receipt."""

    def __init__(self):
        self.inbox, self.sent = asyncio.Queue(), []

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append(req)
        if req['method'] == 'kanban.run':
            async def admit():
                await asyncio.sleep(0.12)  # health: allow HX013 -- must land after the scaled 30 ms default budget
                await self.inbox.put(json.dumps({'id': req['id'], 'result': {
                    'session_id': 's1', 'receipt': {'status': 'terminal', 'admission_id': 'a1'}}}))
            asyncio.ensure_future(admit())

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


@pytest.mark.asyncio
async def test_a_slow_admission_answer_is_awaited_not_failed(monkeypatch, tmp_path):
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, 'wait_for', lambda fut, timeout: real_wait_for(
        fut, None if timeout is None else timeout / 1000))
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0, 'last_output': 'done'})
    owner = _Owner()
    params = {'board': 'b', 'task_id': 't1', 'run_id': 1, 'claim_lock': 'lock'}
    async with GatewayClient(owner) as client:
        receipt = await kanban_worker_client._await_receipt(client, params, str(tmp_path / 'kanban.db'), kanban_worker_client._Claim(answered=0.0))
    code = kanban_worker_client._exit_code(receipt, params, str(tmp_path / 'kanban.db'))
    assert code == 0
    # One submission: the wait never resends the claim, and the budget never goes on the wire.
    assert [r['method'] for r in owner.sent] == ['kanban.run']
    assert all('_timeout' not in r['params'] for r in owner.sent)


class _StalledOwner(_Owner):
    """Admits (its ``owner_admitted`` marker is committed) and then never answers or hangs up."""

    async def send(self, raw):
        self.sent.append(json.loads(raw))


@pytest.mark.asyncio
async def test_an_owner_that_stalls_after_admitting_fails_the_wait_within_the_lock_budget(monkeypatch, tmp_path):
    """Once ``owner_admitted`` commits, ``enforce_max_runtime`` and TTL reclaim both skip the run
    (``owner_reclaim_paused``), so nothing outside the submitter bounds it: the wait for the
    answer must end once the board's real lock wait (``write_txn``'s retried ``BEGIN IMMEDIATE``
    under the busy timeout) plus a margin has passed. Time is scaled 1000x."""
    from hermes_cli import kanban_db_connect
    monkeypatch.setenv('HERMES_KANBAN_BUSY_TIMEOUT_MS', '1000')
    budget = kanban_worker_client.admission_answer_budget()
    assert budget == (kanban_db_connect._BUSY_MAX_RETRIES + 1) * 1.0 + kanban_worker_client.ADMISSION_ANSWER_MARGIN_S
    real_wait_for, budgets = asyncio.wait_for, []

    def scaled(fut, timeout):
        budgets.append(timeout)
        return real_wait_for(fut, None if timeout is None else timeout / 1000)
    monkeypatch.setattr(asyncio, 'wait_for', scaled)
    owner = _StalledOwner()
    params = {'board': 'b', 'task_id': 't1', 'run_id': 1, 'claim_lock': 'lock'}
    async with GatewayClient(owner) as client:
        with pytest.raises(TimeoutError):
            # The outer 5 s guard only stops an unbounded wait from hanging the test.
            await real_wait_for(kanban_worker_client._await_receipt(client, params, str(tmp_path / 'kanban.db'), kanban_worker_client._Claim(answered=0.0)), 5)
    assert budgets == [budget] and [r['method'] for r in owner.sent] == ['kanban.run']
