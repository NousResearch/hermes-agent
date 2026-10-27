"""An owner receipt cannot turn a settled board attempt into a crash or hide a real failure."""
import asyncio
import os
from contextlib import asynccontextmanager

import pytest

from gateway import session_kanban
from hermes_cli import gateway_client, kanban_db as kb, kanban_db_dispatch as dispatch, kanban_worker_client
from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_worker_client import _submit_claim


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        tid = kb.create_task(conn, title='Private fixture', assignee='default')
        task = kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, 'worker_bound',
                             {'pid': os.getpid(), 'claim_lock': task.claim_lock}, run_id=task.current_run_id)
    params = dict(board='default', task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    return path, params


class Owner:
    def __init__(self, status):
        self.status = status
        self.calls = []
        self.events = asyncio.Queue()

    async def rpc(self, method, **params):
        self.calls.append(method)
        if self.status == 'polled-unknown':
            if method == 'prompt.receipt':
                return {'status': 'unknown'}
            return {'session_id': 'fixture-session',
                    'receipt': {'status': 'started', 'admission_id': 'fixture-admission'}}
        if self.status == 'refused':
            raise gateway_client.GatewayRPCError('stale_kanban_claim')
        return {'session_id': 'fixture-session',
                'receipt': {'status': self.status, 'admission_id': 'fixture-admission'}}


@pytest.mark.parametrize('status', ['terminal', 'unknown', 'refused'])
def test_operator_completed_attempt_exits_cleanly_without_retry(attempt, monkeypatch, status):
    path, params = attempt
    with connect_closing(path) as conn:
        assert kb.complete_task(conn, params['task_id'], summary='Operator verified', force=True)
    owner = Owner(status)

    @asynccontextmanager
    async def connect():
        yield owner

    monkeypatch.setattr(gateway_client, 'connect_gateway', connect)
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
    assert asyncio.run(kanban_worker_client.run(params, path)) == 0
    assert owner.calls == ['kanban.run']
    with connect_closing(path) as conn:
        assert dispatch.detect_crashed_workers(conn) == []
        task = kb.get_task(conn, params['task_id'])
        assert task.status == 'done' and task.consecutive_failures == 0
        assert not any(e.kind in ('crashed', 'gave_up') for e in kb.list_events(conn, task.id))
    assert session_kanban.worker_result(path, params | {'claim_lock': 'foreign'}) == {}


@pytest.mark.parametrize('code', [0, 1, kb.KANBAN_RATE_LIMIT_EXIT_CODE, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE])
def test_unknown_owner_receipt_preserves_exact_worker_result(attempt, code, capsys):
    path, params = attempt
    session_kanban._record_worker_result(dict(params, db=str(path)), code, 'This attempt only')
    assert asyncio.run(_submit_claim(Owner('unknown'), params, path)) == code
    assert capsys.readouterr().out.strip() == 'This attempt only'


def test_unknown_active_attempt_without_result_remains_uncertain(attempt):
    path, params = attempt
    with pytest.raises(gateway_client.GatewayClientError, match='execution unknown'):
        asyncio.run(_submit_claim(Owner('unknown'), params, path))
    with connect_closing(path) as conn:
        assert kb.get_task(conn, params['task_id']).status == 'running'


@pytest.mark.asyncio
async def test_stale_refusal_cannot_silence_active_or_foreign_attempt(attempt, monkeypatch):
    path, params = attempt

    @asynccontextmanager
    async def connect():
        yield Owner('refused')

    monkeypatch.setattr(gateway_client, 'connect_gateway', connect)
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
    with pytest.raises(gateway_client.GatewayRPCError, match='stale_kanban_claim'):
        await kanban_worker_client.run(params, path)
    with connect_closing(path) as conn:
        kb.complete_task(conn, params['task_id'], summary='Operator verified', force=True)
    with pytest.raises(gateway_client.GatewayRPCError, match='stale_kanban_claim'):
        await kanban_worker_client.run(params | {'claim_lock': 'foreign'}, path)


def test_polled_unknown_receipt_honors_late_operator_completion(attempt):
    path, params = attempt
    session_kanban._record_worker_result(dict(params, db=str(path)), 1, '')
    with connect_closing(path) as conn:
        kb.complete_task(conn, params['task_id'], summary='Operator verified', force=True)
    owner = Owner('polled-unknown')
    assert asyncio.run(_submit_claim(owner, params, path)) == 0
    assert owner.calls == ['kanban.run', 'prompt.receipt']
    for changed in ({'claim_lock': 'foreign'}, {'run_id': params['run_id'] + 1}, {'task_id': 'foreign'}):
        assert session_kanban.worker_result(path, params | changed) == {}


def test_completed_task_does_not_hide_permission_refusal(attempt):
    path, params = attempt
    with connect_closing(path) as conn:
        kb.complete_task(conn, params['task_id'], summary='Operator verified', force=True)

    class Denied(Owner):
        async def rpc(self, method, **params):
            raise gateway_client.GatewayRPCError('permission_denied')

    with pytest.raises(gateway_client.GatewayRPCError, match='permission_denied'):
        asyncio.run(_submit_claim(Denied('refused'), params, path))


def test_submit_preserves_host_busy_timeout_but_not_claim_identity(attempt, monkeypatch):
    path, params = attempt
    for key, value in dict(HERMES_KANBAN_BOARD=params['board'], HERMES_KANBAN_TASK=params['task_id'],
                           HERMES_KANBAN_RUN_ID=str(params['run_id']), HERMES_KANBAN_CLAIM_LOCK=params['claim_lock'],
                           HERMES_KANBAN_DB=str(path), HERMES_KANBAN_BUSY_TIMEOUT_MS='2500').items():
        monkeypatch.setenv(key, value)

    async def run(submitted, board_db):
        assert submitted == params and board_db == str(path)
        assert os.environ['HERMES_KANBAN_BUSY_TIMEOUT_MS'] == '2500'
        assert not any(key in os.environ for key in ('HERMES_KANBAN_TASK', 'HERMES_KANBAN_RUN_ID',
                                                     'HERMES_KANBAN_CLAIM_LOCK', 'HERMES_KANBAN_DB'))
        return 0

    monkeypatch.setattr(kanban_worker_client, 'run', run)
    assert kanban_worker_client._submit() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['unknown', 'refused', 'permission_denied'])
async def test_result_reconciliation_keeps_event_drain_and_cancels_it(attempt, monkeypatch, status):
    path, params = attempt
    with connect_closing(path) as conn:
        kb.complete_task(conn, params['task_id'], summary='Operator verified', force=True)
    started, cancelled = asyncio.Event(), asyncio.Event()
    drains = []

    class DrainedOwner(Owner):
        events = asyncio.Queue()

        async def rpc(self, method, **submitted):
            assert submitted['_timeout'] == kanban_worker_client.admission_answer_budget()
            await asyncio.wait_for(started.wait(), 2)
            if status == 'permission_denied':
                raise gateway_client.GatewayRPCError(status)
            return await super().rpc(method, **submitted)

    @asynccontextmanager
    async def connect():
        yield DrainedOwner(status)

    async def discard(events):
        drains.append(asyncio.current_task())
        started.set()
        try:
            await events.get()
        finally:
            cancelled.set()

    monkeypatch.setattr(gateway_client, 'connect_gateway', connect)
    monkeypatch.setattr(kanban_worker_client, '_discard', discard, raising=False)
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
    if status == 'permission_denied':
        with pytest.raises(gateway_client.GatewayRPCError, match=status):
            await kanban_worker_client.run(params, path)
    else:
        assert await kanban_worker_client.run(params, path) == 0
    await asyncio.wait_for(cancelled.wait(), 2)
    await asyncio.gather(*drains, return_exceptions=True)
    assert len(drains) == 1 and drains[0].cancelled()
