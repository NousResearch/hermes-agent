"""A committed board Stop reaches only its previously recorded native owner sessions."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_process_stop
from tests.waiting import async_wait_until


@pytest.fixture
def owned_board(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session
    from gateway.session_policy import build_policy
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path / 'kanban'))
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = asyncio.run(initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture'))
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    board_db = tmp_path / 'board.db'
    kbc.init_db(board_db)
    conn = kbc.connect(board_db)
    task_id = kb.create_task(conn, title='Native Stop delivery')
    task = kb.claim_task(conn, task_id)
    calls = []

    def admit(run_id, *, context_changes=None, binding_changes=None):
        context = dict(task_id=task_id, run_id=run_id, db=str(board_db))
        context.update(context_changes or {})
        policy = build_policy({'cwd': str(tmp_path), 'model': 'm', 'source': 'cli'}, {}, profile_terminal=False)
        policy = replace(policy, source='kanban', kanban_json=json.dumps(context))
        ref = create_local_session(authority, connection.actor, {'request_id': f'owner-{run_id}'}, trusted_policy=policy)
        binding = dict(db=str(authority.db.db_path), session_id=ref.session_id, request_id=f'run-{run_id}')
        binding.update(binding_changes or {})
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, 'owner_admitted', binding, run_id=run_id)
        return ref.session_id

    class Client:
        async def rpc(self, method, **params):
            calls.append((method, params))
            return {'execution_generation': 17}

    client = Client()

    @asynccontextmanager
    async def connect(**kwargs):
        yield client

    monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', connect)
    monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                        lambda home: SimpleNamespace(state='ready', endpoint=object()))
    monkeypatch.setattr('hermes_cli.gateway_runtime.gateway_profile_home', lambda home: home)
    yield SimpleNamespace(conn=conn, task_id=task_id, run_id=task.current_run_id, admit=admit,
                          calls=calls, client=client, home=tmp_path)
    conn.close()
    asyncio.run(connection.close())
    authority.db.close()


@pytest.mark.parametrize('mismatch', ['profile', 'board', 'task', 'run'])
def test_foreign_native_policy_binding_cannot_select_a_session(owned_board, mismatch):
    board = owned_board
    context, binding = {}, {}
    if mismatch == 'profile':
        binding['db'] = str(board.home / 'foreign-profile' / 'state.db')
    else:
        context[{'board': 'db', 'task': 'task_id', 'run': 'run_id'}[mismatch]] = {
            'board': str(board.home / 'foreign-board.db'), 'task': 'foreign-task', 'run': board.run_id + 1}[mismatch]
    board.admit(board.run_id, context_changes=context, binding_changes=binding)
    token = kb.list_events(board.conn, board.task_id)[-1].id
    with pytest.raises(RuntimeError, match='another profile|identity is unverified'):
        kb.block_task(board.conn, board.task_id, kind='needs_input', expected_event_id=token)
    assert board.calls == []
    assert kb.get_task(board.conn, board.task_id).status == 'blocked'


def test_stop_replay_excludes_successor_owner_recorded_after_accepted_event(owned_board):
    board = owned_board
    previous = board.admit(board.run_id)
    token = kb.list_events(board.conn, board.task_id)[-1].id
    assert kb.block_task(board.conn, board.task_id, kind='needs_input', expected_event_id=token)
    accepted = kb.list_events(board.conn, board.task_id)[-1].id
    assert board.calls == [('session.resume', {'session_id': previous}),
                           ('session.interrupt', {'session_id': previous, 'execution_generation': 17})]
    assert kb.unblock_task(board.conn, board.task_id, expected_event_id=accepted)
    successor_run = kb.claim_task(board.conn, board.task_id).current_run_id
    successor = board.admit(successor_run)
    board.calls.clear()
    kanban_process_stop.request_native_stop(board.conn, board.task_id, accepted)
    assert [params['session_id'] for _method, params in board.calls] == [previous, previous]
    assert successor != previous and kb.get_task(board.conn, board.task_id).current_run_id == successor_run


@pytest.mark.parametrize('refusal', ['stale', 'rollback'])
def test_no_delivery_until_guarded_block_commits(owned_board, monkeypatch, refusal):
    board = owned_board
    board.admit(board.run_id)
    token = kb.list_events(board.conn, board.task_id)[-1].id
    if refusal == 'stale':
        assert not kb.block_task(board.conn, board.task_id, kind='needs_input', expected_event_id=token - 1)
    else:
        append = kb._append_event

        def failed_write(conn, task_id, kind, *args, **kwargs):
            if kind == 'blocked':
                raise sqlite3.IntegrityError('fixture block event refusal')
            return append(conn, task_id, kind, *args, **kwargs)

        monkeypatch.setattr(kb, '_append_event', failed_write)
        with pytest.raises(sqlite3.IntegrityError, match='fixture block event refusal'):
            kb.block_task(board.conn, board.task_id, kind='needs_input', expected_event_id=token)
    assert board.calls == []
    task = kb.get_task(board.conn, board.task_id)
    assert task.status == 'running' and task.current_run_id == board.run_id
    assert kb.list_events(board.conn, board.task_id)[-1].id == token


@pytest.mark.parametrize('failure', ['unavailable', 'cleanup-unknown'])
def test_native_delivery_failure_preserves_accepted_block(owned_board, monkeypatch, failure):
    board = owned_board
    board.admit(board.run_id)
    token = kb.list_events(board.conn, board.task_id)[-1].id
    if failure == 'unavailable':
        monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                            lambda home: SimpleNamespace(state='missing', endpoint=None))
    else:
        rpc = board.client.rpc

        async def uncertain_stop(method, **params):
            result = await rpc(method, **params)
            if method == 'session.interrupt':
                raise RuntimeError('process_stop_unknown')
            return result

        monkeypatch.setattr(board.client, 'rpc', uncertain_stop)
    reason = 'task blocked; process Stop not confirmed' if failure == 'unavailable' else 'process_stop_unknown'
    with pytest.raises(RuntimeError, match=reason):
        kb.block_task(board.conn, board.task_id, kind='needs_input', expected_event_id=token)
    assert len(board.calls) == (0 if failure == 'unavailable' else 2)
    assert kb.get_task(board.conn, board.task_id).status == 'blocked'
    assert kb.list_events(board.conn, board.task_id)[-1].kind == 'blocked'


@pytest.mark.asyncio
async def test_owner_loop_commits_and_schedules_self_rpc_without_blocking(owned_board, monkeypatch):
    board = owned_board
    board.admit(board.run_id)
    token = kb.list_events(board.conn, board.task_id)[-1].id
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    rpc = board.client.rpc

    async def held_rpc(method, **params):
        if method == 'session.interrupt':
            entered.set()
            await release.wait()
        result = await rpc(method, **params)
        if method == 'session.interrupt':
            finished.set()
        return result

    monkeypatch.setattr(board.client, 'rpc', held_rpc)
    assert kb.block_task(board.conn, board.task_id, kind='needs_input', expected_event_id=token)
    assert board.calls == [] and kb.get_task(board.conn, board.task_id).status == 'blocked'
    await asyncio.wait_for(entered.wait(), 10)
    release.set()
    await asyncio.wait_for(finished.wait(), 10)
    await async_wait_until(lambda: len(board.calls) == 2, timeout=10)
