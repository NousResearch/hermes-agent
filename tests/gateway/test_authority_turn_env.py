"""Per-turn client values cross real authority admission and claim, never the ledger."""
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from agent.session_client_env import bound_turn_env, toolchain_environ
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    (Path(store._db.db_path).resolve().parent / 'config.yaml').write_text(
        'terminal:\n  client_env_turn_passthrough: [JOB_RUN_ID]\n')
    seen = []

    async def answer(event):
        seen.append((dict(bound_turn_env()), toolchain_environ({})))
        return 'done'

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    store._db.create_session('s', source='telegram')
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='turn'), 's')
    authority.launch_sources['s'] = 'acp'
    wire = SimpleNamespace(write=lambda frame: True)
    connection = AuthorityConnection(authority, wire, {'user_id': 'editor'})
    resumed = await connection.dispatch({'id': 0, 'method': 'session.resume', 'params': {'session_id': 's'}})
    assert 'error' not in resumed, resumed
    yield authority, connection, seen
    await connection.close()
    task = authority.sessions['s'].task
    if task is not None:
        await asyncio.wait_for(task, 5)
    store.close_all_db_handles()


async def rpc(connection, method='prompt.submit', **params):
    return await connection.dispatch({'id': 1, 'method': method, 'params': {'session_id': 's', **params}})


async def settled(authority):
    await async_wait_until(lambda: all(row['status'] == 'terminal' for row in
        list_session_admissions(authority.db, session_id='s', pending_only=False)), timeout=5, what='turn settlement')


@pytest.mark.asyncio
async def test_admission_scopes_values_without_persisting_or_logging(owner, caplog):
    authority, connection, seen = owner
    secret = 'private-authority-run-value'
    with caplog.at_level(logging.DEBUG):
        reply = await rpc(connection, input_id='one', text='hello',
                          turn_env={'JOB_RUN_ID': secret, 'UNDECLARED': 'dropped'})
        assert 'error' not in reply, reply
        await settled(authority)
        reply = await rpc(connection, input_id='two', text='next')
        assert 'error' not in reply, reply
        await settled(authority)
    assert seen == [({'JOB_RUN_ID': secret}, {'JOB_RUN_ID': secret}), ({}, {})]
    assert bound_turn_env() == {}
    with authority.db._read_ctx() as conn:
        durable = '\n'.join(conn.iterdump())
    assert secret not in durable and 'turn_env' not in durable and 'UNDECLARED' not in durable
    assert secret not in caplog.text
    assert not authority.turn_envs


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [['bad'], {'JOB_RUN_ID': 3}, {'JOB_RUN_ID': 'x' * 33000}])
async def test_malformed_map_is_invalid_params(owner, bad):
    authority, connection, seen = owner
    reply = await rpc(connection, input_id='bad', text='hello', turn_env=bad)
    assert reply['error']['data']['reason'] == 'invalid_params'
    assert not list_session_admissions(authority.db, session_id='s', pending_only=False)
    assert not seen


@pytest.mark.asyncio
@pytest.mark.parametrize('first', [{'JOB_RUN_ID': 'first-private'}, {}])
async def test_duplicate_resend_keeps_first_values_even_after_execution(owner, first):
    authority, connection, seen = owner
    schedule = authority._schedule
    authority._schedule = lambda ref: None
    try:
        original = await rpc(connection, input_id='same', text='hello', turn_env=first)
        assert 'error' not in original, original
        retry = await rpc(connection, input_id='same', text='hello', turn_env={'JOB_RUN_ID': 'replacement'})
        assert retry['result']['admission_id'] == original['result']['admission_id']
    finally:
        authority._schedule = schedule
    from gateway.session_authority import SessionRef
    schedule(SessionRef('p', 's'))
    await settled(authority)
    retry = await rpc(connection, input_id='same', text='hello', turn_env={'JOB_RUN_ID': 'later'})
    assert retry['result']['admission_id'] == original['result']['admission_id']
    assert seen == [(first, first)]
    assert not authority.turn_envs


@pytest.mark.asyncio
async def test_replacement_and_cancellation_release_transient_values(owner):
    authority, connection, seen = owner
    schedule = authority._schedule
    authority._schedule = lambda ref: None
    try:
        old = await rpc(connection, input_id='old', text='hello', turn_env={'JOB_RUN_ID': 'old-private'})
        assert 'error' not in old, old
        replacement = await rpc(connection, 'prompt.replace_queued', submission_id='new', text='edited',
                                admission_id=old['result']['admission_id'], turn_env={'JOB_RUN_ID': 'new-private'})
        assert 'error' not in replacement, replacement
        assert old['result']['admission_id'] not in authority.turn_envs
        cancelled = await rpc(connection, 'prompt.cancel', admission_id=replacement['result']['admission_id'])
        assert 'error' not in cancelled, cancelled
        assert not authority.turn_envs
        assert not seen
    finally:
        authority._schedule = schedule


@pytest.mark.asyncio
async def test_recovered_queue_does_not_restore_or_accept_retry_values(owner):
    authority, connection, seen = owner
    schedule = authority._schedule
    authority._schedule = lambda ref: None
    try:
        reply = await rpc(connection, input_id='lost', text='hello', turn_env={'JOB_RUN_ID': 'lost-private'})
        assert 'error' not in reply, reply
        authority.turn_envs.clear()  # owner restart loses only the transient map
        reply = await rpc(connection, input_id='lost', text='hello', turn_env={'JOB_RUN_ID': 'retry-private'})
        assert 'error' not in reply, reply
    finally:
        authority._schedule = schedule
    from gateway.session_authority import SessionRef
    schedule(SessionRef('p', 's'))
    await settled(authority)
    assert seen == [({}, {})]


@pytest.mark.asyncio
async def test_running_retry_does_not_replace_values_or_leak_into_queued_turn(owner):
    authority, connection, seen = owner
    started, release = asyncio.Event(), asyncio.Event()
    answer = authority.runner._handle_message

    async def blocked(event):
        if event.text == 'running':
            started.set()
            await release.wait()
        return await answer(event)

    authority.runner._handle_message = blocked
    try:
        original = await rpc(connection, input_id='running', text='running', turn_env={'JOB_RUN_ID': 'first'})
        assert 'error' not in original, original
        await asyncio.wait_for(started.wait(), 5)
        assert not authority.turn_envs  # consumed at execution, not retained for retries
        retry = await rpc(connection, input_id='running', text='running', turn_env={'JOB_RUN_ID': 'changed'})
        assert retry['result']['admission_id'] == original['result']['admission_id']
        queued = await rpc(connection, input_id='queued', text='queued')
        assert 'error' not in queued, queued
    finally:
        release.set()
    await settled(authority)
    assert seen == [({'JOB_RUN_ID': 'first'}, {'JOB_RUN_ID': 'first'}), ({}, {})]


@pytest.mark.asyncio
async def test_replacement_executes_its_own_values_and_conflict_adds_nothing(owner):
    authority, connection, seen = owner
    schedule = authority._schedule
    authority._schedule = lambda ref: None
    try:
        old = await rpc(connection, input_id='old', text='hello', turn_env={'JOB_RUN_ID': 'old'})
        assert 'error' not in old, old
        replacement = await rpc(connection, 'prompt.replace_queued', submission_id='new', text='edited',
                                admission_id=old['result']['admission_id'], turn_env={'JOB_RUN_ID': 'new'})
        assert 'error' not in replacement, replacement
        conflict = await rpc(connection, input_id='new', text='different', turn_env={'JOB_RUN_ID': 'conflict'})
        assert conflict['error']['data']['reason'] == 'admission_conflict'
        assert authority.turn_envs == {replacement['result']['admission_id']: {'JOB_RUN_ID': 'new'}}
    finally:
        authority._schedule = schedule
    from gateway.session_authority import SessionRef
    schedule(SessionRef('p', 's'))
    await settled(authority)
    assert seen == [({'JOB_RUN_ID': 'new'}, {'JOB_RUN_ID': 'new'})]
    assert not authority.turn_envs
