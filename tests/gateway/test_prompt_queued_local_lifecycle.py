"""Queued controls survive real local creation and peer refresh (regression for #148846).

The model execution is held at an explicit boundary; creation, RPC admission, peer metadata,
steer tickets, FIFO claiming and settlement use the real gateway implementations.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gateway import run, session_finite
from gateway.config import GatewayConfig
from gateway.session_authority import admission_row, initialize_session_authority
from gateway.session_controls import AuthorityConnection
from gateway.session_runtime_workers import mutation_tasks
from tests.gateway.test_prompt_queued_transfer import BufferedAgent


class _HeldTurn:
    def __init__(self, consume=True):
        self.agent = BufferedAgent()
        self.consume = consume
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.ran = []
        self.taken = []

    async def execute(self, authority, ref, row):
        self.ran.append(row['payload']['text'])
        authority.adopt_agent(ref.session_id, row['generation'], self.agent)
        if row['request_id'] == 'head':
            self.started.set()
            await asyncio.wait_for(self.release.wait(), 10)
            if self.consume:
                text = self.agent._drain_pending_steer()
                if text is not None:
                    self.taken.append(text)
        return 'done'

    async def finish(self, authority, session_id):
        self.release.set()
        await asyncio.wait_for(authority.sessions[session_id].task, 10)


async def _rpc(connection, method, session_id=None, **params):
    if session_id is not None:
        params['session_id'] = session_id
    return await connection.dispatch({'id': method, 'method': method, 'params': params})


async def _submit(connection, session_id, request_id, **extra):
    reply = await _rpc(connection, 'prompt.submit', session_id,
                       submission_id=request_id, text=request_id, **extra)
    assert 'error' not in reply, reply
    return reply['result']['admission_id']


def _admission(authority, admission_id):
    """One admission row in the authority's own dict shape (what a pending row is built from)."""
    return admission_row(authority.persistence.admissions.get(admission_id=admission_id))


def _pending_controls(authority, admission_id):
    return asdict(authority._pending_receipt(_admission(authority, admission_id))).get('controls') or []


@asynccontextmanager
async def _local_chat(tmp_path, monkeypatch, turn):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *args: {
        'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    runner = run.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(
        runner, profile_id=str(tmp_path), instance_id='queued-controls-test')
    connection = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    monkeypatch.setattr(session_finite, 'execute_finite_admission', turn.execute)
    try:
        created = await _rpc(connection, 'session.create', request_id='desktop', source='gui',
                             cwd=str(tmp_path), model='fixture', toolsets=[])
        assert 'error' not in created, created
        sid = created['result']['stored_session_id']
        live = authority.sessions[sid]
        # This is the production peer refresh, not a hand-written UPDATE of the test row.
        runner.session_store._record_gateway_session_peer(sid, live.route, live.source)
        stored = authority.persistence.sessions.get(sid)
        assert (stored.source, stored.created_source) == ('local', 'desktop')
        assert authority.persistence.bindings.receipt(sid).policy['source'] == 'gui'
        yield authority, connection, sid
    finally:
        turn.release.set()
        tasks = [live.task for live in authority.sessions.values() if live.task is not None]
        try:
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
            await connection.close()
            writes = mutation_tasks(authority)
            if writes:
                await asyncio.wait_for(asyncio.gather(*writes, return_exceptions=True), 10)
        finally:
            runner.session_store.close_all_db_handles()
            runner.close_all_session_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('consumed', [True, False])
@pytest.mark.parametrize('clarify_batch', [False, True], ids=['plain', 'clarify-batch'])
async def test_refreshed_desktop_steer_is_consumed_once_or_returns_to_its_fifo(
        tmp_path, monkeypatch, consumed, clarify_batch):
    turn = _HeldTurn(consume=consumed)
    async with _local_chat(tmp_path, monkeypatch, turn) as (authority, connection, sid):
        await _submit(connection, sid, 'head')
        await asyncio.wait_for(turn.started.wait(), 10)
        follower = await _submit(connection, sid, 'follower', clarify_batch=clarify_batch)
        await _submit(connection, sid, 'tail')
        row = _admission(authority, follower)
        assert row['payload']['text'] == 'follower'
        assert row['payload'].get('cli_turn_v1', {}) == ({'clarify_batch': True} if clarify_batch else {})
        assert _pending_controls(authority, follower) == ['steer', 'replace']

        first = await _rpc(connection, 'prompt.steer_queued', sid, admission_id=follower)
        assert 'error' not in first, first
        again = await _rpc(connection, 'prompt.steer_queued', sid, admission_id=follower)
        assert again.get('result') == first['result'], again
        await turn.finish(authority, sid)

        rows = {row.request_id: row for row in authority.persistence.admissions.list(session_id=sid, pending_only=False)}
        assert (rows['follower'].status, rows['follower'].outcome) == (
            'terminal', 'steered' if consumed else 'completed')
        assert turn.taken == (['follower'] if consumed else [])
        assert turn.ran == (['head', 'tail'] if consumed else ['head', 'follower', 'tail'])
        assert turn.agent._drain_pending_steer() is None, 'no steer survives for an unrelated turn'


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [
    {},
    {'api_text': 'model-facing text'},
    {'model_override': {'model': 'other-model'}},
    {'turn_author': {'id': 'alice', 'name': 'Alice', 'is_bot': False}},
], ids=['viewer-only', 'api-text', 'model-override', 'turn-author'])
async def test_refreshed_desktop_replace_preserves_fifo_and_refuses_whole_turn_inputs(tmp_path, monkeypatch, extra):
    turn = _HeldTurn()
    async with _local_chat(tmp_path, monkeypatch, turn) as (authority, connection, sid):
        await _submit(connection, sid, 'head')
        await asyncio.wait_for(turn.started.wait(), 10)
        follower = await _submit(connection, sid, 'follower', clarify_batch=True, **extra)
        await _submit(connection, sid, 'tail')
        before = _admission(authority, follower)
        replacement = {'admission_id': follower, 'submission_id': 'edited', 'text': 'edited'}

        if extra:
            assert _pending_controls(authority, follower) == []
            for method, params in (('prompt.steer_queued', {'admission_id': follower}),
                                   ('prompt.replace_queued', replacement)):
                refused = await _rpc(connection, method, sid, **params)
                assert refused.get('error', {}).get('data', {}).get('reason') == 'unsupported_control', refused
            after = _admission(authority, follower)
            assert after['status'] == 'queued'
            assert after['payload'] == before['payload'], 'a refusal must preserve all admitted turn input'
        else:
            assert _pending_controls(authority, follower) == ['steer', 'replace']
            edited = await _rpc(connection, 'prompt.replace_queued', sid, **replacement)
            assert 'error' not in edited, edited
            again = await _rpc(connection, 'prompt.replace_queued', sid, **replacement)
            assert again.get('result') == edited['result'], again
            retired = _admission(authority, follower)
            assert (retired['status'], retired['outcome']) == ('terminal', 'cancelled')

        await turn.finish(authority, sid)
        assert turn.taken == []
        assert turn.ran == (['head', 'follower', 'tail'] if extra else ['head', 'tail', 'edited'])
        rows = {row.request_id: row for row in authority.persistence.admissions.list(session_id=sid, pending_only=False)}
        for request_id in ('head', 'tail', 'follower' if extra else 'edited'):
            assert (rows[request_id].status, rows[request_id].outcome) == ('terminal', 'completed')
        if not extra:
            assert (rows['follower'].status, rows['follower'].outcome) == ('terminal', 'cancelled')
