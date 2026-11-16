"""Desktop's typed /reasoning and Move to project on the shared owner (B14, B21 / C-hx-2).

Main applied both to the live sidecar session. On the owner the next turn runs under the frozen
LocalSessionPolicy, so each is a revision-fenced ``session.mutate`` that rewrites that policy.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


def _owner(tmp_path, monkeypatch, handle):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _set_session_reasoning_override=lambda route, value: None,
                             _handle_message=handle, _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    start = tmp_path / 'project-a'
    start.mkdir()
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='gui', cwd=str(start),
                                                            model='frozen', toolsets=[], reasoning='low'))
    return store, db, authority, owner, ref


def _mutation(db, ref, request_id, operation, payload):
    snap = db.get_session(ref.session_id)
    return {'id': request_id, 'method': 'session.mutate', 'params': {
        'session_id': ref.session_id, 'request_id': request_id, 'operation': operation, 'payload': payload,
        'expected_revision': snap['runtime_revision'], 'expected_generation': snap['runtime_generation']}}


@pytest.mark.asyncio
async def test_session_reasoning_mutation_is_the_next_turns_reasoning(tmp_path, monkeypatch):
    from gateway.session_policy import policy_for_source
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    live = authority.sessions[ref.session_id]
    try:
        before = db.get_session(ref.session_id)
        frozen_prompt = before['system_prompt']
        response = await owner.dispatch(_mutation(db, ref, 'r1', 'reasoning', {'effort': 'high'}))
        assert 'result' in response, response
        assert response['result']['effort'] == 'high'
        # The policy the next turn executes under, and what every viewer's session.info reports.
        assert policy_for_source(authority.runner, live.source).reasoning_config == {'enabled': True, 'effort': 'high'}
        info = await owner.dispatch({'id': 2, 'method': 'session.info', 'params': {'session_id': ref.session_id}})
        assert info['result']['reasoning_effort'] == 'high'
        after = db.get_session(ref.session_id)
        assert after['runtime_revision'] > before['runtime_revision']
        assert after['runtime_generation'] == before['runtime_generation'] + 1
        assert after['system_prompt'] == frozen_prompt
        # Durable: a restarted owner restores the new level from the receipt.
        from hermes_state_local import local_receipt
        assert json.loads(local_receipt(db, ref.session_id)['policy']['config_json'])['agent']['reasoning_effort'] == 'high'
        bad = await owner.dispatch(_mutation(db, ref, 'r2', 'reasoning', {'effort': 'hgih'}))
        assert bad['error']['data']['reason'] == 'invalid_params'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_move_to_project_rehomes_the_policy_and_row_and_refuses_mid_turn(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    from gateway.session_policy import policy_for_source
    from hermes_state_local import local_receipt
    import hermes_state_runtime as rt
    started, release, seen = asyncio.Event(), asyncio.Event(), []

    async def handle(event):
        seen.append(policy_for_source(authority.runner, event.source).cwd)
        started.set()
        await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    target = tmp_path / 'project-b'
    target.mkdir()
    try:
        queued = await authority.submit(owner.actor, Submission('turn-1', ref, {'text': 'one'}, 'queue'))
        await asyncio.wait_for(started.wait(), 10)
        busy = await owner.dispatch(_mutation(db, ref, 'mv-busy', 'workspace', {'cwd': str(target)}))
        assert busy['error']['data']['reason'] == 'session_busy', busy
        assert db.get_session(ref.session_id)['cwd'] == str(tmp_path / 'project-a')
        release.set()
        await async_wait_until(lambda: rt.get_session_admission(db, admission_id=queued.admission_id)['status'] == 'terminal',
                               what='turn-1 to settle')
        moved = await owner.dispatch(_mutation(db, ref, 'mv', 'workspace', {'cwd': str(target)}))
        assert moved['result']['cwd'] == str(target), moved
        policy = local_receipt(db, ref.session_id)['policy']
        assert policy['cwd'] == str(target) and json.loads(policy['terminal_json'])['TERMINAL_CWD'] == str(target)
        assert db.get_session(ref.session_id)['cwd'] == str(target)
        info = await owner.dispatch({'id': 3, 'method': 'session.info', 'params': {'session_id': ref.session_id}})
        assert info['result']['cwd'] == str(target)
        started.clear()
        second = await authority.submit(owner.actor, Submission('turn-2', ref, {'text': 'two'}, 'queue'))
        await asyncio.wait_for(started.wait(), 10)
        await async_wait_until(lambda: rt.get_session_admission(db, admission_id=second.admission_id)['status'] == 'terminal',
                               what='turn-2 to settle')
        assert seen == [str(tmp_path / 'project-a'), str(target)]
        missing = await owner.dispatch(_mutation(db, ref, 'mv-missing', 'workspace', {'cwd': str(tmp_path / 'nope')}))
        assert missing['error']['data']['reason'] == 'invalid_params'
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()
