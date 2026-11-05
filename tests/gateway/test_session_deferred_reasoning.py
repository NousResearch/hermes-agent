"""A reasoning level picked while a reply streams lands for the next turn, like a model pick (MP-24)."""
import asyncio
import threading

import pytest

from tests.gateway.test_session_mutation_model_busy import _owner, _run_turn, _running_model
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('picks, expected', [
    ([('reasoning', {'effort': 'low'}), ('model', {'model': 'picked'}), ('reasoning', {'effort': 'high'})],
     ('picked', 'high')),
    ([('model', {'model': 'picked', 'reasoning': 'low'}), ('reasoning', {'effort': 'high'})], ('picked', 'high')),
    ([('reasoning', {'effort': 'high'}), ('model', {'model': 'picked', 'reasoning': 'low'})], ('picked', 'low')),
], ids=['level-model-level', 'model-with-level-then-level', 'level-then-model-with-level'])
async def test_levels_picked_during_a_running_turn_apply_to_the_next_turn_in_pick_order(
        tmp_path, monkeypatch, picks, expected):
    from gateway.session_contract import Submission
    from gateway.session_policy import policy_for_source
    started, release = threading.Event(), asyncio.Event()
    seen, state = [], {}

    async def handle(event):
        live = state['authority'].sessions[state['ref'].session_id]
        effort = (policy_for_source(state['authority'].runner, live.source).reasoning_config or {}).get('effort')
        seen.append((event.text, _running_model(state['authority'], state['ref']), effort))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    authority.runner._set_session_reasoning_override = lambda *a: None
    try:
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        for n, (operation, payload) in enumerate(picks):
            snap = db.get_session(ref.session_id)
            reply = await owner.dispatch({'id': n, 'method': 'session.mutate', 'params': {
                'session_id': ref.session_id, 'request_id': f'pick-{n}', 'expected_revision': snap['runtime_revision'],
                'expected_generation': snap['runtime_generation'], 'operation': operation, 'payload': payload}})
            assert reply.get('result', {}).get('status') == 'deferred', reply
        release.set()
        await _run_turn(authority, owner, db, ref, 'NEXT')
        assert seen[-1] == ('NEXT', *expected), seen
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation, payload, effort', [
    ('reasoning', {'effort': 'high'}, 'high'),
    ('model', {'model': 'picked'}, 'low'),  # a model picked without a level keeps the deferred one
], ids=['level', 'model'])
async def test_a_pick_committed_after_the_turn_settles_supersedes_only_what_it_picks(
        tmp_path, monkeypatch, operation, payload, effort):
    from gateway import session_deferred_pick
    from gateway.session_contract import Submission
    from gateway.session_policy import policy_for_source
    import hermes_state_runtime as rt
    started, release = threading.Event(), asyncio.Event()
    seen, state = [], {}

    async def handle(event):
        live = state['authority'].sessions[state['ref'].session_id]
        effort = (policy_for_source(state['authority'].runner, live.source).reasoning_config or {}).get('effort')
        seen.append((event.text, _running_model(state['authority'], state['ref']), effort))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'

    async def pick(n, operation, payload):
        snap = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': n, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': f'pick-{n}', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': operation, 'payload': payload}})
        return reply['result'].get('status')
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    authority.runner._set_session_reasoning_override = lambda *a: None
    apply = session_deferred_pick.apply_deferred_pick
    try:
        first = await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        assert await pick(0, 'reasoning', {'effort': 'low'}) == 'deferred'
        # The newer pick takes the mutation lock before the drain reaches the boundary.
        monkeypatch.setattr(session_deferred_pick, 'apply_deferred_pick', lambda *a: asyncio.sleep(0))
        release.set()
        await async_wait_until(
            lambda: rt.get_session_admission(db, admission_id=first.admission_id)['status'] == 'terminal',
            timeout=10, what='the running turn settled')
        assert await pick(1, operation, payload) is None  # committed now, not deferred
        monkeypatch.setattr(session_deferred_pick, 'apply_deferred_pick', apply)
        await _run_turn(authority, owner, db, ref, 'NEXT')
        assert seen[-1] == ('NEXT', payload.get('model', seen[0][1]), effort), seen
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()
