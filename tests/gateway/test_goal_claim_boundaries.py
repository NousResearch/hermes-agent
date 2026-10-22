"""Goal wakes are fenced at the real authority claim boundary, including disconnects."""
import asyncio
import json
import threading

import pytest

from gateway.config import Platform
from gateway.session_automation import admit_route_wake
from gateway.session_contract import Principal, Submission
from gateway.session_local import create_local_session
from hermes_state_runtime import get_session_admission, list_session_admissions


async def queued_goal(owner, monkeypatch, tmp_path):
    from hermes_cli import goals

    runner = owner.runner
    runner.adapters = {}
    runner._adapters_for_profile = lambda _profile=None: runner.adapters
    runner._adapter_for_source = lambda source: runner.adapters[source.platform]
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    actor = Principal('human', owner.profile_id,
                      frozenset({'session:create', 'session:submit', 'session:control'}), 'test')
    ref = create_local_session(owner, actor, dict(request_id='create', source='cli',
        cwd=str(tmp_path), model='fixture', toolsets=[]))
    live = owner.sessions[ref.session_id]
    runner._session_key_for_source = lambda source: live.route
    entry = runner.session_store.lookup_by_session_key(live.route)
    monkeypatch.setattr(goals, '_get_session_db', lambda: owner.db)
    goal = goals.GoalManager(entry.session_id)
    goal.set('finish fixture')
    metadata = {
        'goal_continuation_generation': goal.state.continuation_generation,
        'goal_continuation_created_at': goal.state.created_at,
    }
    assert await admit_route_wake(runner, runner.adapters[Platform.LOCAL], live.source,
                                 'GOAL_WAKE', 'goal-continuation', metadata)
    wake = list_session_admissions(owner.db, session_id=ref.session_id)[0]
    user = await owner.submit(actor, Submission('human-next', ref, {'text': 'ordinary input'}, 'queue'))
    return ref, live, entry, wake, user


async def legacy_queued_goal(owner, monkeypatch, tmp_path):
    ref, live, entry, wake, user = await queued_goal(owner, monkeypatch, tmp_path)
    key = 'goal:' + entry.session_id
    raw = json.loads(owner.db.get_meta(key))
    raw.pop('continuation_generation', None)
    owner.db._execute_write(lambda conn: conn.execute(
        'UPDATE state_meta SET value=? WHERE key=?', (json.dumps(raw), key)))
    owner.db._execute_write(lambda conn: conn.execute(
        'UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
        (json.dumps({'text': wake['payload']['text'], 'local_automation_v1': {
            key: value for key, value in wake['payload']['local_automation_v1'].items()
            if key not in {'goal_continuation_generation', 'goal_continuation_created_at'}
        }}), wake['admission_id'])))
    wake = next(row for row in list_session_admissions(owner.db, session_id=ref.session_id)
                if row['admission_id'] == wake['admission_id'])
    return ref, live, entry, wake, user


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['pause', 'clear'])
@pytest.mark.parametrize('phase', ['before_preflight', 'after_preflight'])
async def test_inactive_goal_is_fenced_and_human_successor_runs(owner, monkeypatch, tmp_path, action, phase):
    from gateway import session_authority
    from hermes_cli import goals

    ref, live, entry, wake, user = await queued_goal(owner, monkeypatch, tmp_path)
    claim = session_authority.claim_session_input

    def pause_after_async_preflight(*args, **kwargs):
        getattr(goals.GoalManager(entry.session_id), action)()
        return claim(*args, **kwargs)

    if phase == 'before_preflight':
        getattr(goals.GoalManager(entry.session_id), action)()
    else:
        monkeypatch.setattr(session_authority, 'claim_session_input', pause_after_async_preflight)
    frames = []
    live.event_stream.observers.add(frames.append)
    waiter = owner.waiters.setdefault(wake['admission_id'], asyncio.get_running_loop().create_future())
    generation = owner.db.get_session(ref.session_id)['runtime_generation']
    claimed, _ = await owner._claim_next(ref, live)
    assert claimed is False
    saved = get_session_admission(owner.db, admission_id=wake['admission_id'])
    assert (saved['status'], saved['outcome']) == ('terminal', 'cancelled')
    assert waiter.done()
    assert owner.db.get_session(ref.session_id)['runtime_generation'] == generation
    complete = [f['params'] for f in frames if f['params']['type'] == 'message.complete']
    assert len(complete) == 1 and complete[0]['admission_id'] == wake['admission_id']
    next_row, _ = await owner._claim_next(ref, live)
    assert next_row['admission_id'] == user.admission_id
    assert next_row['status'] == 'started'


@pytest.mark.asyncio
async def test_caller_cancellation_cannot_drop_goal_withdrawal_settlement(owner, monkeypatch, tmp_path):
    from gateway import session_authority
    from hermes_cli import goals

    ref, live, entry, wake, user = await queued_goal(owner, monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    claim = session_authority.claim_session_input
    def blocked_claim(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return claim(*args, **kwargs)
    monkeypatch.setattr(session_authority, 'claim_session_input', blocked_claim)
    waiter = owner.waiters.setdefault(wake['admission_id'], asyncio.get_running_loop().create_future())
    task = asyncio.create_task(owner._claim_next(ref, live))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        goals.GoalManager(entry.session_id).pause()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.wait_for(asyncio.shield(waiter), 10)
        pending = tuple(getattr(owner, '_mutation_tasks', ()))
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), 10)
        saved = get_session_admission(owner.db, admission_id=wake['admission_id'])
        assert (saved['status'], saved['outcome']) == ('terminal', 'cancelled')
        assert get_session_admission(owner.db, admission_id=user.admission_id)['status'] == 'queued'
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_pause_resume_rejects_old_goal_wake_when_cleanup_misses(owner, monkeypatch, tmp_path):
    from hermes_cli import goals

    ref, live, entry, wake, user = await queued_goal(owner, monkeypatch, tmp_path)
    mgr = goals.GoalManager(entry.session_id)
    mgr.pause()
    mgr.resume()
    claimed, _ = await owner._claim_next(ref, live)
    assert claimed is False
    saved = get_session_admission(owner.db, admission_id=wake['admission_id'])
    assert (saved['status'], saved['outcome']) == ('terminal', 'cancelled')
    successor, _ = await owner._claim_next(ref, live)
    assert successor['admission_id'] == user.admission_id


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['unchanged', 'pause_resume', 'set'])
async def test_legacy_goal_wake_requires_untouched_legacy_state(owner, monkeypatch, tmp_path, mutation):
    from hermes_cli import goals

    ref, live, entry, wake, user = await legacy_queued_goal(owner, monkeypatch, tmp_path)
    if mutation == 'pause_resume':
        manager = goals.GoalManager(entry.session_id)
        manager.pause()
        manager.resume()
    elif mutation == 'set':
        goals.GoalManager(entry.session_id).set('replacement goal')

    claimed, _ = await owner._claim_next(ref, live)
    if mutation == 'unchanged':
        assert claimed['admission_id'] == wake['admission_id']
        assert claimed['status'] == 'started'
        return
    saved = get_session_admission(owner.db, admission_id=wake['admission_id'])
    assert (claimed, saved['status'], saved['outcome']) == (False, 'terminal', 'cancelled')
    successor, _ = await owner._claim_next(ref, live)
    assert successor['admission_id'] == user.admission_id


@pytest.mark.asyncio
@pytest.mark.parametrize('status,expected', [('active', 'started'), ('paused', 'terminal')])
async def test_queued_goal_uses_compressed_tip_not_archived_parent(owner, monkeypatch, tmp_path, status, expected):
    from hermes_cli import goals
    from tests.gateway.test_local_resume_compression_child import _compress

    ref, live, entry, wake, _user = await queued_goal(owner, monkeypatch, tmp_path)
    root = entry.session_id
    store = owner.runner.session_store
    store._record_gateway_session_peer(root, live.route, live.source)
    _compress(owner.db, root, 'tip', owner.db.get_session(root)['source'])
    store._record_gateway_session_peer('tip', live.route, live.source)
    assert owner.db.get_compression_tip(root) == 'tip'
    assert goals.migrate_goal_to_session(root, 'tip')
    if status == 'paused':
        goals.GoalManager('tip').pause()
    await owner._claim_next(ref, live)
    assert get_session_admission(owner.db, admission_id=wake['admission_id'])['status'] == expected
