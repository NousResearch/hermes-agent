"""``session.automation``: the classic CLI's /goal, /subgoal, /loop, /heartbeat managers run by the owner.

Real authority + SQLite: every mutation lands in the owner's ``state_meta`` row for the session's physical
target, the goal judge runs here (patched), a manager refusal comes back as main's exception text, and a
session driven this way is skipped by the gateway's own post-turn goal hook (never judged twice).
"""
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    from hermes_cli import goals

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path / 'fixture'), instance_id='owner')
    monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)  # the owner's own store
    viewer = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': str(tmp_path / 'fixture')})
    ref = create_local_session(authority, viewer.actor, {'request_id': 'r', 'source': 'cli',
                                                         'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    target = local_receipt(authority.db, ref.session_id)['entry']['session_id']

    async def call(kind, name, *args, **kwargs):
        reply = await viewer.dispatch({'id': 1, 'method': 'session.automation', 'params': {
            'session_id': ref.session_id, 'kind': kind, 'call': name, 'args': list(args), 'kwargs': kwargs}})
        return reply.get('result', reply.get('error'))
    try:
        yield SimpleNamespace(call=call, db=authority.db, target=target, authority=authority, ref=ref, viewer=viewer)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_goal_mutations_land_in_the_owners_row(owner):
    reply = await owner.call('goal', 'set', 'ship it', max_turns=5)
    state = json.loads(reply['state'])
    assert (state['goal'], state['status'], state['max_turns']) == ('ship it', 'active', 5)
    assert json.loads(owner.db.get_meta('goal:' + owner.target))['goal'] == 'ship it'
    reply = await owner.call('goal', 'add_subgoal', 'and test it')
    assert reply['value'] == 'and test it'
    assert json.loads(owner.db.get_meta('goal:' + owner.target))['subgoals'] == ['and test it']
    reply = await owner.call('goal', 'remove_subgoal', 9)
    assert reply['error'] == {'type': 'IndexError', 'message': 'index out of range (1..1)'}
    await owner.call('goal', 'pause', reason='user-paused')
    assert json.loads(owner.db.get_meta('goal:' + owner.target))['status'] == 'paused'
    await owner.call('goal', 'clear')
    assert json.loads(owner.db.get_meta('goal:' + owner.target))['status'] == 'cleared'


@pytest.mark.asyncio
async def test_the_goal_judge_runs_on_the_owner(owner, monkeypatch):
    from hermes_cli import goals
    seen = []

    def judge(goal, response, **kw):
        seen.append((goal, response))
        return 'continue', 'only the outline exists', False, None, False
    monkeypatch.setattr(goals, 'judge_goal', judge)
    await owner.call('goal', 'set', 'write the report')
    reply = await owner.call('goal', 'evaluate_after_turn', 'drafted the outline')
    decision = reply['value']
    assert seen == [('write the report', 'drafted the outline')]
    assert decision['should_continue'] is True
    assert decision['message'] == '↻ Continuing toward goal (1/20): only the outline exists'
    assert decision['continuation_prompt'].startswith('[Continuing toward your standing goal]')
    assert json.loads(owner.db.get_meta('goal:' + owner.target))['turns_used'] == 1

    from gateway.session_viewer_automation import viewer_driven
    assert not viewer_driven(owner.authority, owner.ref.session_id), 'no subscription: the gateway hooks run'
    owner.viewer.subscriptions[owner.ref.session_id] = 'sub-1'
    assert viewer_driven(owner.authority, owner.ref.session_id), 'the attached CLI judges; the gateway hook skips'


@pytest.mark.asyncio
async def test_loop_and_heartbeat_ticks_are_claimed_on_the_owner(owner):
    await owner.call('loop', 'set', 'check the queue', interval_seconds=60, times=1)
    wake = (await owner.call('loop', 'fire_tick'))['value']
    assert wake.startswith('[/loop wakeup #1')
    reply = await owner.call('loop', 'complete_tick', 'nothing new')
    assert reply['value']['message'] == '✓ Loop finished — ran 1/1 times.'
    assert json.loads(owner.db.get_meta('loop:' + owner.target))['status'] == 'done'
    reply = await owner.call('heartbeat', 'set', 'pulse', 60)
    assert json.loads(reply['state'])['interval_seconds'] == 60
    assert (await owner.call('heartbeat', 'clear'))['value'] is True
    assert json.loads(owner.db.get_meta('heartbeat:' + owner.target))['status'] == 'cleared'


@pytest.mark.asyncio
async def test_unknown_calls_are_refused(owner):
    assert (await owner.call('goal', '_save'))['message'] == 'invalid_params'
    assert (await owner.call('cron', 'state'))['message'] == 'invalid_params'


@pytest.mark.asyncio
async def test_cli_driven_loop_changes_repaint_the_other_viewers(owner):
    """A classic-CLI viewer drives its /loop through this verb; an Ink or Desktop viewer of the same
    session paints the loop bar from ``session.control.update``, so every mutation publishes it (reads don't)."""
    updates = []
    owner.authority.sessions[owner.ref.session_id].event_stream.observers.add(
        lambda frame: frame['params']['type'] == 'session.control.update' and updates.append(frame['params']['payload']))
    await owner.call('loop', 'set', 'check the queue', interval_seconds=60)
    await owner.call('loop', 'fire_tick')
    assert [u['control']['loop']['awaiting_response'] for u in updates] == [False, True]
    await owner.call('loop', 'state')
    await owner.call('loop', 'goal_blocks_tick')
    assert len(updates) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('kind, call', [('goal', 'is_waiting'), ('loop', 'goal_blocks_tick')])
async def test_a_read_that_clears_a_goal_barrier_repaints_the_other_viewers(owner, kind, call):
    """``is_waiting`` / ``goal_blocks_tick`` drop a satisfied barrier (the goal resumes, the loop defers to
    it): that write must reach the other viewers even though an unchanged poll stays quiet."""
    import asyncio
    import sys
    child = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import sys; sys.stdin.read()',
                                                 stdin=asyncio.subprocess.PIPE)
    try:
        await owner.call('goal', 'set', 'wait for the child')
        await owner.call('goal', 'wait_on', child.pid)
        await owner.call('loop', 'set', 'check the queue', interval_seconds=60)
        updates = []
        owner.authority.sessions[owner.ref.session_id].event_stream.observers.add(
            lambda frame: frame['params']['type'] == 'session.control.update'
            and updates.append(frame['params']['payload']))
        await owner.call(kind, call)
        assert updates == []  # barrier still holds: nothing changed, nothing published
    finally:
        child.stdin.close()
        await asyncio.wait_for(child.wait(), 10)
    await owner.call(kind, call)
    assert len(updates) == 1 and 'wait_barrier' not in updates[0]['control']['goal']
    assert updates[0]['control']['loop']['deferred_by_goal'] is True
