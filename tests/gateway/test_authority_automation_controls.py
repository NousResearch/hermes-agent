"""``session.control`` on the session owner: the composer's automation create/edit actions.

Desktop's store sends ``include_loop_min_interval`` on every read and action, and its composer
creates or edits a goal, loop or heartbeat through ``session.control``. The authority registered
``session.control`` for the card buttons only: the read refused the negotiation key as an unknown
param, and every ``*.create`` / ``*.update`` action answered ``invalid_params``. These tests drive
the real ``AuthorityConnection.dispatch`` so the wire contract, the handler and the persisted row
are all the owner's.
"""
from types import SimpleNamespace

import asyncio
import pytest
import pytest_asyncio

PROFILE_MIN_INTERVAL = 90


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from pathlib import Path

    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    # The state.db the owner serves is the profile home the goal / loop / heartbeat managers write
    # through: one file is both the routing row and the automation row.
    home = Path(store._db.db_path).parent
    (home / 'config.yaml').write_text(
        'model: {default: fake-model, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
        f'loops: {{min_interval_seconds: {PROFILE_MIN_INTERVAL}}}\n'
    )
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    from hermes_cli import goals as _goals
    _goals._DB_CACHE.clear()
    # The real runner class (never constructed): the gateway's own /goal, /loop and /heartbeat
    # handlers run against it, so the legacy admission the owner shares is the production one.
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_db = store._db
    runner._async_session_store = AsyncSessionStore(store)
    runner.adapters = {}
    runner._draining = False
    runner._queued_events = {}
    runner._cached_agent_for = lambda route: None
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._adapters_for_profile = lambda profile: runner.adapters
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='owner')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def create_session(request_id='automation'):
        reply = await connection.dispatch({'id': 0, 'method': 'session.create', 'params': {
            'request_id': request_id, 'source': 'tui', 'cwd': str(tmp_path)}})
        assert 'result' in reply, reply
        return reply['result']['session_id']

    sid = await create_session()

    async def call(method, **params):
        return await connection.dispatch({'id': 1, 'method': method, 'params': params})

    async def result(method, **params):
        reply = await call(method, **params)
        assert 'result' in reply, reply
        return reply['result']

    async def refusal(method, **params):
        reply = await call(method, **params)
        assert 'error' in reply, reply
        return reply['error']

    async def snapshot(**params):
        return (await result('session.control.read', session_id=sid, **params))['control']

    from gateway.session_ancillary import physical_target
    from gateway.session_contract import SessionRef
    target = physical_target(authority, SessionRef(str(home), sid))

    def rows():
        return {kind: authority.db.get_meta(f'{kind}:{target}') for kind in ('goal', 'loop', 'heartbeat')}

    yield SimpleNamespace(authority=authority, connection=connection, home=home, sid=sid, target=target,
                          call=call, result=result, refusal=refusal, snapshot=snapshot, rows=rows,
                          create_session=create_session)
    from hermes_cli.goals import GoalManager
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_cli.loops import LoopManager
    GoalManager(target).clear()
    LoopManager(target).clear()
    HeartbeatManager(target).clear()
    from hermes_cli import goals
    goals._DB_CACHE.clear()


def _seed(owner, kind, state):
    owner.authority.db.set_meta(f'{kind}:{owner.target}', state.to_json())


@pytest.mark.asyncio
async def test_control_read_answers_the_negotiated_loop_minimum(owner):
    control = await owner.snapshot(include_loop_min_interval=True)
    assert control['loop_min_interval_seconds'] == PROFILE_MIN_INTERVAL, control


@pytest.mark.asyncio
async def test_control_read_keeps_the_stable_shape_without_the_opt_in(owner):
    control = await owner.snapshot()
    assert 'loop_min_interval_seconds' not in control, control


@pytest.mark.asyncio
async def test_a_pause_action_answers_the_negotiated_loop_minimum(owner):
    from hermes_cli.loops import LoopState

    _seed(owner, 'loop', LoopState('PAUSE_ME', status='active', interval_seconds=PROFILE_MIN_INTERVAL,
                                   created_at=10, next_due_at=10))

    result = await owner.result('session.control', session_id=owner.sid, action='loop.pause', args={},
                                include_loop_min_interval=True)

    assert result['control']['loop']['status'] == 'paused'
    assert result['control']['loop_min_interval_seconds'] == PROFILE_MIN_INTERVAL, result


@pytest.mark.asyncio
async def test_goal_create_persists_the_goal_with_its_criteria(owner):
    result = await owner.result(
        'session.control', session_id=owner.sid, action='goal.create',
        args={'prompt': 'ship the owner path', 'criteria': ['tests pass', 'no sidecar'],
              'max_turns': 7})

    assert result['dispatch']['type'] == 'send', result
    assert result['dispatch']['message'], result
    goal = result['control']['goal']
    assert (goal['title'], goal['status'], goal['max_turns']) == ('ship the owner path', 'active', 7), goal
    assert goal['subgoals'] == ['tests pass', 'no sidecar'], goal


@pytest.mark.asyncio
async def test_goal_create_refuses_a_second_goal(owner):
    await owner.result('session.control', session_id=owner.sid, action='goal.create',
                       args={'prompt': 'first'})

    error = await owner.refusal('session.control', session_id=owner.sid, action='goal.create',
                                args={'prompt': 'second'})

    assert error['message'] == 'automation_exists', error
    assert (await owner.snapshot())['goal']['title'] == 'first'


@pytest.mark.asyncio
async def test_loop_create_persists_the_interval_and_its_route(owner):
    result = await owner.result(
        'session.control', session_id=owner.sid, action='loop.create',
        args={'prompt': 'check the queue', 'interval_seconds': PROFILE_MIN_INTERVAL * 2,
              'run_limit': 3, 'stop_condition': 'queue empty'})

    loop = result['control']['loop']
    assert (loop['prompt'], loop['interval_seconds'], loop['times'], loop['until']) == (
        'check the queue', float(PROFILE_MIN_INTERVAL * 2), 3, 'queue empty'), loop
    from hermes_cli.loops import LoopState
    persisted = LoopState.from_json(owner.rows()['loop'])
    assert persisted.route.get('platform') == 'local' and persisted.route.get('chat_id') == owner.sid, persisted.route


@pytest.mark.asyncio
async def test_loop_create_refuses_an_interval_below_the_profile_floor(owner):
    error = await owner.refusal('session.control', session_id=owner.sid, action='loop.create',
                                args={'prompt': 'too eager', 'interval_seconds': PROFILE_MIN_INTERVAL - 1})

    assert (error['message'], error['data'].get('field')) == ('invalid_params', 'interval_seconds'), error
    assert owner.rows()['loop'] is None


@pytest.mark.asyncio
async def test_heartbeat_create_persists_the_interval(owner):
    result = await owner.result('session.control', session_id=owner.sid, action='heartbeat.create',
                                args={'prompt': 'say something useful', 'interval_seconds': 600})

    heartbeat = result['control']['heartbeat']
    assert (heartbeat['prompt'], heartbeat['interval_seconds'], heartbeat['status']) == (
        'say something useful', 600, 'active'), heartbeat


@pytest.mark.asyncio
async def test_heartbeat_create_refuses_an_interval_below_its_floor(owner):
    error = await owner.refusal('session.control', session_id=owner.sid, action='heartbeat.create',
                                args={'prompt': 'too eager', 'interval_seconds': 5})

    assert (error['message'], error['data'].get('field')) == ('invalid_params', 'interval_seconds'), error
    assert owner.rows()['heartbeat'] is None


@pytest.mark.asyncio
async def test_goal_update_rewrites_prompt_and_criteria_in_place(owner):
    created = await owner.result('session.control', session_id=owner.sid, action='goal.create',
                                 args={'prompt': 'old goal', 'criteria': ['old criterion'], 'max_turns': 9})
    turns_used = created['control']['goal']['turns_used']

    result = await owner.result('session.control', session_id=owner.sid, action='goal.update',
                                args={'prompt': 'new goal', 'criteria': ['new criterion'], 'max_turns': 12})

    goal = result['control']['goal']
    assert (goal['title'], goal['subgoals'], goal['max_turns']) == (
        'new goal', ['new criterion'], 12), goal
    assert goal['status'] == 'active' and goal['turns_used'] == turns_used, goal


@pytest.mark.asyncio
async def test_update_of_a_missing_automation_is_refused(owner):
    error = await owner.refusal('session.control', session_id=owner.sid, action='heartbeat.update',
                                args={'prompt': 'nothing there', 'interval_seconds': 600})

    assert error['message'] == 'automation_missing', error
    assert owner.rows()['heartbeat'] is None


@pytest.mark.asyncio
async def test_concurrent_creators_admit_one_goal(owner, monkeypatch):
    import threading

    from hermes_cli import goals as goals_mod

    # Both creators finish their existing-goal check before either writes: without one admission
    # for the session the second replaces the first one's goal (and its criteria).
    both_checked = threading.Barrier(2, timeout=1.0)
    real_set = goals_mod.GoalManager.set

    def racing_set(manager, *args, **kwargs):
        try:
            both_checked.wait()
        except threading.BrokenBarrierError:
            pass
        return real_set(manager, *args, **kwargs)

    monkeypatch.setattr(goals_mod.GoalManager, 'set', racing_set)
    replies = await asyncio.gather(*(
        owner.call('session.control', session_id=owner.sid, action='goal.create',
                   args={'prompt': f'creator {index}', 'criteria': [f'criterion {index}']})
        for index in range(2)))

    created = [reply['result'] for reply in replies if 'result' in reply]
    refused = [reply['error']['message'] for reply in replies if 'error' in reply]
    assert len(created) == 1 and refused == ['automation_exists'], replies
    winner = created[0]['control']['goal']
    assert winner['subgoals'] == [winner['title'].replace('creator', 'criterion')], winner


@pytest.mark.asyncio
async def test_a_create_waits_for_the_slash_goal_admission_lock(owner):
    import threading

    from hermes_cli.goals import goal_admission_lock

    lock = goal_admission_lock(owner.target)
    held, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            held.set()
            release.wait(5)
    holder = threading.Thread(target=hold, name='admission-holder')
    holder.start()
    try:
        assert held.wait(5)
        task = asyncio.create_task(owner.call('session.control', session_id=owner.sid, action='goal.create',
                                              args={'prompt': 'queued behind /goal'}))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        assert owner.rows()['goal'] is None
        release.set()
        reply = await asyncio.wait_for(task, timeout=10)
        assert 'result' in reply, reply
    finally:
        release.set()
        holder.join(5)
    assert owner.rows()['goal'] is not None


@pytest.mark.asyncio
async def test_the_slash_goal_command_reads_what_the_composer_created(owner):
    from gateway.session_contract import SessionRef
    from gateway.session_commands import execute_command

    await owner.result('session.control', session_id=owner.sid, action='goal.create',
                       args={'prompt': 'one goal, one row'})
    ref = SessionRef(str(owner.home), owner.sid)

    output = await execute_command(owner.connection, ref,
                                   {'session_id': owner.sid, 'name': 'goal', 'arg': 'status'}, dispatch=True)

    assert 'one goal, one row' in output['output'], output


@pytest.mark.asyncio
async def test_the_slash_goal_command_owns_the_row_a_composer_create_sees(owner):
    from gateway.session_contract import SessionRef
    from gateway.session_commands import execute_command

    ref = SessionRef(str(owner.home), owner.sid)
    await execute_command(owner.connection, ref, {'session_id': owner.sid, 'name': 'goal', 'arg': 'from the slash'},
                          dispatch=True)

    error = await owner.refusal('session.control', session_id=owner.sid, action='goal.create',
                                args={'prompt': 'from the composer'})

    assert error['message'] == 'automation_exists', error
    assert (await owner.snapshot())['goal']['title'] == 'from the slash'


@pytest.mark.asyncio
async def test_a_create_still_refuses_an_unknown_argument(owner):
    error = await owner.refusal('session.control', session_id=owner.sid, action='goal.create',
                                args={'prompt': 'x', 'model': 'sneaky'})

    assert error['message'] == 'invalid_params', error
    assert owner.rows()['goal'] is None


@pytest.mark.asyncio
async def test_a_read_still_refuses_an_unknown_argument(owner):
    error = await owner.refusal('session.control.read', session_id=owner.sid, bogus=True)

    assert error['message'] == 'invalid_params', error


@pytest.mark.asyncio
async def test_a_creator_without_the_control_capability_is_refused(owner):
    from gateway.session_controls import AuthorityConnection

    viewer = AuthorityConnection(owner.authority, object(), {
        'user_id': 'human', 'capabilities': ['session:read']})
    reply = await viewer.dispatch({'id': 2, 'method': 'session.control', 'params': {
        'session_id': owner.sid, 'action': 'goal.create', 'args': {'prompt': 'not mine'}}})

    assert reply['error']['message'] == 'permission_denied', reply
    assert owner.rows()['goal'] is None


@pytest.mark.asyncio
async def test_an_unpersisted_create_is_not_reported_as_success(owner, monkeypatch):
    from hermes_cli import goals as goals_mod

    monkeypatch.setattr(goals_mod, 'save_goal', lambda *args, **kwargs: None)

    error = await owner.refusal('session.control', session_id=owner.sid, action='goal.create',
                                args={'prompt': 'lost on the way to the row'})

    assert error['message'] == 'storage_unavailable', error
    assert owner.rows()['goal'] is None


@pytest.mark.asyncio
async def test_a_created_heartbeat_joins_the_gateway_poller(owner):
    await owner.result('session.control', session_id=owner.sid, action='heartbeat.create',
                       args={'prompt': 'ping', 'interval_seconds': 600})

    live = owner.authority.sessions[owner.sid]
    key = owner.authority.runner._session_key_for_source(live.source)
    assert owner.authority.runner._heartbeat_watch[key] == (live.source, owner.target)


@pytest.mark.asyncio
async def test_the_broadcast_never_carries_the_negotiated_minimum(owner):
    frames = []
    live = owner.authority.sessions[owner.sid]
    live.event_stream.observers.add(lambda frame: frames.append(frame) if frame['params']['type'] ==
                                    'session.control.update' else None)
    try:
        await owner.result('session.control', session_id=owner.sid, action='heartbeat.create',
                           args={'prompt': 'ping', 'interval_seconds': 600}, include_loop_min_interval=True)
    finally:
        live.event_stream.observers.clear()

    assert frames, 'the create must repaint the other viewers'
    control = frames[-1]['params']['payload']['control']
    assert 'loop_min_interval_seconds' not in control, control
    assert control['heartbeat']['prompt'] == 'ping'


