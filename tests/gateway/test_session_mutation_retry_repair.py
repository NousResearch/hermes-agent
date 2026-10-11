"""An exact mutation retry reconciles the runtime projection the failed first attempt left behind."""
from types import SimpleNamespace

import pytest
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
import hermes_state_runtime as rt


@pytest.mark.asyncio
async def test_exact_retry_repairs_runtime_after_post_commit_failure_without_second_event(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_message('s', role='user', content='keep')
        db.append_message('s', role='assistant', content='drop')
        db.append_message('s', role='user', content='drop too')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        evictions = []

        def evict(route):
            evictions.append(route)
            if len(evictions) == 1:
                # The transcript rewind is already committed; the still-running process
                # fails while repairing its cached agent.
                raise RuntimeError('injected post-commit failure')

        authority = SessionAuthority(SimpleNamespace(_draining=False, _evict_cached_agent=evict),
                                     profile_id='owned', instance_id='owner', db=db, epoch=epoch)
        live = authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        watermark = live.event_stream.watermark()
        request = {'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's',
            'request_id': 'rewind', 'expected_revision': 0, 'expected_generation': 0,
            'operation': 'rewind', 'payload': {'target_message_id': db.get_messages('s')[2]['id']}}}
        try:
            with pytest.raises(RuntimeError, match='injected post-commit failure'):
                await owner.dispatch(request)
            assert [m['content'] for m in db.get_messages('s')] == ['keep', 'drop'], 'rewind did not commit'
            assert live.event_stream.since(*watermark)['events'] == [], 'a failed attempt published its event'
            retried = await owner.dispatch(request)
            assert retried['result']['rewound_count'] == 1
            assert evictions == ['route', 'route'], 'exact retry skipped the runtime repair'
            assert live.event_stream.since(*watermark)['events'] == [], 'exact retry re-emitted the one-shot event'
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_rewind_replay_never_evicts_a_later_admissions_running_agent(tmp_path):
    """R3: an exact rewind retry after a later admission claimed must leave that turn's adopted
    agent in the real cache, so the interrupt RPC delivers Stop instead of latching it."""
    import threading
    from collections import OrderedDict
    from gateway.run import GatewayRunner
    from gateway.session_contract import SessionRef

    class Agent:
        interrupts = 0

        def interrupt(self, *args, **kwargs):
            Agent.interrupts += 1

    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_message('s', role='user', content='keep')
        db.append_message('s', role='assistant', content='drop')
        db.append_message('s', role='user', content='drop too')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        runner = object.__new__(GatewayRunner)
        runner._agent_cache, runner._agent_cache_lock, runner._running_agents = OrderedDict(), threading.Lock(), {}
        runner._draining = False
        authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        request = {'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's',
            'request_id': 'rewind', 'expected_revision': 0, 'expected_generation': 0,
            'operation': 'rewind', 'payload': {'target_message_id': db.get_messages('s')[2]['id']}}}
        try:
            first = await owner.dispatch(request)
            rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                                   request_id='later', payload={'text': 'later'})
            row = rt.claim_session_input(db, epoch=epoch, session_id='s')
            agent = Agent()
            runner._agent_cache['route'] = (agent, 'signature')
            runner._running_agents['route'] = agent
            authority.adopt_agent('s', row['generation'], agent)
            assert (await owner.dispatch(request))['result'] == first['result']
            ref = SessionRef('owned', 's')
            assert authority.agent(ref) is agent, 'rewind replay evicted the running successor agent'
            await authority.interrupt(owner.actor, ref, row['generation'])
            assert Agent.interrupts == 1 and authority.pending_stops == {}
        finally:
            await owner.close()
