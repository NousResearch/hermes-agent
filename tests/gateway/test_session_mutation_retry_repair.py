"""An exact mutation retry reconciles the runtime projection the failed first attempt left behind."""
from types import SimpleNamespace

import pytest
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_exact_retry_repairs_runtime_after_post_commit_failure_without_second_event(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        seed_messages(db, 's', {'role': 'user', 'content': 'keep'}, {'role': 'assistant', 'content': 'drop'},
                      {'role': 'user', 'content': 'drop too'})
        epoch = db.runtime.begin_epoch(instance_id='owner')
        evictions = []

        def evict(route):
            evictions.append(route)
            if len(evictions) == 1:
                # The transcript rewind is already committed; the still-running process
                # fails while repairing its cached agent.
                raise RuntimeError('injected post-commit failure')

        authority = SessionAuthority(SimpleNamespace(_draining=False, _evict_cached_agent=evict),
                                     profile_id='owned', instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        live = authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        watermark = live.event_stream.watermark()
        request = {'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's',
            'request_id': 'rewind', 'expected_revision': 0, 'expected_generation': 0,
            'operation': 'rewind', 'payload': {'target_message_id': transcript_rows(db, 's')[2]['id']}}}
        try:
            with pytest.raises(RuntimeError, match='injected post-commit failure'):
                await owner.dispatch(request)
            assert [m['content'] for m in transcript_rows(db, 's')] == ['keep', 'drop'], 'rewind did not commit'
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

    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        seed_messages(db, 's', {'role': 'user', 'content': 'keep'}, {'role': 'assistant', 'content': 'drop'},
                      {'role': 'user', 'content': 'drop too'})
        epoch = db.runtime.begin_epoch(instance_id='owner')
        runner = object.__new__(GatewayRunner)
        runner._agent_cache, runner._agent_cache_lock, runner._running_agents = OrderedDict(), threading.Lock(), {}
        runner._draining = False
        authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        request = {'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's',
            'request_id': 'rewind', 'expected_revision': 0, 'expected_generation': 0,
            'operation': 'rewind', 'payload': {'target_message_id': transcript_rows(db, 's')[2]['id']}}}
        try:
            first = await owner.dispatch(request)
            db.admissions.admit(epoch=epoch, principal_id='human', session_id='s',
                                request_id='later', payload={'text': 'later'})
            row = db.admissions.claim(epoch=epoch, session_id='s')
            agent = Agent()
            runner._agent_cache['route'] = (agent, 'signature')
            runner._running_agents['route'] = agent
            authority.adopt_agent('s', row.generation, agent)
            assert (await owner.dispatch(request))['result'] == first['result']
            ref = SessionRef('owned', 's')
            assert authority.agent(ref) is agent, 'rewind replay evicted the running successor agent'
            await authority.interrupt(owner.actor, ref, row.generation)
            assert Agent.interrupts == 1 and authority.pending_stops == {}
        finally:
            await owner.close()
