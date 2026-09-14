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
