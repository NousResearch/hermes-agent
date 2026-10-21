"""Shared mutation publication and live-state coordination."""
from types import SimpleNamespace
import asyncio
import pytest
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
import hermes_state_runtime as rt


@pytest.mark.asyncio
async def test_metadata_mutation_publishes_once_while_work_is_queued(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned', instance_id='owner', db=db, epoch=epoch)
        live = authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'keep'})
        watermark = live.event_stream.watermark()
        request = {'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's', 'request_id': 'edit',
            'expected_revision': 0, 'operation': 'sidebar', 'payload': {'title': 'Title', 'pinned': True}}}
        try:
            first = await owner.dispatch(request)
            assert first['result']['revision'] == 1
            events = live.event_stream.since(*watermark)['events']
            assert len(events) == 1 and events[0]['type'] == 'session.updated'
            assert events[0]['payload'] == first['result']
            assert await owner.dispatch(request) == first
            assert live.event_stream.since(*watermark)['events'] == events
            assert rt.list_session_admissions(db, session_id='s')[0]['status'] == 'queued'
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_rewind_refuses_live_task_even_before_admission_claim(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_message('s', role='user', content='keep')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False, _evict_cached_agent=lambda route: None),
                                     profile_id='owned', instance_id='owner', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        # Admitted but not yet claimed: the ledger row alone must fence the rewind.
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'go'})
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        try:
            response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's',
                'request_id': 'rewind', 'expected_revision': 0, 'expected_generation': 0,
                'operation': 'rewind', 'payload': {'target_message_id': db.get_messages('s')[0]['id']}}})
            assert response['error']['message'] == 'session_busy'
            assert db.get_messages('s')[0]['content'] == 'keep'
        finally:
            await owner.close()
