"""Shared mutation publication and live-state coordination."""
from types import SimpleNamespace
import asyncio
import pytest
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_metadata_mutation_publishes_once_while_work_is_queued(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned', instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        live = authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        db.admissions.admit(epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'keep'})
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
            assert db.admissions.list(session_id='s')[0].status == 'queued'
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_rewind_refuses_live_task_even_before_admission_claim(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        seed_messages(db, 's', {'role': 'user', 'content': 'keep'})
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False, _evict_cached_agent=lambda route: None),
                                     profile_id='owned', instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        # Admitted but not yet claimed: the ledger row alone must fence the rewind.
        db.admissions.admit(epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'go'})
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        try:
            response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's',
                'request_id': 'rewind', 'expected_revision': 0, 'expected_generation': 0,
                'operation': 'rewind', 'payload': {'target_message_id': transcript_rows(db, 's')[0]['id']}}})
            assert response['error']['message'] == 'session_busy'
            assert transcript_rows(db, 's')[0]['content'] == 'keep'
        finally:
            await owner.close()
