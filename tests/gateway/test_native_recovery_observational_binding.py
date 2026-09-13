"""Static native admission metadata, never restart, claim, drain or resolution."""
from copy import copy
import json
import weakref

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_admission import admission_fingerprint
from gateway.session_authorities import owner_scope
from gateway.session_authority import LiveSession
from gateway.session_contract import Principal, SessionRef
from gateway.session_envelope import snapshot_native
from gateway.session_ingress_context import native_callback
from tests.gateway.test_native_reconnect_recovery import state  # noqa: F401


def stored_native(state, status):
    authority = state.authorities['alpha']
    runner = state.runner
    adapter = runner._profile_adapters['alpha'][Platform.TELEGRAM]
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='saved-native', chat_type='dm',
                           user_id='allowed', profile='alpha')
    source._transport_adapter_ref = weakref.ref(adapter)
    event = MessageEvent(text='retained input', message_id='stored-input', source=source)
    with owner_scope(authority), native_callback(runner, event, state.homes['alpha'], 'alpha'):
        payload = snapshot_native(runner, event)
        entry = runner.session_store.get_or_create_session(source)
        sid = entry.session_id
        if authority.db.get_session(sid) is None:
            authority.db.create_session(sid, source='telegram')
        digest = admission_fingerprint(canonical_target=sid, payload={'input': payload, 'intent': 'queue'})
        # Seed the already-persisted state directly, not by starting/losing work.
        authority.db._execute_write(lambda conn: conn.execute('''INSERT INTO session_admissions(
            admission_id,request_id,principal_id,target_session_id,lineage_json,
            payload_json,payload_digest,intent,status,owner_epoch,generation)
            VALUES(?,?,?,?,?,?,?,'queue',?,?,1)''',
            ('stored-admission', event.message_id, 'fixture-owner', sid, json.dumps([sid]),
             json.dumps(payload), digest, status, authority.epoch)))
    return authority, sid, source, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['unknown', 'started'])
async def test_cold_native_metadata_gets_read_view_without_rearming_or_writing(state, status):
    authority, sid, source, adapter = stored_native(state, status)
    assert sid not in authority.sessions
    before = list(authority.db._conn.iterdump())
    actor = Principal('reader', authority.profile_id, frozenset({'session:read'}), 'viewer')
    with owner_scope(authority):
        result = await authority.recover_native_sessions([(sid, source, adapter)])
        assert result[sid] == ('unknown_execution' if status == 'unknown' else 'active')
        # Without the native view, authorize would try the absent API binding.
        handle = await authority.resolve(actor, SessionRef(authority.profile_id, sid))
    assert handle.execution_state == ('unknown' if status == 'unknown' else 'running')
    assert state.scheduled == []
    assert authority.sessions[sid].task is None
    assert state.runner._adapter_for_source(authority.sessions[sid].source) is adapter
    assert list(authority.db._conn.iterdump()) == before


@pytest.mark.asyncio
async def test_existing_unknown_view_is_not_refreshed_by_observational_recovery(state):
    authority, sid, source, adapter = stored_native(state, 'unknown')
    prior_source = copy(source)
    live = LiveSession(prior_source, 'existing-view-route')
    authority.sessions[sid] = live
    controls, events, subscribers = live.controls, live.event_stream, live.subscribers
    before = list(authority.db._conn.iterdump())
    with owner_scope(authority):
        assert await authority.recover_native_sessions([(sid, source, adapter)]) == {sid: 'unknown_execution'}
    assert authority.sessions[sid] is live and live.source is prior_source
    assert live.route == 'existing-view-route'
    assert (live.controls, live.event_stream, live.subscribers) == (controls, events, subscribers)
    assert live.task is None and state.scheduled == []
    assert list(authority.db._conn.iterdump()) == before


@pytest.mark.asyncio
async def test_unapproved_sender_cannot_acquire_even_an_observational_binding(state):
    authority, sid, source, adapter = stored_native(state, 'unknown')
    state.runner.allowed.clear()
    before = list(authority.db._conn.iterdump())
    with owner_scope(authority):
        assert await authority.recover_native_sessions([(sid, source, adapter)]) == {sid: 'permission_denied'}
    assert sid not in authority.sessions and state.scheduled == []
    assert list(authority.db._conn.iterdump()) == before
