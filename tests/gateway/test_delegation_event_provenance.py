"""Explicit delegation identity survives canonical admission, never granting authority."""
import json

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.response_filters import display_kind_for_event, display_metadata_for_event
from gateway.run_notifications import _watch_notification_metadata
from gateway.session import SessionSource
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.test_automation_notification_category import _authority, _native_owner


ID = 'deleg_0123456789abcdef0123456789abcdef'
ENVELOPE = dict(event_schema='hermes.internal_event.v1', event_id=f'async_delegation:{ID}:terminal',
                event_kind='workflow.async_delegation.terminal', workflow_id=f'delegation:{ID}',
                display_kind='internal_event', user_originated=False, terminal=True)
PRIVATE_ENVELOPE = dict(ENVELOPE, delegation_id=ID)
PROJECTION = {key: value for key, value in PRIVATE_ENVELOPE.items() if key != 'display_kind'}


def _completion():
    return dict(ENVELOPE, type='async_delegation', delegation_id=ID, status='completed', summary='result')


def test_producer_projection_retains_route_and_delivery_proof():
    event = dict(_completion(), session_key='agent:main:local:owner', parent_session_id='parent',
                 proven_session_id='tip', completion_ids=[ID], _automation_identities=['sibling'])
    metadata = _watch_notification_metadata(event, SessionSource(platform=Platform.LOCAL, chat_id='owner'))
    assert metadata.get('delegation_event') == PRIVATE_ENVELOPE
    assert metadata['gateway_session_key'] == event['session_key']
    assert metadata['gateway_session_id'] == 'parent'
    assert metadata['gateway_proven_session_id'] == 'tip'
    assert metadata['gateway_completion_ids'] == [ID]
    assert metadata['automation_identities'] == ['sibling']
    assert metadata['display_kind'] == 'async_delegation_complete'  # main's closed card set


@pytest.mark.parametrize('internal', [False, True])
def test_projection_is_not_inferred_from_user_text(internal):
    event = MessageEvent(text=json.dumps(PROJECTION), internal=internal)
    assert display_metadata_for_event(event) == {}
    assert display_kind_for_event(event) == ('internal_notification' if internal else None)
    if not internal:
        event.metadata['delegation_event'] = dict(PRIVATE_ENVELOPE)
        assert display_metadata_for_event(event) == {}
        assert display_kind_for_event(event) is None


@pytest.mark.parametrize('change', [{'event_id': 'another-unit'}, {'terminal': 1}, {'user_originated': 0},
                                     {'gateway_session_id': 'other'}, {'event_type': 'completion'}])
def test_invalid_private_projection_refuses_without_widening_card_set(change):
    from gateway.session_automation import automation_display_metadata
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        automation_display_metadata({'delegation_event': dict(PRIVATE_ENVELOPE, **change)})
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        automation_display_metadata({'display_kind': 'internal_event', 'display_text': 'spoof'})


@pytest.mark.asyncio
async def test_native_projection_is_durable_without_changing_route_authority(tmp_path, monkeypatch):
    from gateway.session_envelope import restore_native
    runner, authority, adapter, source, route_metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    try:
        metadata = _watch_notification_metadata(dict(_completion(), session_key=route_metadata['gateway_session_key'],
                                                     parent_session_id=route_metadata['gateway_session_id']), source)
        assert metadata.get('delegation_event') == PRIVATE_ENVELOPE
        event = MessageEvent(text='result', source=source, internal=True, metadata=metadata)
        receipt = await authority.admit_automation(adapter, event, 'delegation-notice')
        assert (await authority.admit_automation(adapter, event, 'delegation-notice')).admission_id == receipt.admission_id
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        restored = restore_native(row['payload'], runner)
        assert display_kind_for_event(restored) == 'internal_event'
        assert display_metadata_for_event(restored) == PROJECTION
        assert restored.source.message_id is None
        event.metadata['gateway_session_id'] = 'foreign'
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await authority.admit_automation(adapter, event, 'foreign')
        event.internal = False
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            await authority.admit_automation(adapter, event, 'user')
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_unviewed_local_note_retains_identity_and_hidden_intent_stays_hidden(tmp_path, monkeypatch):
    from gateway.session_completion_notes import note_if_unviewed
    from gateway.session_contract import Principal
    from gateway.session_display import display_turn_scope
    from gateway.session_local import create_local_session
    from gateway.session_automation import restore_local_automation
    runner, authority = await _authority(tmp_path, monkeypatch)
    try:
        actor = Principal('owner', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'fixture')
        ref = create_local_session(authority, actor, dict(request_id='local', source='cli', model='fixture', toolsets=[]))
        live = authority.sessions[ref.session_id]
        sid = runner.session_store.peek_session_id(live.route)
        evt = dict(_completion(), session_key=live.route, parent_session_id=sid)
        metadata = _watch_notification_metadata(evt, live.source)
        assert metadata.get('delegation_event') == PRIVATE_ENVELOPE
        identity = json.dumps(['async_delegation', ID])
        event = MessageEvent(text='result', source=live.source, internal=True, metadata=metadata)
        adapter = runner._adapter_for_source(live.source)
        receipt = await authority.admit_automation(adapter, event, identity)
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        restored = restore_local_automation(authority, ref, row)
        assert display_kind_for_event(restored) == 'internal_event'
        assert display_metadata_for_event(restored)['event_id'] == PROJECTION['event_id']
        with display_turn_scope(restored.message_id, {'kind': 'hidden'}):
            assert display_kind_for_event(restored) == 'hidden'
        assert await note_if_unviewed(authority, ref, row) == ''
        rows = authority.db.get_messages_as_conversation(sid)
        assert rows[-1]['display_kind'] == 'internal_event'
        assert rows[-1]['display_metadata']['event_id'] == PROJECTION['event_id']
    finally:
        authority.db.close()


def test_interim_batch_failure_never_becomes_a_terminal_envelope():
    from tools.process_registry_notifications import TimelineNotification
    event = dict(_completion(), task_failure_notice=True, status='running')
    notice = TimelineNotification.for_delegation('still running', event)
    assert notice.display_kind == 'async_delegation_complete'
    assert 'event_id' not in notice.display_metadata
