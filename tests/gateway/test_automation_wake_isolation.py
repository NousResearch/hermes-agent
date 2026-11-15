"""Reset wakes use the same platform isolation policy as their real session keys."""
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.test_automation_notification_category import _SOURCE, _native_owner


@pytest.mark.asyncio
@pytest.mark.parametrize('thread_id', [None, 'topic'])
async def test_reset_wake_keeps_a_per_user_platform_route(tmp_path, monkeypatch, thread_id):
    fields = {**_SOURCE, 'chat_type': 'group', 'thread_id': thread_id}
    runner, authority, adapter, source, metadata = await _native_owner(
        tmp_path, monkeypatch, 'human', fields,
        group_sessions_per_user=False, thread_sessions_per_user=False,
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, extra={
            'group_sessions_per_user': True, 'thread_sessions_per_user': True})})
    try:
        route = metadata['gateway_session_key']
        fresh = runner.session_store.reset_session(route)
        assert fresh is not None
        event = MessageEvent(text='done', source=source, internal=True, metadata={
            'gateway_session_key': route, 'gateway_session_id': fresh.session_id})
        receipt = await authority.admit_automation(adapter, event, 'done')
        assert receipt.ref.session_id == fresh.session_id
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert row is not None
        assert row['payload']['native_text_v1']['source']['user_id'] == source.user_id
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('thread_id', [None, 'topic'])
async def test_reset_wake_requires_its_requester_on_a_shared_platform_route(tmp_path, monkeypatch, thread_id):
    fields = {**_SOURCE, 'chat_type': 'group', 'thread_id': thread_id}
    runner, authority, adapter, source, metadata = await _native_owner(
        tmp_path, monkeypatch, 'human', fields,
        group_sessions_per_user=True, thread_sessions_per_user=True,
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, extra={
            'group_sessions_per_user': False, 'thread_sessions_per_user': False})})
    try:
        route = metadata['gateway_session_key']
        fresh = runner.session_store.reset_session(route)
        assert fresh is not None
        wake_metadata = {'gateway_session_key': route, 'gateway_session_id': fresh.session_id}
        anonymous = MessageEvent(text='done', source=source, internal=True, metadata=dict(wake_metadata))
        with pytest.raises(RuntimeStoreError, match='not_found'):
            await authority.admit_automation(adapter, anonymous, 'done')
        other = MessageEvent(text='done', source=adapter.build_source(**{**fields, 'user_id': '7'}),
                             internal=True, metadata={**wake_metadata, 'requester_user_id': '7'})
        with pytest.raises(RuntimeStoreError, match='not_found'):
            await authority.admit_automation(adapter, other, 'done')
        own = MessageEvent(text='done', source=source, internal=True,
                           metadata={**wake_metadata, 'requester_user_id': source.user_id})
        receipt = await authority.admit_automation(adapter, own, 'done')
        assert receipt.ref.session_id == fresh.session_id
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert row is not None
        assert row['payload']['native_text_v1']['source']['user_id'] == source.user_id
    finally:
        authority.db.close()
