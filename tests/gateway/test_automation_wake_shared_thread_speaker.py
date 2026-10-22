"""A pinned producer wake belongs to the thread session, not its first speaker.

Run with scripts/run_tests.sh. No Discord connection or model execution is needed;
only FIFO scheduling is disabled, so admission and its durable envelope stay real.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_automation import committed_route_envelope
from gateway.session_ingress_context import native_callback, register_transport_home
from hermes_state import SessionDB
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter


async def _authority(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = db
    authority = await initialize_session_authority(
        runner, profile_id='default', instance_id='shared-thread-fixture', db=db)
    return runner, authority


async def _native_owner(tmp_path, monkeypatch):
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '200')
    runner, authority = await _authority(tmp_path, monkeypatch)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    thread = dict(chat_id='42', chat_type='thread', thread_id='43')
    # The first speaker seeds the route. Later speakers must not rewrite its origin.
    first = adapter.build_source(**thread, user_id='100', user_name='dispatcher')
    entry = runner.session_store.get_or_create_session(first)
    human = MessageEvent(
        text='Run a background job', message_id='human-1',
        source=adapter.build_source(**thread, user_id='200', user_name='human', message_id='human-1'))
    with native_callback(runner, human, tmp_path):
        await authority.admit_native(human)
    assert runner.session_store._generate_session_key(human.source) == entry.session_key
    assert entry.origin.user_id == '100'
    assert committed_route_envelope(authority, entry)['source']['user_id'] == '200'
    assert not any(row['payload']['native_text_v1'].get('automation')
                   for row in list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False))
    return runner, authority, entry


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['completion', 'async_delegation'])
async def test_pinned_thread_wake_admits_after_speaker_changes(tmp_path, monkeypatch, kind):
    runner, authority, entry = await _native_owner(tmp_path, monkeypatch)
    event = dict(type=kind, session_id='proc_fixture', delegation_id='deleg_fixture',
                 session_key=entry.session_key, parent_session_id=entry.session_id,
                 platform='discord', chat_id='42', chat_type='thread', thread_id='43',
                 exit_code=0, output='done', started_at=1.0)
    try:
        committed = committed_route_envelope(authority, entry)
        assert await runner._inject_watch_notification('[done]', event, raise_not_accepted=True) is True
        owner = authority.logical_owner(entry.session_id)
        rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
        wakes = [row for row in rows if row['payload']['native_text_v1'].get('automation')]
        assert len(wakes) == 1
        envelope = wakes[0]['payload']['native_text_v1']
        assert envelope['route'] == entry.session_key
        assert envelope['automation']['owner'] == owner
        assert envelope['source']['user_id'] == '200'
        assert envelope['provenance'] == committed['provenance']
        assert 'message_id' not in envelope['source']
        assert await runner._inject_watch_notification('[done]', event, raise_not_accepted=True) is True
        repeated = list_session_admissions(authority.db, session_id=owner, pending_only=False)
        assert [row['admission_id'] for row in repeated] == [row['admission_id'] for row in rows]
    finally:
        authority.db.close()
