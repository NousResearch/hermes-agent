"""A producer wake is admitted on the committed route identity, not on the display labels of its origin."""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import RuntimeStoreError

_HUMAN = dict(chat_id='42', chat_type='dm', user_id='42', user_name='P Woz', chat_name='P Woz')


async def _committed_route(tmp_path, monkeypatch):
    from gateway.session_ingress_context import native_callback, register_transport_home
    from plugins.platforms.discord.adapter import DiscordAdapter
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='labels', db=db)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='***', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    human = MessageEvent(text='human', source=adapter.build_source(**_HUMAN), message_id='human-1')
    with native_callback(runner, human, tmp_path):
        await authority.admit_native(human)
    route = runner.session_store._generate_session_key(human.source)
    metadata = {'gateway_session_key': route, 'gateway_session_id': runner.session_store.peek_session_id(route)}
    return authority, adapter, metadata


@pytest.mark.asyncio
@pytest.mark.parametrize('labels', [
    dict(user_name='Filip', chat_name=None),
    dict(user_name=None, chat_name='Some other title', chat_topic='topic'),
])
async def test_wake_from_an_origin_with_other_display_labels_is_admitted(tmp_path, monkeypatch, labels):
    """A session created by a plugin injection (or before the user renamed their profile) keeps an origin
    whose labels differ from the human's committed source; the wake built from that origin must not be
    refused forever (``admission_conflict``): ids, not names, identify the route."""
    authority, adapter, metadata = await _committed_route(tmp_path, monkeypatch)
    try:
        origin = adapter.build_source(**{**_HUMAN, **labels})
        event = MessageEvent(text='notice', source=origin, internal=True, metadata=dict(metadata))
        receipt = await authority.admit_automation(adapter, event, 'notice')
        assert receipt.ref.session_id
        assert (await authority.admit_automation(adapter, event, 'notice')).admission_id == receipt.admission_id
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['user_id', 'chat_id', 'thread_id'])
async def test_wake_from_another_identity_is_still_refused(tmp_path, monkeypatch, field):
    authority, adapter, metadata = await _committed_route(tmp_path, monkeypatch)
    try:
        origin = adapter.build_source(**{**_HUMAN, field: '7'})
        event = MessageEvent(text='notice', source=origin, internal=True, metadata=dict(metadata))
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await authority.admit_automation(adapter, event, 'notice')
    finally:
        authority.db.close()
