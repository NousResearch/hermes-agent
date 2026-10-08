"""Background and room producers retain source ownership across media and transcript lifetimes."""






import pytest



@pytest.mark.asyncio
async def test_automation_after_compression_and_released_media_uses_logical_owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.session_ingress_context import native_callback
    from gateway.session_ingress_media import release_admission_media
    from gateway.session_automation import check_automation_route
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input, settle_session_input, list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    photo = tmp_path / 'photo.png'
    photo.write_bytes(b'photo')
    human = MessageEvent(text='human', source=source, message_id='human',
                         media_urls=[str(photo)], media_types=['image/png'])
    with native_callback(runner, human, get_hermes_home()):
        receipt = await authority.admit_native(human)
    sid = receipt.ref.session_id
    claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id,
                         generation=claim['generation'], outcome='completed')
    assert release_admission_media(authority.db, receipt.admission_id) == 1
    authority.db.publish_compression_child(parent_session_id=sid, child_session_id='child', source='discord',
        messages=[{'role': 'assistant', 'content': 'summary'}], require_compression_lease=False)
    route = runner.session_store._generate_session_key(source)
    runner.session_store.advance_compression_session(route, sid, 'child')
    event = MessageEvent(text='finished', source=source, internal=True,
        metadata={'gateway_session_key': route, 'gateway_session_id': sid})
    admitted = await authority.admit_automation(adapter, event, 'completion')
    assert admitted.ref.session_id == sid
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert check_automation_route(runner, rows[-1]['payload'], 'child', source, adapter)[1] == route
    assert (await authority.admit_automation(adapter, event, 'completion')).admission_id == admitted.admission_id
